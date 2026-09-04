"""
Ask a model on OpenRouter and read its answer back as data.

The pipeline asks models several different questions -- which class a label means, which
of two overlapping objects holds the other -- and they differ only in what is said and
what pictures come with it. This holds what they have in common: building the message,
surviving a rate limit, and getting JSON out of a reply that may be wrapped in prose.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
"""
Where a question is sent.
"""

DEFAULT_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
"""
The model the pipeline asks unless told otherwise, as the extraction stage uses.
"""

RETRIED_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
"""
The answers worth asking again after: a rate limit and the server's own failures.
"""

MAXIMUM_ATTEMPTS = 5
"""
How often one question is asked before its failure is raised.
"""


class ModelRefusedError(RuntimeError):
    """
    Raised when a reply holds no JSON, so there is no answer to read out of it.
    """

    def __init__(self, answer: str):
        """
        :param answer: What the model said instead.
        """
        self.answer = answer
        super().__init__(f"the model answered without JSON in it: {answer[:400]}")


def text_part(text: str) -> Dict[str, Any]:
    """
    :param text: What to say.
    :return: It, as a part of a message.
    """
    return {"type": "text", "text": text}


def image_part(image: Path) -> Dict[str, Any]:
    """
    :param image: The PNG to show.
    :return: It, as a part of a message.
    """
    return rendered_part(Path(image).read_bytes())


def rendered_part(image: bytes) -> Dict[str, Any]:
    """
    :param image: A PNG as it came out of a renderer, never written to disk.
    :return: It, as a part of a message, carried inline rather than by URL so that
        nothing has to be hosted for a model to see it.
    """
    encoded = base64.b64encode(image).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def ask(
    content: Sequence[Dict[str, Any]],
    system: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Put one question to a model.

    :param content: The message, as :func:`text_part` and :func:`image_part` build it.
    :param system: What the model is told it is doing.
    :param model: Which model to ask.
    :param timeout: How long one attempt may take, in seconds.
    :return: The whole response, so that what was answered and what it cost stay
        together in whatever is written to disk.
    :raises RuntimeError: If ``OPENROUTER_API_KEY`` is not set.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set, and asking a model needs it."
        )

    payload = json.dumps(
        {
            "model": model,
            # Every question here has one right answer and is asked once, so there is
            # nothing for sampling to explore. It does not make a run repeatable --
            # expert routing and batching still move under us -- but it removes the
            # variance that is ours to remove, and this pipeline's answers did vary
            # between runs on the borderline questions.
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": list(content)},
            ],
        }
    )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "ai.uni-bremen.de",
        "X-Title": "Uni Bremen",
    }

    for attempt in range(MAXIMUM_ATTEMPTS):
        try:
            response = requests.post(
                url=OPENROUTER_URL, headers=headers, data=payload, timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_failure:
            failure = http_failure
            worth_retrying = response.status_code in RETRIED_STATUS_CODES
            reason = f"returned {response.status_code}"
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as network_failure:
            failure = network_failure
            worth_retrying = True
            reason = f"failed with {type(network_failure).__name__}"

        if not worth_retrying or attempt == MAXIMUM_ATTEMPTS - 1:
            raise failure
        waited = 2**attempt
        print(f"    the request {reason}, asking again in {waited}s ...")
        time.sleep(waited)


def answer_text(response: Dict[str, Any]) -> str:
    """
    :param response: What :func:`ask` returned.
    :return: What the model said, empty where it said nothing -- a reply can carry a
        null content, and a caller reading it as text should get text.
    """
    return response["choices"][0]["message"].get("content") or ""


def parse_json_answer(answer: str) -> Any:
    """
    Read the JSON out of a reply.

    Models asked for JSON answer with it in a fenced block, or with a sentence in front
    of it, often enough that a reply is worth searching rather than only parsed.

    :param answer: What the model said.
    :return: The JSON object or array in it.
    :raises ModelRefusedError: If there is none, an empty reply included: a model that
        answers with nothing has refused as surely as one that answers with prose.
    """
    if not answer or not answer.strip():
        raise ModelRefusedError(answer or "")
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        pass

    for opening, closing in (("{", "}"), ("[", "]")):
        start, end = answer.find(opening), answer.rfind(closing)
        if start != -1 and end > start:
            try:
                return json.loads(answer[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ModelRefusedError(answer)


def usage_of(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    :param response: What :func:`ask` returned.
    :return: What the question cost, when the response says.
    """
    return response.get("usage")
