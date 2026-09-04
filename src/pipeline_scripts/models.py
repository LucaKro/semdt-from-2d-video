"""
The models a run may be given, named rather than spelled out.

Every question this pipeline asks comes with pictures, so every model here reads images;
a text-only model would fail on the first call rather than answer worse. They are listed
cheapest first, with what a million prompt tokens costs, because the whole run is about a
hundred calls and the difference between the ends of this list is the difference between
two cents and a dollar.
"""

from __future__ import annotations

from enum import Enum


class Model(Enum):
    """
    A model to put the pipeline's questions to.

    The value is the identifier OpenRouter knows it by.
    """

    QWEN3_VL_32B = "qwen/qwen3-vl-32b-instruct"
    """
    $0.10 per million prompt tokens. Dense 32B; the cheapest of these.
    """

    QWEN3_VL_30B = "qwen/qwen3-vl-30b-a3b-instruct"
    """
    $0.15. Mixture-of-experts, 3B active. What every run in this repository so far used,
    and what the reported numbers come from.
    """

    GPT_5_6_LUNA = "openai/gpt-5.6-luna"
    """
    $0.20.
    """

    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    """
    $0.30.
    """

    CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4.5"
    """
    $1.00.
    """

    GEMINI_2_5_PRO = "google/gemini-2.5-pro"
    """
    $1.25.
    """

    CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4.5"
    """
    $3.00. Worth trying on the steps that were unstable: the ownership answers vary
    between runs on about six of the thirty-three patterns, and the vocabulary step
    composes a class differently from one run to the next.
    """

    def __str__(self) -> str:
        """
        :return: The identifier, so that passing a member where a name is wanted works.
        """
        return self.value
