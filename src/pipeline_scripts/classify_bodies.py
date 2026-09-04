"""
Ask what each of a split scene's bodies is.

The vocabulary step answered what the *label* ``cabinet`` means in the taxonomy. This
asks the other question that was hiding under it: whether this particular object really
is one. It could not be asked before, because until the mesh was split, colouring
``cabinet_4`` painted exactly ``door_10``'s faces and a model would rightly have answered
"a door"::

    python -m pipeline_scripts.classify_bodies \
        dataset/kitchenlab_new_mesh_agreement_dataset pipeline_out_relations

Bodies are addressed by the name they carry everywhere else -- ``drawer_19``, the same
key in ``split.json``, in the pairings, and in the world the split built -- and an answer
is kept only if it names bodies that were actually in the picture. Nothing else in the
pipeline can mount a part into a whole if the two ends stop meaning the same objects, so
the identity is checked rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from experiments.warsaw.world_loader import LabelSegment, WarsawWorldLoader

from pipeline_scripts import model_client
from pipeline_scripts.locations import TAXONOMY

SYSTEM_PROMPT = """\
You are naming objects in a scanned room, using an ontology of classes.

Each picture shows the same room from a different side, with a few objects painted in
distinct colors and everything else left as it was scanned. Say what each painted object
is. You are told the label a previous step gave it and the class that label was mapped
to; both can be wrong about this particular object, which is what you are being asked.

Rules:
- Name every object you are given, once, by the name it is listed under.
- "class" is a name from the ontology's classes, or a name you propose.
- If you propose one, "is_new_class" is true and "superclass" is a class of the ontology.
- Judge the object, not the paint: the colors mark what to look at, nothing more.
- A class marked "abstract" cannot be given to an object. Name one of its subclasses,
  or propose a new class with it as the superclass.

Answer with JSON and nothing else:
{"objects": [{"name": "drawer_19", "class": "Drawer", "is_new_class": false,
              "superclass": "Furniture", "confidence": 0.0, "reason": "one sentence"}]}"""

ANSWER_FIELDS = ("class", "is_new_class", "superclass", "confidence", "reason")
"""
What is kept of an answer about one body.
"""


def split_segments(
    loader: WarsawWorldLoader, faces_path: Path
) -> List[LabelSegment]:
    """
    Rebuild the scene's segments from the faces the split left them.

    The name a segment carries is made of its label and instance, so a segment rebuilt
    with the split's faces answers to exactly the name its body has -- which is what lets
    an answer about ``drawer_19`` reach the body called ``drawer_19``.

    :param loader: The loaded scene.
    :param faces_path: The ``*_faces.npz`` the split wrote.
    :return: One segment per body, carrying only the faces that body kept.
    """
    kept = np.load(faces_path)
    by_name = {str(segment.name): segment for segment in loader.label_segments}
    return [
        LabelSegment(
            class_name=by_name[name].class_name,
            instance=by_name[name].instance,
            faces=kept[name],
        )
        for name in kept.files
    ]


def question_for(
    group: Sequence[LabelSegment],
    colors: Dict[Any, Any],
    images: Dict[str, bytes],
    taxonomy: Dict[str, Any],
    vocabulary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build the message asking what a group of bodies are.

    :param group: The bodies painted in the pictures.
    :param colors: The color each was given, by name.
    :param images: The renders, by viewpoint.
    :param taxonomy: The exported taxonomy.
    :param vocabulary: What the vocabulary step answered per label.
    :return: The message, as :func:`model_client.ask` takes it.
    """
    listed = "\n".join(
        f"{segment.name}: painted {colors[segment.name].closest_css3_name()}, "
        f"labelled \"{segment.class_name}\" by the scan, which was read as "
        f"{(vocabulary['labels'].get(segment.class_name) or {}).get('class') or 'no class'}"
        for segment in group
    )
    content = [
        model_client.text_part(
            f"## The ontology\n{json.dumps(taxonomy)}\n\n"
            f"## The objects to name\n{listed}"
        )
    ]
    for viewpoint, image in sorted(images.items()):
        content.append(model_client.text_part(f"The room from the {viewpoint}."))
        content.append(model_client.rendered_part(image))
    return content


def check(answered: Dict[str, Any], group: Sequence[LabelSegment]) -> List[str]:
    """
    Say what is wrong with an answer about a group, if anything.

    :param answered: What the model said, by body name.
    :param group: The bodies it was asked about.
    :return: One sentence per problem, empty when there are none.
    """
    wanted = {str(segment.name) for segment in group}
    problems = []
    missing = sorted(wanted - set(answered))
    unknown = sorted(set(answered) - wanted)
    if missing:
        problems.append(f"nothing was said about {', '.join(missing)}")
    if unknown:
        problems.append(
            f"{', '.join(unknown)} were named but are not in the picture"
        )
    for name, answer in answered.items():
        if name in wanted and not answer.get("class"):
            problems.append(f"{name} was given no class")
    return problems


def read(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param response: What the model answered with.
    :return: Per body name, the fields of its answer that are kept.
    """
    answered = model_client.parse_json_answer(model_client.answer_text(response))
    # Asked for {"objects": [...]}, a model will sometimes answer with the array alone.
    objects = answered if isinstance(answered, list) else answered.get("objects", [])
    return {
        str(one.get("name")): {field: one.get(field) for field in ANSWER_FIELDS}
        for one in objects
        if one.get("name")
    }


def build(arguments: argparse.Namespace) -> None:
    """
    Name every body of a split scene.

    :param arguments: The command line arguments.
    """
    evidence = arguments.evidence_directory
    taxonomy = json.loads(TAXONOMY.read_text())
    vocabulary = json.loads((evidence / "vocabulary.json").read_text())

    loader = WarsawWorldLoader(input_directory=arguments.scene_directory)
    bodies = split_segments(loader, evidence / "split_faces.npz")
    print(f"{len(bodies)} bodies to name, {arguments.group_size} at a time")

    kept = evidence / "classification_answers"
    images_directory = evidence / "classifications"
    images_directory.mkdir(parents=True, exist_ok=True)

    named: Dict[str, Any] = {}
    for rendered in loader.render_label_segment_groups(
        group_size=arguments.group_size,
        segments=bodies,
        headless=arguments.headless,
    ):
        group = rendered.segments
        for viewpoint, image in rendered.images.items():
            (images_directory / f"group{rendered.index}__{viewpoint}.png").write_bytes(
                image
            )

        answer_path = kept / f"group{rendered.index}.json"
        problems: Sequence[str] = ()
        for attempt in range(1 + arguments.corrections):
            if arguments.reuse_answers and answer_path.exists() and not problems:
                response = json.loads(answer_path.read_text())
            else:
                content = question_for(
                    group, rendered.colors, rendered.images, taxonomy, vocabulary
                )
                if problems:
                    content.append(
                        model_client.text_part(
                            "Your previous answer could not be used:\n- "
                            + "\n- ".join(problems)
                            + "\nAnswer the same question again, correcting that."
                        )
                    )
                response = model_client.ask(
                    content, SYSTEM_PROMPT, model=arguments.model
                )
                kept.mkdir(parents=True, exist_ok=True)
                answer_path.write_text(json.dumps(response, indent=2))

            try:
                answered = read(response)
            except model_client.ModelRefusedError as refusal:
                answered, problems = {}, [str(refusal)]
            else:
                problems = check(answered, group)
            if not problems:
                break
            if attempt + 1 <= arguments.corrections:
                print(f"  group {rendered.index}: {problems[0]}, asking again ...")

        for segment in group:
            name = str(segment.name)
            answer = answered.get(name, {field: None for field in ANSWER_FIELDS})
            answer["label"] = segment.class_name
            answer["faces"] = int(len(segment))
            named[name] = answer
            new = " [new]" if answer.get("is_new_class") else ""
            print(f"  {name:<22} -> {answer.get('class')}{new}")
        for problem in problems:
            print(f"      ! {problem}")

    arguments.output.write_text(
        json.dumps(
            {
                "model": arguments.model,
                "scene": str(loader.scene.mesh_path),
                "bodies": named,
            },
            indent=2,
        )
    )

    counted = Counter(
        answer["class"] for answer in named.values() if answer.get("class")
    )
    agreed = sum(
        1
        for name, answer in named.items()
        if answer.get("class")
        == ((vocabulary["labels"].get(answer["label"]) or {}).get("class"))
    )
    print(
        f"\n{len(counted)} distinct classes over {len(named)} bodies; "
        f"{agreed} agree with what their label was mapped to"
    )
    print(f"written to {arguments.output}")


def main() -> None:
    """
    Name the bodies of the split scene named on the command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "scene_directory", type=Path, help="Directory holding the scene's mesh."
    )
    parser.add_argument(
        "evidence_directory",
        type=Path,
        help="The directory the evidence, adjudication and split runs wrote.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the classes, by default classifications.json beside the "
        "split they were made from.",
    )
    parser.add_argument("--model", default=model_client.DEFAULT_MODEL,
                        help="Which model to ask.")
    parser.add_argument("--group-size", type=int, default=8,
                        help="How many bodies to paint and ask about at once.")
    parser.add_argument(
        "--corrections",
        type=int,
        default=1,
        help="How often an unusable answer is put back to the model with what was "
        "wrong with it.",
    )
    parser.add_argument(
        "--reuse-answers",
        action="store_true",
        help="Read back the kept responses instead of asking again.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Render without opening a window."
    )
    arguments = parser.parse_args()
    if arguments.output is None:
        arguments.output = arguments.evidence_directory / "classifications.json"
    build(arguments)


if __name__ == "__main__":
    main()
