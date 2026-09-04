"""
Reset the semantic annotation class taxonomy to its hand-authored baseline.

Undoes the codebase modifications made by refine_class_structure.py:
1. Strips all class definitions from generated_classes.py, keeping only imports
2. Regenerates ormatic_interface.py (ORM) so it no longer references the removed classes

Run this before a fresh extraction+refinement cycle to start from a clean taxonomy.
"""

import argparse
import ast
import logging
import subprocess
import sys
from pathlib import Path

from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationFilePaths,
)

SIMPLE_GENERATED_CLASSES = """\
from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import SemanticEnvironmentAnnotation
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
"""


def strip_class_definitions(filepath: Path) -> int:
    """Remove all class definitions from a Python file, keeping imports and other
    top-level statements. Returns the number of classes removed."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    # Collect line ranges occupied by top-level ClassDef nodes (including decorators)
    remove = set()
    class_count = 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_count += 1
            start = (
                node.decorator_list[0].lineno if node.decorator_list else node.lineno
            )
            remove.update(range(start, node.end_lineno + 1))

    if class_count == 0:
        return 0

    kept = [line for i, line in enumerate(lines, start=1) if i not in remove]

    # Strip trailing blank lines
    while kept and kept[-1].strip() == "":
        kept.pop()
    kept.append("\n")

    filepath.write_text("".join(kept), encoding="utf-8")
    return class_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Replace generated_classes.py with a minimal imports-only file "
        "(skips AST stripping and ORM regeneration).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    generated_file = Path(SemanticAnnotationFilePaths.GENERATED_CLASSES_FILE.value)

    if args.simple:
        generated_file.write_text(SIMPLE_GENERATED_CLASSES, encoding="utf-8")
        logging.info(f"Wrote minimal generated_classes.py to {generated_file}")
        return

    # --- Step 1: Strip class definitions from generated_classes.py ---
    if generated_file.exists():
        n = strip_class_definitions(generated_file)
        if n:
            logging.info(f"Removed {n} class definition(s) from {generated_file}")
        else:
            logging.info(f"No class definitions to remove in {generated_file}")
    else:
        logging.info(f"Nothing to reset: {generated_file} does not exist")

    # --- Step 2: Regenerate ORM without the generated classes ---
    # Run in a subprocess so the fresh Python process never sees the deleted module.
    # generated_classes.py sits at:
    #   <project>/src/semantic_digital_twin/semantic_annotations/generated_classes.py
    # generate_orm.py sits at:
    #   <project>/scripts/generate_orm.py
    # So we go up 4 levels (annotations -> s_d_t pkg -> src -> project).
    generate_orm_script = (
        generated_file.parent.parent.parent.parent
        / "scripts"
        / "generate_orm.py"
    )

    if not generate_orm_script.exists():
        logging.error(f"ORM generation script not found at {generate_orm_script}")
        sys.exit(1)

    logging.info(f"Regenerating ORM via {generate_orm_script} ...")
    result = subprocess.run(
        [sys.executable, str(generate_orm_script)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logging.error(f"ORM generation failed:\n{result.stderr}")
        sys.exit(1)

    if result.stdout:
        logging.info(result.stdout.rstrip())

    logging.info("Taxonomy reset complete.")


if __name__ == "__main__":
    main()
