"""
Keep the classes a run generates inside that run.

A scene needs classes the taxonomy does not have -- a ``Ceiling``, a ``KitchenIsland``.
Written into the ontology's own package, as they were, they are a tracked file that every
run leaves modified, one careless ``git add`` away from becoming part of the shared
ontology, and importable by the next run's taxonomy export.

They belong to the run that proposed them, so they are written into its directory. The
run's directory is then put at the *front* of the annotations package's search path,
which makes its file the one imported under
``semantic_digital_twin.semantic_annotations.generated_classes`` -- the name the ORM
refers to them by -- ahead of the package's own empty one. The ORM generator finds them
by walking that path, so it maps them without being told anything.

This has to be done before anything imports the ORM or the classes, which is why it is a
call at the top of a step rather than something a step can opt into later.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

MODULE_NAME = "semantic_digital_twin.semantic_annotations.generated_classes"
"""
The name the ORM refers to a generated class by, whatever directory it is written in.
"""

FILE_NAME = "generated_classes.py"
"""
What the file is called inside a run's directory.
"""


def path_in(run: Path) -> Path:
    """
    :param run: A run's directory.
    :return: Where its generated classes are written.
    """
    return Path(run) / FILE_NAME


def use(run: Path) -> Optional[ModuleType]:
    """
    Make a run's generated classes the ones this interpreter means.

    :param run: The run's directory.
    :return: The module, once something imports it, or None if the run generated none.
    :raises RuntimeError: If the classes were already imported from somewhere else, since
        by then everything holding one of them holds the wrong one.
    """
    run = Path(run).resolve()
    if not path_in(run).exists():
        return None

    already = sys.modules.get(MODULE_NAME)
    if already is not None:
        if Path(getattr(already, "__file__", "")).resolve() == path_in(run):
            return already
        raise RuntimeError(
            f"{MODULE_NAME} was already imported from "
            f"{getattr(already, '__file__', 'nowhere')}, so this run's classes cannot "
            f"be the ones in use. Call use() before importing the ORM."
        )

    import semantic_digital_twin.semantic_annotations as annotations

    # In front of the package's own file, so a walk of the package finds this one first.
    if str(run) not in annotations.__path__:
        annotations.__path__.insert(0, str(run))
    return sys.modules.get(MODULE_NAME)
