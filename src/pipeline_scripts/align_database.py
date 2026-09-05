"""
Give the database the columns the ORM has grown, without dropping what is in it.

``create_all`` creates tables that are absent and leaves alone every table that is
already there, however far it has drifted from the class it is meant to hold. A run
regenerates the ORM from the ontology as it is committed, so the day the ontology gains
an attribute -- a colour becoming a class with subclasses, a modification learning to
point at the annotation it made -- every world written after that names a column the
database has never heard of, and the write fails on the first insert::

    column "polymorphic_type" of relation "ColorDAO" does not exist

The drift is additive: the ORM asks for columns that are missing, never for the removal
of one that is there. So it can be closed by adding them, which costs no stored world.
Rows written before the column existed get the value they would have been written with,
which for a class discriminator is the class that was the only one at the time::

    python -m pipeline_scripts.align_database

A column that is required and has no such value is not guessed at. It is reported, and
closing that gap is a decision about the stored worlds rather than a migration.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import Column, MetaData, Table, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapper


def drift(engine: Engine, metadata: MetaData) -> Dict[Table, List[Column]]:
    """
    Find the columns the ORM declares and the database does not have.

    :param engine: The database to look at.
    :param metadata: The ORM's tables.
    :return: Per table, the columns that are missing from it. Tables the database does
        not have at all are left out, since creating those is what ``create_all`` does.
    """
    reader = inspect(engine)
    present = set(reader.get_table_names())
    missing: Dict[Table, List[Column]] = {}
    for table in metadata.sorted_tables:
        if table.name not in present:
            continue
        columns = {column["name"] for column in reader.get_columns(table.name)}
        absent = [column for column in table.columns if column.name not in columns]
        if absent:
            missing[table] = absent
    return missing


def value_for_rows_written_before(
    table: Table, column: Column, mappers: Iterable[Mapper]
) -> Optional[str]:
    """
    Say what a row written before this column existed should hold in it.

    A class discriminator is the one case where that is knowable: rows in the table
    predate the subclasses, so they are instances of the class that maps the table
    itself, and its identity is the value the ORM would have written.

    :param table: The table the column belongs to.
    :param column: The column being added.
    :param mappers: The ORM's mappers, to find the class the table itself maps.
    :return: The value to fill in, or None if there is nothing it can be.
    """
    for mapper in mappers:
        if mapper.local_table is not table or mapper.base_mapper is not mapper:
            continue
        on = mapper.polymorphic_on
        if on is not None and on.name == column.name:
            return mapper.polymorphic_identity
    return None


def align(engine: Engine, base: type) -> Tuple[List[str], List[str]]:
    """
    Add the missing columns to the database.

    :param engine: The database to change.
    :param base: The ORM's declarative base, which holds both the tables and the
        mappers that say what a stored row already is.
    :return: What was added, and what could not be added without a decision about the
        rows that are already stored.
    """
    mappers = list(base.registry.mappers)
    added: List[str] = []
    refused: List[str] = []
    for table, columns in drift(engine, base.metadata).items():
        quoted = f'"{table.name}"'
        with engine.begin() as connection:
            stored = connection.execute(text(f"select count(*) from {quoted}")).scalar()
            for column in columns:
                fill = value_for_rows_written_before(table, column, mappers)
                if stored and not column.nullable and fill is None:
                    refused.append(
                        f"{table.name}.{column.name} is required, holds no value a "
                        f"stored row can be given, and {stored} rows are stored"
                    )
                    continue
                kind = column.type.compile(engine.dialect)
                connection.execute(
                    text(f"alter table {quoted} add column \"{column.name}\" {kind}")
                )
                if fill is not None:
                    connection.execute(
                        text(f"update {quoted} set \"{column.name}\" = :fill"),
                        {"fill": fill},
                    )
                if not column.nullable:
                    connection.execute(
                        text(
                            f"alter table {quoted} alter column "
                            f"\"{column.name}\" set not null"
                        )
                    )
                for reference in column.foreign_keys:
                    target = reference.column
                    connection.execute(
                        text(
                            f"alter table {quoted} add foreign key "
                            f"(\"{column.name}\") references "
                            f"\"{target.table.name}\" (\"{target.name}\")"
                        )
                    )
                filled = f", filled with {fill!r}" if fill is not None else ""
                added.append(f"{table.name}.{column.name} {kind}{filled}")
    return added, refused


def build(arguments: argparse.Namespace) -> None:
    """
    Align the database the environment points at.

    :param arguments: The command line arguments.
    :raises SystemExit: If a column cannot be added without a decision about the rows
        that are stored, or if no database is configured.
    """
    from semantic_digital_twin.orm.ormatic_interface import Base

    if os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI") is None:
        raise SystemExit(
            "SEMANTIC_DIGITAL_TWIN_DATABASE_URI is not set, so there is no database "
            "to align."
        )
    from semantic_digital_twin.orm.utils import semantic_digital_twin_sessionmaker

    engine = semantic_digital_twin_sessionmaker()().bind
    Base.metadata.create_all(bind=engine)
    added, refused = align(engine, Base)

    if not added and not refused:
        print("  the database already has every column the ORM asks for")
    for line in added[: arguments.report]:
        print(f"  added {line}")
    if len(added) > arguments.report:
        print(f"  ... and {len(added) - arguments.report} more columns")
    if added:
        print(f"  {len(added)} columns added")
    if refused:
        raise SystemExit(
            "these columns cannot be added to a database that already holds rows:\n  "
            + "\n  ".join(refused)
            + "\nThe stored worlds predate them, so closing this is a decision about "
            "those worlds rather than a migration."
        )


def main() -> None:
    """
    Align the database from the command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report",
        type=int,
        default=8,
        help="How many of the added columns to name before counting the rest.",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
