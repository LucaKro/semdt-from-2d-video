"""
Give each run its own corner of the database, the way it has its own directory.

A run writes two worlds and, for the classes it had to invent, tables of its own. Those
tables outlive the run: ``create_all`` creates what is absent and leaves alone whatever
is already standing, however little it now resembles the class it is meant to hold. So
the run after this one inherits them. One run generated ``Ceiling(HasRootRegion)`` and
the next ``Ceiling(HasRootBody)``, and the second failed on the first one's table::

    insert or update on table "CeilingDAO" violates foreign key constraint
    DETAIL: Key (database_id)=(7046) is not present in table "HasRootRegionDAO".

A run reads nothing another run concluded, and this is the one place that was not true
of. Each run gets a schema named for it, everything it writes goes there, and the tables
are built by the same ORM that is about to write to them -- so there is nothing for a
run to inherit and nothing to drift. Throwing a run away is :func:`drop`, beside
deleting its directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

VARIABLE = "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
"""
The environment variable the semantic digital twin takes its database from. A step is
pointed at its run's schema by setting this before it starts, which is why every step
reaches the right one without being told: they are all children of the run.
"""


def schema_for(run: Path) -> str:
    """
    Name the schema a run writes into.

    :param run: The run's directory.
    :return: The schema name, derived from the directory so the two can be read off one
        another without anything recording the pairing.
    """
    return "run_" + re.sub(r"[^0-9a-zA-Z]", "_", run.name)


def base_uri() -> str:
    """
    Read the database the environment points at.

    :return: The URI, as it is set.
    :raises SystemExit: If nothing is set.
    """
    uri = os.environ.get(VARIABLE)
    if uri is None:
        raise SystemExit(f"{VARIABLE} is not set, so there is no database to write to.")
    return uri


def uri_in(schema: str, uri: Optional[str] = None) -> str:
    """
    Point a database URI at one schema and nothing else.

    The search path holds the run's schema alone, without ``public`` behind it. With
    ``public`` in the path the ORM would find the tables standing there and build none
    of its own, which is the inheritance this is here to stop.

    :param schema: The schema to write into.
    :param uri: The URI to change, by default the one in the environment.
    :return: The URI, asking for that search path.
    """
    url = make_url(uri if uri is not None else base_uri())
    query = dict(url.query)
    query["options"] = f"-csearch_path={schema}"
    return url.set(query=query).render_as_string(hide_password=False)


def create(schema: str, uri: Optional[str] = None) -> None:
    """
    Make a run's schema, if it is not already there.

    :param schema: The schema to make.
    :param uri: The database to make it in, by default the one in the environment.
    """
    engine = create_engine(uri if uri is not None else base_uri())
    with engine.begin() as connection:
        connection.execute(text(f'create schema if not exists "{schema}"'))
    engine.dispose()


def drop(schema: str, at_a_time: int = 100, uri: Optional[str] = None) -> int:
    """
    Throw away everything a run wrote to the database.

    Not ``DROP SCHEMA ... CASCADE``: a run's schema holds the whole ORM, and taking a
    lock on eleven hundred tables in one transaction runs the server out of the shared
    memory it keeps locks in::

        psycopg.errors.OutOfMemory: out of shared memory
        HINT:  You might need to increase max_locks_per_transaction.

    So the tables go in batches, each its own transaction, and the empty schema last.

    :param schema: The schema to throw away.
    :param at_a_time: How many tables to drop per transaction.
    :param uri: The database to drop it in, by default the one in the environment.
    :return: How many tables were dropped.
    """
    engine = create_engine(uri if uri is not None else base_uri())
    listing = text(
        "select tablename from pg_tables where schemaname = :schema order by tablename"
    )
    dropped = 0
    try:
        while True:
            with engine.begin() as connection:
                names = [
                    row[0]
                    for row in connection.execute(listing, {"schema": schema})
                ][:at_a_time]
                if not names:
                    break
                targets = ", ".join(f'"{schema}"."{name}"' for name in names)
                connection.execute(text(f"drop table {targets} cascade"))
                dropped += len(names)
        with engine.begin() as connection:
            connection.execute(text(f'drop schema if exists "{schema}" cascade'))
    finally:
        engine.dispose()
    return dropped


def use(run: Path) -> str:
    """
    Point this process, and everything it starts, at a run's schema.

    :param run: The run's directory.
    :return: The schema now in use.
    """
    schema = schema_for(run)
    os.environ[VARIABLE] = uri_in(schema)
    return schema
