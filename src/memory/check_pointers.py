from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)


def create_sqlite_checkpointer(
    database_path: Path,
) -> tuple[SqliteSaver, sqlite3.Connection]:
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(connection)

    return checkpointer, connection