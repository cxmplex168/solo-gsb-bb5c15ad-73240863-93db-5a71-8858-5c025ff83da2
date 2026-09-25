"""Startup schema/migration robustness: concurrent init and failure recovery."""
from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from app import database
from app.database import init_db


def _versions(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        ]
    finally:
        conn.close()


def _table_names(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        conn.close()


class TestConcurrentInit:
    def test_two_starters_apply_each_version_exactly_once(
        self, tmp_path, monkeypatch
    ):
        # Two service processes sharing one SQLite file start at the same
        # time.  Slow every migration down so the race window is wide open:
        # with the ledger read outside the write lock (the old behaviour)
        # both starters see an empty ledger and the second one crashes
        # re-running migration 1 (duplicate CREATE TABLE).
        def slowed(fn):
            def run(conn, now):
                time.sleep(0.1)
                fn(conn, now)

            return run

        monkeypatch.setattr(
            database,
            "MIGRATIONS",
            {v: slowed(fn) for v, fn in database.MIGRATIONS.items()},
        )

        db_path = str(tmp_path / "shared.db")
        errors = []

        def starter():
            try:
                init_db(db_path)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=starter) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Both starters come up; the later one waited on the write lock and
        # re-read the ledger inside it instead of re-running migrations.
        assert errors == []
        assert _versions(db_path) == [1, 2, 3, 4, 5]
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM ("
                " SELECT version FROM schema_migrations"
                " GROUP BY version HAVING COUNT(*) > 1)"
            ).fetchone()[0] == 0
        finally:
            conn.close()

        # A later restart is a pure no-op.
        init_db(db_path)
        assert _versions(db_path) == [1, 2, 3, 4, 5]


class TestMigrationFailureRecovery:
    def test_failed_migration_rolls_back_and_restart_resumes(
        self, tmp_path, monkeypatch
    ):
        db_path = str(tmp_path / "recover.db")
        real_migrations = dict(database.MIGRATIONS)

        def broken_migration(conn, applied_at):
            conn.execute("CREATE TABLE broken_half (id INTEGER)")
            raise RuntimeError("boom")

        monkeypatch.setattr(
            database,
            "MIGRATIONS",
            {**real_migrations, 6: broken_migration},
        )
        with pytest.raises(RuntimeError, match="boom"):
            init_db(db_path)

        # The failed version left nothing behind: no half-created table and
        # no ledger entry; the versions that completed stay applied.
        assert "broken_half" not in _table_names(db_path)
        assert _versions(db_path) == [1, 2, 3, 4, 5]

        # After the fix, a restart resumes the upgrade at the failed
        # version instead of starting over or stopping for good.
        def fixed_migration(conn, applied_at):
            conn.execute("CREATE TABLE fixed_feature (id INTEGER)")
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at)"
                " VALUES (6, ?)",
                (applied_at,),
            )

        monkeypatch.setattr(
            database,
            "MIGRATIONS",
            {**real_migrations, 6: fixed_migration},
        )
        init_db(db_path)
        assert _versions(db_path) == [1, 2, 3, 4, 5, 6]
        assert "fixed_feature" in _table_names(db_path)

    def test_base_schema_bootstrap_is_atomic(self, tmp_path, monkeypatch):
        # A failure part-way through the base schema must not leave a
        # half-created set of tables behind (executescript's implicit
        # commit used to make exactly that possible).
        monkeypatch.setattr(
            database,
            "SCHEMA_V0",
            "CREATE TABLE t1 (id INTEGER);\n"
            "CREATE TABLE t2 (id INTEGER);\n"
            "THIS IS NOT VALID SQL;\n",
        )
        with pytest.raises(sqlite3.Error):
            init_db(str(tmp_path / "bootstrap.db"))

        names = _table_names(str(tmp_path / "bootstrap.db"))
        assert "t1" not in names
        assert "t2" not in names
