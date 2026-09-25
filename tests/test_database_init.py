"""Startup bootstrap tests: concurrent init, exactly-once migrations, atomicity.

``init_db`` runs the whole bootstrap in one ``BEGIN IMMEDIATE`` transaction:
the write lock is taken before the migration ledger is read, so two service
processes started against the same SQLite file serialise instead of racing
(base schema, per-version migrations and version registrations commit or
roll back together -- no ``executescript`` implicit commit).  A failure
midway leaves neither half-created structures nor a wrong version record,
and the next start re-runs exactly the versions that never committed.
"""
from __future__ import annotations

import sqlite3
import threading

import pytest


ALL_VERSIONS = [1, 2, 3, 4, 5]


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


def _tables(db_path):
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


def _init_concurrently(db_path, n_threads):
    """Run ``init_db`` from n threads released at the same instant."""
    from app import database

    barrier = threading.Barrier(n_threads)
    errors = []

    def worker():
        try:
            barrier.wait(timeout=10)
            database.init_db(db_path)
        except Exception as exc:  # collected and asserted on below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return errors


class TestConcurrentStartup:
    def test_simultaneous_init_on_fresh_database(self, tmp_path):
        # Two service processes sharing one SQLite file and started together
        # must both come up: the loser of the write-lock race re-reads the
        # ledger and sees every version already applied.
        for round_ in range(3):
            db_path = str(tmp_path / f"fresh-{round_}.db")
            errors = _init_concurrently(db_path, 8)
            assert errors == []
            assert _versions(db_path) == ALL_VERSIONS
            assert {
                "sessions",
                "session_sources",
                "readings",
                "alert_rules",
                "alert_events",
                "correlated_rules",
                "checkpoints",
                "retention_tombstones",
                "schema_migrations",
            } <= _tables(db_path)

    def test_simultaneous_init_on_legacy_database(self, tmp_path):
        from app import database
        from app.database import utc_now
        from app.time_utils import store_event_time

        # A v0-era database (base schema only, live data, no ledger) upgraded
        # by several starters at once: data preserved, each version applied
        # exactly once.
        db_path = str(tmp_path / "legacy.db")
        now = store_event_time(utc_now())
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(database.SCHEMA_V0)
            conn.execute(
                "INSERT INTO sessions (session_id, source_ids,"
                " allowed_lateness_seconds, created_at) VALUES (?, ?, 0, ?)",
                ("legacy-s", '["a","b"]', now),
            )
            conn.execute(
                "INSERT INTO readings (session_id, source_id, seq,"
                " observed_at, payload, status, first_received_at) VALUES"
                " ('legacy-s', 'a', 1, '2026-01-01T00:00:00.000000Z',"
                " '41.5', 'buffered', ?)",
                (now,),
            )
            conn.commit()
        finally:
            conn.close()

        errors = _init_concurrently(db_path, 6)
        assert errors == []
        assert _versions(db_path) == ALL_VERSIONS
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute(
                "SELECT created_at FROM sessions WHERE session_id='legacy-s'"
            ).fetchone()[0] == now
            assert conn.execute(
                "SELECT payload, status FROM readings"
                " WHERE session_id='legacy-s' AND source_id='a' AND seq=1"
            ).fetchone() == ("41.5", "buffered")
        finally:
            conn.close()

    def test_simultaneous_connect_on_fresh_database(self, tmp_path):
        # ``PRAGMA journal_mode=WAL`` does not honour busy_timeout, so
        # concurrent starters used to fail here with "database is locked".
        from app import database

        db_path = str(tmp_path / "wal.db")
        barrier = threading.Barrier(12)
        errors = []

        def worker():
            try:
                barrier.wait(timeout=10)
                conn = database.connect(db_path)
                try:
                    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                    assert mode == "wal"
                finally:
                    conn.close()
            except Exception as exc:  # collected and asserted on below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert errors == []


class TestExactlyOnce:
    def test_each_migration_runs_exactly_once(self, tmp_path, monkeypatch):
        from app import database

        calls = {version: 0 for version in database.MIGRATIONS}
        for version, migration in list(database.MIGRATIONS.items()):

            def counted(conn, applied_at, _m=migration, _v=version):
                calls[_v] += 1
                _m(conn, applied_at)

            monkeypatch.setitem(database.MIGRATIONS, version, counted)

        db_path = str(tmp_path / "once.db")
        database.init_db(db_path)
        database.init_db(db_path)  # second start: nothing re-runs
        assert calls == {version: 1 for version in calls}
        assert _versions(db_path) == ALL_VERSIONS


class TestFailureAtomicity:
    def test_failed_bootstrap_rolls_everything_back(self, tmp_path, monkeypatch):
        from app import database

        def broken(conn, applied_at):
            conn.execute("CREATE TABLE half_applied (x TEXT)")
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at)"
                " VALUES (99, ?)",
                (applied_at,),
            )
            raise RuntimeError("boom")

        monkeypatch.setitem(database.MIGRATIONS, 99, broken)
        db_path = str(tmp_path / "fresh.db")
        with pytest.raises(RuntimeError, match="boom"):
            database.init_db(db_path)

        # No half-created structure and no version record survive.
        assert _tables(db_path) == set()

        # After the fix (broken migration gone) the next start upgrades.
        monkeypatch.delitem(database.MIGRATIONS, 99)
        database.init_db(db_path)
        assert _versions(db_path) == ALL_VERSIONS
        assert "half_applied" not in _tables(db_path)

    def test_failed_upgrade_preserves_existing_database(
        self, tmp_path, monkeypatch
    ):
        from app import database
        from app.database import utc_now
        from app.time_utils import store_event_time

        db_path = str(tmp_path / "existing.db")
        database.init_db(db_path)
        now = store_event_time(utc_now())
        conn = database.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO sessions (session_id, source_ids,"
                " allowed_lateness_seconds, created_at) VALUES (?, ?, 0, ?)",
                ("s1", '["a"]', now),
            )
            conn.execute(
                "INSERT INTO checkpoints (session_id, name, created_at,"
                " stream_offset, published_seq) VALUES ('s1', 'cp', ?, 0, 0)",
                (now,),
            )
            conn.commit()
        finally:
            conn.close()

        def broken(conn, applied_at):
            conn.execute("CREATE TABLE half_applied (x TEXT)")
            raise RuntimeError("boom")

        monkeypatch.setitem(database.MIGRATIONS, 99, broken)
        with pytest.raises(RuntimeError, match="boom"):
            database.init_db(db_path)

        # The failed attempt registered nothing and removed nothing.
        assert _versions(db_path) == ALL_VERSIONS
        assert "half_applied" not in _tables(db_path)
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute(
                "SELECT created_at FROM sessions WHERE session_id='s1'"
            ).fetchone()[0] == now
            assert conn.execute(
                "SELECT stream_offset FROM checkpoints"
                " WHERE session_id='s1' AND name='cp'"
            ).fetchone()[0] == 0
        finally:
            conn.close()

        # Restart after the fix is a clean no-op; data keeps its values.
        monkeypatch.delitem(database.MIGRATIONS, 99)
        database.init_db(db_path)
        assert _versions(db_path) == ALL_VERSIONS
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id='s1'"
            ).fetchone()[0] == 1
        finally:
            conn.close()
