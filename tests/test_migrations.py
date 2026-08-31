from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from script_writer.database import Registry


class MigrationTests(unittest.TestCase):
    def test_v1_registry_migrates_to_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO schema_meta VALUES ('version', '1')")
            connection.commit()
            connection.close()

            registry = Registry(path)
            registry.initialize()
            version = registry.connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'version'"
            ).fetchone()[0]
            tables = {
                row[0]
                for row in registry.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
            self.assertEqual(version, "2")
            self.assertIn("intelligence_records", tables)
            self.assertIn("intelligence_fts", tables)
            self.assertIn("outcome_records", tables)
            registry.close()
