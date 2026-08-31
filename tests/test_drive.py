from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from script_writer.drive import GoogleDriveSource


class DriveListTests(unittest.TestCase):
    def test_list_is_paginated_and_filters_non_json(self) -> None:
        first_request = MagicMock()
        first_request.execute.return_value = {
            "nextPageToken": "page-2",
            "files": [
                {
                    "id": "one",
                    "name": "one.json",
                    "mimeType": "application/json",
                    "modifiedTime": "2026-01-01T00:00:00Z",
                    "size": "12",
                    "md5Checksum": "abc",
                },
                {
                    "id": "ignored",
                    "name": "notes.txt",
                    "mimeType": "text/plain",
                    "modifiedTime": "2026-01-01T00:00:00Z",
                },
            ],
        }
        second_request = MagicMock()
        second_request.execute.return_value = {
            "files": [
                {
                    "id": "two",
                    "name": "two.JSON",
                    "mimeType": "application/octet-stream",
                    "modifiedTime": "2026-01-02T00:00:00Z",
                }
            ]
        }
        files_resource = MagicMock()
        files_resource.list.side_effect = [first_request, second_request]
        service = MagicMock()
        service.files.return_value = files_resource
        source = object.__new__(GoogleDriveSource)
        source.folder_id = "folder-id"
        source.service = service

        items = source.list_files()

        self.assertEqual([item.file_id for item in items], ["one", "two"])
        self.assertEqual(items[0].revision_key, "md5:abc")
        self.assertTrue(items[1].revision_key.startswith("meta:"))
        self.assertEqual(files_resource.list.call_count, 2)
