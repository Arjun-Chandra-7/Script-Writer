from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from .domain import RemoteFile, RemoteSource


READ_ONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


class DriveDependencyError(RuntimeError):
    pass


class GoogleDriveSource(RemoteSource):
    """Read-only, paginated adapter for one Google Drive folder."""

    def __init__(self, folder_id: str, credentials_file: Path):
        if not credentials_file.is_file():
            raise FileNotFoundError(
                f"Google credentials not found at {credentials_file}. "
                "Create a service account, share the source folder with its email as Viewer, "
                "and mount its JSON key at that path."
            )
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise DriveDependencyError(
                "Google Drive dependencies are missing; install the project first"
            ) from exc

        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_file), scopes=[READ_ONLY_SCOPE]
        )
        self.folder_id = folder_id
        self.service: Any = build(
            "drive", "v3", credentials=credentials, cache_discovery=False
        )

    def list_files(self) -> list[RemoteFile]:
        files: list[RemoteFile] = []
        page_token: str | None = None
        escaped_folder = self.folder_id.replace("'", "\\'")
        while True:
            response = (
                self.service.files()
                .list(
                    q=f"'{escaped_folder}' in parents and trashed = false",
                    spaces="drive",
                    fields=(
                        "nextPageToken,files("
                        "id,name,mimeType,modifiedTime,size,md5Checksum)"
                    ),
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    orderBy="createdTime",
                )
                .execute()
            )
            for item in response.get("files", []):
                name = str(item.get("name", ""))
                mime_type = str(item.get("mimeType", ""))
                if mime_type != "application/json" and not name.lower().endswith(".json"):
                    continue
                raw_size = item.get("size")
                files.append(
                    RemoteFile(
                        file_id=str(item["id"]),
                        name=name,
                        mime_type=mime_type,
                        modified_time=str(item.get("modifiedTime", "")),
                        size=int(raw_size) if raw_size is not None else None,
                        md5_checksum=item.get("md5Checksum"),
                    )
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                return files

    def download(self, file_id: str, destination: BinaryIO) -> None:
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise DriveDependencyError(
                "Google Drive dependencies are missing; install the project first"
            ) from exc

        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        downloader = MediaIoBaseDownload(destination, request, chunksize=1024 * 1024)
        done = False
        while not done:
            _status, done = downloader.next_chunk(num_retries=3)
