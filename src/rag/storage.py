"""Keep the original uploaded files, so a document can be re-indexed or returned later."""

from pathlib import Path
from typing import Protocol

import httpx

_TIMEOUT_SECONDS = 60


class StorageError(RuntimeError):
    """A file could not be written to or removed from storage."""


class FileStorage(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def delete(self, key: str) -> None:
        """Remove a file. Removing a file that is not there is not an error."""
        ...


class LocalStorage:
    """Files in a local directory, for running without Supabase."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise StorageError(f"storage key escapes the storage directory: {key!r}")
        return path


class SupabaseStorage:
    """A private Supabase Storage bucket, written with the service key.

    The bucket is private and only this backend holds the key, so users reach
    their files through the API, never through a public URL.
    """

    def __init__(
        self, url: str, service_key: str, bucket: str, client: httpx.Client | None = None
    ) -> None:
        self._base = f"{url.rstrip('/')}/storage/v1"
        self._bucket = bucket
        self._headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}"}
        self._client = client or httpx.Client(timeout=_TIMEOUT_SECONDS)
        self._bucket_ready = False

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._ensure_bucket()
        response = self._request(
            "POST",
            f"/object/{self._bucket}/{key}",
            content=data,
            headers={"Content-Type": content_type, "x-upsert": "true"},
        )
        if response.status_code not in (200, 201):
            raise StorageError(f"Upload failed with HTTP {response.status_code}.")

    def delete(self, key: str) -> None:
        response = self._request("DELETE", f"/object/{self._bucket}/{key}")
        if response.status_code not in (200, 204, 400, 404):
            raise StorageError(f"Delete failed with HTTP {response.status_code}.")

    def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        response = self._request(
            "POST", "/bucket", json={"id": self._bucket, "name": self._bucket, "public": False}
        )
        # 400 and 409 both mean the bucket already exists, depending on the version.
        if response.status_code not in (200, 201, 400, 409):
            raise StorageError(f"Could not create the bucket: HTTP {response.status_code}.")
        self._bucket_ready = True

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        headers = self._headers | kwargs.pop("headers", {})
        try:
            return self._client.request(method, self._base + path, headers=headers, **kwargs)
        except httpx.HTTPError as error:
            raise StorageError("Could not reach file storage.") from error
