"""Google Drive connector (read-only, Phase 1) — user OAuth.

Flow: frontend hits /drive/auth/url, user consents in browser, Google
redirects to /auth/google/callback, token is stored on disk and refreshed
automatically afterwards.

Config:
  GOOGLE_OAUTH_CLIENT_FILE=/path/to/client_secret.json   (OAuth client from GCP console)
  DRIVE_FOLDER_ID=<folder id from the Drive folder URL>
  DRIVE_OAUTH_REDIRECT_URI (optional, default http://localhost:8787/auth/google/callback)

Full sync (watcher + queue) is a later phase; this is pull-only.
"""

import io
import json
import os
import tempfile
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from ..models import DriveFile

# Google may hand back a broader scope set than requested (e.g. adds openid);
# without this oauthlib hard-fails the token exchange with "Scope has changed".
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

CREDENTIALS_DIR = Path(__file__).resolve().parent.parent.parent / "credentials"
TOKEN_FILE = CREDENTIALS_DIR / "token.json"
# State persisted to disk: uvicorn --reload restarts the process between
# auth-url issuance and the callback, wiping any in-memory value.
STATE_FILE = CREDENTIALS_DIR / "oauth_state"


class DriveNotConfigured(Exception):
    pass


class DriveNotAuthorized(Exception):
    pass


def _client_file() -> str:
    client_file = os.environ.get("GOOGLE_OAUTH_CLIENT_FILE")
    if not client_file or not Path(client_file).exists():
        raise DriveNotConfigured(
            "Set GOOGLE_OAUTH_CLIENT_FILE to the OAuth client secret JSON from GCP console."
        )
    return client_file


def _redirect_uri() -> str:
    return os.environ.get(
        "DRIVE_OAUTH_REDIRECT_URI", "http://localhost:8787/auth/google/callback"
    )


def _new_flow() -> Flow:
    return Flow.from_client_secrets_file(
        _client_file(), scopes=SCOPES, redirect_uri=_redirect_uri()
    )


def load_credentials() -> Credentials | None:
    if not TOKEN_FILE.exists():
        return None
    credentials = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        _save_credentials(credentials)
    return credentials if credentials.valid else None


def _save_credentials(credentials: Credentials) -> None:
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
    TOKEN_FILE.chmod(0o600)


def is_authorized() -> bool:
    try:
        return load_credentials() is not None
    except Exception:
        return False


def build_auth_url() -> str:
    flow = _new_flow()
    url, state = flow.authorization_url(access_type="offline", prompt="consent")
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    # Persist state AND the PKCE code verifier: the callback runs in a fresh
    # Flow instance (possibly a fresh process under --reload), and Google
    # rejects the exchange with "Missing code verifier" without it.
    STATE_FILE.write_text(
        json.dumps({"state": state, "code_verifier": flow.code_verifier}),
        encoding="utf-8",
    )
    STATE_FILE.chmod(0o600)
    return url


def handle_oauth_callback(code: str, state: str | None) -> None:
    if not STATE_FILE.exists():
        raise DriveNotAuthorized("No pending OAuth flow — restart the connect flow.")
    pending = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    if not state or state != pending.get("state"):
        raise DriveNotAuthorized("OAuth state mismatch — restart the connect flow.")
    STATE_FILE.unlink(missing_ok=True)
    flow = _new_flow()
    flow.code_verifier = pending.get("code_verifier")
    flow.fetch_token(code=code)
    _save_credentials(flow.credentials)


def disconnect() -> None:
    TOKEN_FILE.unlink(missing_ok=True)


def _get_service():
    credentials = load_credentials()
    if credentials is None:
        raise DriveNotAuthorized("Drive not connected — authorize via /drive/auth/url.")
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def get_folder_id() -> str:
    folder_id = os.environ.get("DRIVE_FOLDER_ID")
    if not folder_id:
        raise DriveNotConfigured("Set DRIVE_FOLDER_ID to the Drive folder to read from.")
    return folder_id


def list_pdfs() -> list[DriveFile]:
    service = _get_service()
    query = f"'{get_folder_id()}' in parents and mimeType='application/pdf' and trashed=false"
    files: list[DriveFile] = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, size, modifiedTime)",
                pageSize=100,
                pageToken=page_token,
                # Folder may live on a shared drive; without these flags the
                # API silently returns an empty list.
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for f in response.get("files", []):
            files.append(
                DriveFile(
                    id=f["id"],
                    name=f["name"],
                    size=int(f["size"]) if f.get("size") else None,
                    modified_time=f.get("modifiedTime"),
                )
            )
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files


def download_pdf(file_id: str) -> tuple[Path, str]:
    """Download a Drive file to a temp path. Returns (path, original_name)."""
    service = _get_service()
    info = (
        service.files()
        .get(fileId=file_id, fields="name", supportsAllDrives=True)
        .execute()
    )
    name = info["name"]

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    with io.FileIO(tmp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    os.close(fd)
    return Path(tmp_path), name
