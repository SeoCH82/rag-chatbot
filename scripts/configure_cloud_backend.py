"""Create local credentials for the Streamlit-to-LangServe link, without printing them."""
from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def configure(root: Path = ROOT, backend_url: str | None = None) -> None:
    config = json.loads((root / "deployment_config.json").read_text(encoding="utf-8"))
    backend_url = backend_url or config["backend_url"]
    parsed = urlparse(backend_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path not in {"", "/"}:
        raise ValueError("backend_url must be an HTTPS origin.")
    directory = root / ".server_secrets"
    directory.mkdir(exist_ok=True, mode=0o700)
    credentials = directory / "backend.json"
    if credentials.exists():
        key = json.loads(credentials.read_text(encoding="utf-8"))["api_key"]
        if not isinstance(key, str) or len(key) < 32:
            raise ValueError("Existing backend credentials are invalid.")
    else:
        key = secrets.token_urlsafe(48)
        with credentials.open("x", encoding="utf-8") as output:
            json.dump({"api_key": key}, output)
        credentials.chmod(0o600)
    cloud_secrets = directory / "streamlit_secrets.toml"
    cloud_secrets.write_text(
        f'BACKEND_URL = {json.dumps(backend_url)}\n'
        f'BACKEND_API_KEY = {json.dumps(key)}\n', encoding="utf-8",
    )
    cloud_secrets.chmod(0o600)
    print("Private Cloud connection settings are ready in .server_secrets/streamlit_secrets.toml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url")
    configure(backend_url=parser.parse_args().backend_url)
