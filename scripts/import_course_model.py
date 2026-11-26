"""Register the bundled coursework GGUF through Ollama's loopback API."""

import json
import re
import sys
from pathlib import Path

from ollama import Client


def register(model: str, modelfile: Path) -> None:
    source = modelfile.read_text(encoding="utf-8")
    model_line = re.search(r"^FROM\s+(.+)$", source, re.MULTILINE)
    template = re.search(r'^TEMPLATE\s+"""(.*?)"""\s*$', source, re.MULTILINE | re.DOTALL)
    if model_line is None or template is None:
        raise ValueError("The coursework Modelfile needs a GGUF path and a template.")
    gguf = modelfile.parent / model_line.group(1).strip()
    parameters = {
        key: json.loads(value)
        for key, value in re.findall(r"^PARAMETER\s+(\w+)\s+(.+)$", source, re.MULTILINE)
    }
    client = Client(host="http://127.0.0.1:11434", timeout=None, trust_env=False)
    print("Copying the local GGUF to the Ollama model store...", flush=True)
    digest = client.create_blob(gguf)
    for update in client.create(
        model, files={gguf.name: digest}, template=template.group(1),
        parameters=parameters, stream=True,
    ):
        print(update.status, flush=True)


if __name__ == "__main__":
    register(sys.argv[1], Path(sys.argv[2]))
