from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import tiktoken
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredPowerPointLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "cache" / "huggingface" / "sentence-transformers"
TIKTOKEN_CACHE_DIR = PROJECT_ROOT / "cache" / "tiktoken"
_TOKEN_PARTS = re.compile(r"[가-힣]|[A-Za-z0-9_]+|[^\s]")
_ENCODER = None
_ENCODER_CHECKED = False
PROFILE_FACT_LINE = re.compile(r"^-\s*([^:]+):\s*(.*?)\s*$")
PUBLIC_SENSITIVE_FIELDS = {
    "학번",
    "생년월일",
    "이메일",
    "휴대전화",
    "집전화번호",
    "보호자 휴대전화",
    "우편번호",
    "기본 주소",
    "상세 주소",
    "전체 주소",
}


def configure_local_cache() -> None:
    cache_dir = PROJECT_ROOT / "cache"
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "huggingface" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))


def embedding_cache_is_ready() -> bool:
    if not EMBEDDING_CACHE_DIR.is_dir():
        return False
    return any(
        path.name in {"model.safetensors", "pytorch_model.bin"}
        for path in EMBEDDING_CACHE_DIR.rglob("*")
        if path.is_file()
    )


def tiktoken_len(text: str) -> int:
    """Count tokens without making an unexpected network request.

    tiktoken downloads its BPE table the first time an encoding is requested.
    A copied desktop bundle may be offline, so only use tiktoken when that table
    is already present in the project cache. Otherwise use a deterministic
    Korean/ASCII approximation suitable for document chunk sizing.
    """
    global _ENCODER, _ENCODER_CHECKED

    if not _ENCODER_CHECKED:
        _ENCODER_CHECKED = True
        if TIKTOKEN_CACHE_DIR.is_dir() and any(TIKTOKEN_CACHE_DIR.iterdir()):
            try:
                _ENCODER = tiktoken.get_encoding("cl100k_base")
            except Exception:
                _ENCODER = None

    if _ENCODER is not None:
        return len(_ENCODER.encode(text))

    count = 0
    for part in _TOKEN_PARTS.findall(text):
        if part.isascii() and (part.isalnum() or "_" in part):
            count += max(1, math.ceil(len(part) / 4))
        else:
            count += 1
    return max(1, count)


def iter_files(data_dir: Path) -> Iterable[Path]:
    suffixes = {".pdf", ".docx", ".pptx", ".md", ".txt"}
    for path in sorted(data_dir.rglob("*")):
        relative_parts = path.relative_to(data_dir).parts
        if "training" in relative_parts:
            continue
        if ".template." in path.name:
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix == ".pptx":
        loader = UnstructuredPowerPointLoader(str(path))
    elif suffix in {".md", ".txt"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = str(path)
        doc.metadata["source_name"] = path.name
    return docs


def sanitize_document_for_public_server(document: Document) -> Document:
    if document.metadata.get("source_name") != "private_profile.md":
        return document

    safe_lines: list[str] = []
    for raw_line in document.page_content.splitlines():
        match = PROFILE_FACT_LINE.match(raw_line.strip())
        if match and match.group(1).strip() in PUBLIC_SENSITIVE_FIELDS:
            continue
        safe_lines.append(raw_line)
    document.page_content = "\n".join(safe_lines)
    return document


def load_documents(data_dir: Path, public_mode: bool = False) -> list[Document]:
    documents: list[Document] = []
    for path in iter_files(data_dir):
        loaded = load_file(path)
        if public_mode:
            loaded = [sanitize_document_for_public_server(doc) for doc in loaded]
        documents.extend(doc for doc in loaded if doc.page_content.strip())
    return documents


def build_index(
    data_dir: Path,
    index_dir: Path,
    embedding_model: str,
    force: bool,
    public_mode: bool = False,
) -> None:
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    files = list(iter_files(data_dir))
    if not files:
        raise SystemExit(f"No PDF/DOCX/PPTX/MD/TXT files found in: {data_dir}")

    docs = load_documents(data_dir, public_mode=public_mode)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise SystemExit("No text chunks were produced.")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            "device": "cpu",
            "local_files_only": embedding_cache_is_ready(),
        },
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=str(EMBEDDING_CACHE_DIR),
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    # Build into a sibling directory first. If loading or embedding fails, the
    # previously working index remains untouched.
    staging_dir = index_dir.with_name(f"{index_dir.name}.building")
    backup_dir = index_dir.with_name(f"{index_dir.name}.previous")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    vectorstore.save_local(str(staging_dir))

    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if index_dir.exists():
        index_dir.rename(backup_dir)
    try:
        staging_dir.rename(index_dir)
    except Exception:
        if backup_dir.exists() and not index_dir.exists():
            backup_dir.rename(index_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    print(f"files={len(files)}")
    print(f"documents={len(docs)}")
    print(f"chunks={len(chunks)}")
    print("chunk_size=900")
    print("chunk_overlap=100")
    length_method = "tiktoken/cl100k_base" if _ENCODER is not None else "offline_approximation"
    print(f"length_function={length_method}")
    print(f"embedding_model={embedding_model}")
    print("embedding_device=cpu")
    print("normalize_embeddings=True")
    print(f"index_dir={index_dir}")
    print(f"public_mode={public_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / "faiss_index")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Remove sensitive profile fields before building a public-server index.",
    )
    return parser.parse_args()


def main() -> None:
    configure_local_cache()
    args = parse_args()
    build_index(
        args.data_dir,
        args.index_dir,
        args.embedding_model,
        args.force,
        public_mode=args.public,
    )


if __name__ == "__main__":
    main()
