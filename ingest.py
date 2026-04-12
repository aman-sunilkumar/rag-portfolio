import yaml
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

with open("prompts/config.yaml") as f:
    config = yaml.safe_load(f)

with open("docs/metadata.json") as f:
    metadata_map = json.load(f)


def prepare_metadata(entry: dict) -> dict:
    """ChromaDB requires primitive types — convert lists to comma-separated strings."""
    m = dict(entry)
    if isinstance(m.get("role_relevance"), list):
        m["role_relevance"] = ",".join(m["role_relevance"])
    if isinstance(m.get("tags"), list):
        m["tags"] = ",".join(m["tags"])
    return m


def ingest_documents(docs_dir="docs/"):
    docs_path = Path(docs_dir)
    documents = []

    for file_path in sorted(docs_path.iterdir()):
        if file_path.name in ("metadata.json",) or file_path.suffix not in (".pdf", ".txt", ".md"):
            continue
        loader = PyPDFLoader(str(file_path)) if file_path.suffix == ".pdf" else TextLoader(str(file_path), encoding="utf-8")
        loaded = loader.load()
        file_meta = prepare_metadata(metadata_map.get(file_path.name, {}))
        file_meta["source"] = file_path.name
        for doc in loaded:
            doc.metadata.update(file_meta)
        documents.extend(loaded)
        print(f"  Loaded: {file_path.name} | type={file_meta.get('doc_type','?')} | team={file_meta.get('team','?')}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    print(f"\n{len(documents)} document(s) -> {len(chunks)} chunks")

    if Path("./chroma_db").exists():
        shutil.rmtree("./chroma_db")
        print("Cleared old ChromaDB.")

    embeddings = OpenAIEmbeddings(model=config["embeddings"]["model"])
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
    print(f"Stored {len(chunks)} chunks in ChromaDB.")


if __name__ == "__main__":
    ingest_documents()
