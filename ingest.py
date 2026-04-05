import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

with open("prompts/config.yaml") as f:
    config = yaml.safe_load(f)


def ingest_documents(docs_dir="docs/"):
    docs_path = Path(docs_dir)
    documents = []

    for file_path in docs_path.iterdir():
        if file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            continue
        loaded = loader.load()
        documents.extend(loaded)
        print(f"  Loaded: {file_path.name} ({len(loaded)} section(s))")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"\nTotal: {len(documents)} document(s) -> {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model=config["embeddings"]["model"])
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at ./chroma_db")
    return vectorstore


if __name__ == "__main__":
    ingest_documents()
