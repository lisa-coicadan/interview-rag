import re

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="interview_rag")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

SECTION_PATTERN = re.compile(r"\n((?:[^\n]+\n)?[^\n]{1,120})\n[-=]{3,}\n")


def chunk_text(text, max_chars=1200):
    """Découpe le texte section par section, en utilisant la convention
    'Titre\\n------' déjà présente dans les documents. Une section trop
    longue est ensuite sous-découpée par paragraphe, sans jamais couper un
    mot en deux."""
    parts = SECTION_PATTERN.split(text)

    sections = []
    if parts[0].strip():
        sections.append(("(introduction)", parts[0].strip()))
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((title, content))

    chunks = []
    for title, content in sections:
        full = f"{title}\n{content}" if content else title
        if len(full) <= max_chars:
            chunks.append(full)
        else:
            chunks.extend(_split_long_section(title, content, max_chars))
    return chunks


def _split_long_section(title, content, max_chars):
    # Repaquette ligne par ligne (pas par paragraphe séparé par une ligne
    # vide, car ces documents enchaînent surtout des lignes simples) pour ne
    # jamais laisser le titre seul dans son chunk.
    lines = content.split("\n")
    sub_chunks = []
    current = title

    def flush_if_too_long(block):
        while len(block) > max_chars:
            cut = block.rfind(" ", 0, max_chars)
            if cut == -1:
                cut = max_chars
            sub_chunks.append(block[:cut])
            block = f"{title} (suite)\n" + block[cut:].lstrip()
        return block

    for line in lines:
        candidate = current + "\n" + line
        if len(candidate) <= max_chars:
            current = candidate
        else:
            sub_chunks.append(current)
            current = flush_if_too_long(f"{title} (suite)\n{line}")

    sub_chunks.append(current)
    return sub_chunks