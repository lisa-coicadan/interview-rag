import os
from utils import client, collection, get_embedding, chunk_text

DOSSIER_A_INGERER = "documents"

def get_metadata(filepath):
    relative = os.path.relpath(filepath, "documents")
    parts = relative.split(os.sep)
    nom_fichier = os.path.splitext(parts[-1])[0]

    if parts[0] == "entreprises":
        meta = {"category": "entreprise", "company": parts[1], "type": nom_fichier}
    elif parts[0] == "experiences":
        meta = {"category": "experience", "type": nom_fichier}
    elif parts[0] == "technique":
        meta = {"category": "technique", "topic": nom_fichier}
    else:
        meta = {"category": "general", "type": nom_fichier}
    meta["source_file"] = filepath
    return meta

for root, dirs, files in os.walk(DOSSIER_A_INGERER):
    for filename in files:
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(root, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            texte = f.read()

        # Supprime les anciens chunks de ce fichier avant de le réingérer :
        # sans ça, si le fichier a rétréci, les chunks en trop de la version
        # précédente restent orphelins dans la base.
        collection.delete(where={"source_file": filepath})

        chunks = chunk_text(texte)
        metadata = get_metadata(filepath)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath}_{i}"
            embedding = get_embedding(chunk)
            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        print(f"{filepath} : {len(chunks)} chunks ajoutés")

print("Total dans la collection :", collection.count())