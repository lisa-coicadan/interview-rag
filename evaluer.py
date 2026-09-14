from utils import collection, get_embedding, client

questions_test = [
    "Quel est le job chez [Entreprise] ? Résume les tâches attendues et les compétences demandées.",
]

for question in questions_test:
    print("=" * 80)
    print("QUESTION :", question)
    print("=" * 80)

    question_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    for i, doc in enumerate(results["documents"][0]):
        print(f"\n--- Chunk {i+1} (distance={results['distances'][0][i]:.3f}) ---")
        print("Metadata :", results["metadatas"][0][i])
        print(doc[:300], "...")

    contexte = "\n\n---\n\n".join(results["documents"][0])
    prompt = f"""Contexte récupéré :
{contexte}

Question :
{question}

Réponds uniquement à partir du contexte ci-dessus. Si le contexte ne contient pas la réponse, dis-le."""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    print("\n>>> RÉPONSE GÉNÉRÉE :")
    print(response.output_text)
    print("\n")