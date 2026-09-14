from utils import collection, get_embedding, client

question = "Quelles expériences montrent ma discipline personnelle ?"
question_embedding = get_embedding(question)

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5
)

chunks_trouves = results["documents"][0]
contexte = "\n\n---\n\n".join(chunks_trouves)

prompt = f"""Contexte récupéré :
{contexte}

Question :
{question}

Réponds uniquement à partir du contexte ci-dessus. Si le contexte ne contient pas la réponse, dis-le."""

response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt
)

print(response.output_text)