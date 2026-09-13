import os
import streamlit as st
from utils import collection, get_embedding, client

DOSSIER_ENTREPRISES = "documents/entreprises"
entreprises = sorted([
    d for d in os.listdir(DOSSIER_ENTREPRISES)
    if os.path.isdir(os.path.join(DOSSIER_ENTREPRISES, d)) and not d.startswith(".")
])
conversations = entreprises + ["General", "Technical learning"]

st.sidebar.title("Conversations")
conversation_active = st.sidebar.radio("Choisis une conversation :", conversations)

st.title(f"Interview Copilot — {conversation_active}")

if "historiques" not in st.session_state:
    st.session_state.historiques = {conv: [] for conv in conversations}

historique = st.session_state.historiques[conversation_active]

for q, r, p in historique:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(r)
        with st.expander("Voir le prompt exact envoyé au LLM"):
            st.code(p)

question = st.chat_input("Ta question :")

if question:
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    chunks_trouves = results["documents"][0]
    contexte_rag = "\n\n---\n\n".join(chunks_trouves)

    historique_texte = ""
    for q_prec, r_prec, p_prec in historique:
        historique_texte += f"Question précédente : {q_prec}\nRéponse précédente : {r_prec}\n\n"

    prompt = f"""Historique de la conversation :
{historique_texte if historique_texte else "(aucun échange précédent)"}

Contexte récupéré pour la question actuelle :
{contexte_rag}

Question actuelle :
{question}

Réponds uniquement à partir du contexte et de l'historique ci-dessus. Si l'information n'y est pas, dis-le."""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    historique.append((question, response.output_text, prompt))
    st.rerun()