# Interview RAG

Assistant personnel de préparation aux entretiens, construit sur mes propres documents (CV, expériences professionnelles, préparations par entreprise, notes techniques). Plutôt qu'un chatbot générique, ce projet retrouve les passages pertinents dans ma base documentaire avant de générer une réponse — un pipeline RAG (*Retrieval-Augmented Generation*) construit "à la main", sans framework d'abstraction (pas de LangChain), pour comprendre chaque étape en détail.

Projet personnel et pédagogique : l'objectif principal était de comprendre le fonctionnement d'un RAG de bout en bout, pas seulement d'obtenir un outil fonctionnel.

## Architecture

```text
Documents (documents/)
      ↓
Chunking (découpage en morceaux avec chevauchement)
      ↓
Embeddings (OpenAI text-embedding-3-small)
      ↓
ChromaDB (base vectorielle, persistée sur disque)
      ↓
Question utilisateur
      ↓
Embedding de la question
      ↓
Recherche par similarité dans ChromaDB
      ↓
Chunks les plus pertinents
      ↓
Prompt augmenté (contexte + historique + question)
      ↓
OpenAI API (gpt-4o-mini)
      ↓
Réponse
```

## Stack

- Python
- API OpenAI (`responses.create` pour la génération, `embeddings.create` pour les embeddings)
- ChromaDB (base vectorielle locale, persistée dans `chroma_db/`)
- Streamlit (interface web)
- `python-dotenv` (variables d'environnement)

## Structure du code

```text
utils.py       → client OpenAI, connexion ChromaDB, get_embedding(), chunk_text()
ingest.py      → parcourt documents/, découpe et vectorise chaque fichier, alimente ChromaDB
app.py         → interface Streamlit (conversations séparées par entreprise, historique, RAG)
main.py        → version terminal du pipeline de recherche + réponse, utile pour des tests rapides
documents/     → mes documents sources (non versionné, voir Confidentialité)
chroma_db/     → base vectorielle générée par ingest.py (non versionné, régénérable)
```

## Fonctionnement du RAG

1. **Ingestion** (`ingest.py`, à relancer quand les documents changent) : chaque fichier `.txt` de `documents/` est découpé en chunks de taille fixe avec chevauchement, chaque chunk est transformé en vecteur (embedding), puis stocké dans ChromaDB avec des métadonnées dérivées de son chemin (`company`, `category`, `type`).
2. **Requête** (`app.py` / `main.py`, à chaque question) : la question est elle-même transformée en embedding, comparée aux vecteurs stockés (similarité cosinus), et les chunks les plus proches sont récupérés.
3. **Génération** : les chunks récupérés sont insérés dans un prompt avec la question, envoyé à l'API OpenAI. Le modèle est explicitement instruit de répondre uniquement à partir du contexte fourni.

Point important : le RAG n'entraîne pas le modèle. Aucune donnée personnelle n'est mémorisée par OpenAI d'une requête à l'autre — le contexte pertinent est rappelé et réinjecté à chaque appel.

## Choix techniques

- **Pas de LangChain** : le pipeline (chunking, embeddings, recherche, prompt augmenté) est écrit à la main pour comprendre chaque mécanisme, plutôt que de dépendre d'une abstraction.
- **Métadonnées dérivées du chemin de fichier** : la structure de `documents/` (`entreprises/<company>/`, `experiences/`, `technique/`) sert directement à construire les métadonnées ChromaDB, sans configuration manuelle par document.
- **Historique de conversation géré manuellement** : `client.responses.create()` est sans état — l'historique (stocké dans `st.session_state`) est reconstruit et réinjecté dans le prompt à chaque question, pas une mémoire native du modèle.
- **Pas de filtrage par métadonnées à la recherche** (pour l'instant) : la recherche interroge tout le corpus plutôt que de se limiter à l'entreprise de la conversation active, pour permettre au modèle de croiser des expériences transverses (ex : citer Retab dans une conversation sur une autre entreprise).

## Limites connues

Une évaluation qualitative sur quelques questions de test a mis en évidence deux limites concrètes :

- **Erreurs de retrieval** : pour certaines questions, le chunk le plus pertinent n'est pas toujours le mieux classé, ou peut être absent du top des résultats si sa formulation s'éloigne de la question posée.
- **Conflation** : le modèle peut occasionnellement mélanger des informations provenant de deux chunks différents, produisant une réponse plausible mais factuellement incorrecte (attribuer un fait à la mauvaise expérience). C'est une limite structurelle du RAG à surveiller, pas un bug isolé.

Ces deux limites illustrent qu'un RAG ne se juge pas en "ça marche / ça ne marche pas", mais sur plusieurs axes indépendants : qualité du retrieval, fidélité de la génération, et absence d'hallucination.

## Confidentialité

`documents/` (mes documents personnels) et `chroma_db/` (données dérivées) ne sont pas versionnés — voir `.gitignore`. Ce dépôt ne contient que le code du pipeline, pas mes données.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine avec :

```text
OPENAI_API_KEY=ta_clé_ici
```

Ingérer les documents (dossier `documents/` à fournir soi-même) :

```bash
python3 ingest.py
```

Lancer l'interface :

```bash
streamlit run app.py
```

## Suite possible (V2)

- Filtrage par métadonnées à la recherche (restreindre une conversation à une entreprise).
- Mémoire persistante de conversation (au-delà de la session en cours).
- Architecture agentique : un orchestrateur décidant dynamiquement quelle source interroger (RAG personnel, RAG entreprise, recherche web) plutôt qu'un pipeline déterministe fixe.
