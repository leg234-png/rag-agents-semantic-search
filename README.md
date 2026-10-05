# RAG Agents & Semantic Search – Agent IA avec recherche sémantique et outils

Un assistant de service client qui répond à partir d'une **base documentaire** (12 documents,
41 passages) pour une agence de voyage **fictive**, Atlas Voyages. Il combine trois briques :

1. **Recherche** : BM25 (lexicale), recherche **dense par embeddings** (OpenAI ou Mistral) et
   **hybride** par Reciprocal Rank Fusion, évaluées sur 30 questions annotées.
2. **Agent ReAct** orchestré avec **LangGraph**, qui choisit ses outils (recherche documentaire,
   calcul des frais d'annulation, conversion de devises), observe les résultats puis répond.
3. **Réponses vérifiées** : chaque réponse cite ses sources, et les citations sont contrôlées contre
   les passages réellement consultés. Une citation inventée fait signaler la réponse comme non vérifiée.

Le tout est exposé via une **API FastAPI**.

## Architecture

```
                 ┌───────────────────────────────────────────────┐
 question ─────> │ decide (LLM) : quelle action ?                │ <──────────────┐
                 └──────┬───────────────────────────┬────────────┘                │
                        │ outil                     │ final_answer / budget      │ observation
                        v                           v                            │
              ┌──────────────────┐        ┌──────────────────────┐               │
              │ act : outil      │ ───────┼──────────────────────┼───────────────┘
              │ - search_docs    │        │ finish : vérifie que │
              │ - cancellation   │        │ les citations ont    │ ──> réponse + sources
              │ - convert        │        │ bien été observées   │     + verified
              └──────────────────┘        └──────────────────────┘

 search_docs ──> BM25 ─┐
                       ├─ RRF (hybride) ──> top-k passages
           embeddings ─┘   (cache disque des embeddings du corpus)
```

Garde-fous : budget de 5 actions, arguments d'outils validés par Pydantic (les erreurs sont
renvoyées à l'agent pour qu'il se corrige), calculs délégués à des fonctions déterministes plutôt
qu'au LLM, citations vérifiées.

## Évaluation de la recherche

30 questions formulées comme un client les poserait, souvent sans les mots de la documentation
(« Ma valise fait 28 kilos, ça me coûte combien en plus ? » → section *Excédent de poids*).
Chaque question est annotée avec le passage pertinent.

| Méthode | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---|---|---|---|
| BM25 | 0,400 | 0,767 | 0,900 | 0,598 |
| Dense (`text-embedding-3-small`) | _à compléter_ | | | |
| Hybride BM25 + dense (RRF) | _à compléter_ | | | |

> `python -m rag_agents.evaluate retrieval --methods bm25 dense hybrid`

BM25 retrouve le bon passage dans le top 5 pour 90 % des questions, mais en 1re position pour
seulement 40 % : les paraphrases ne partagent pas assez de mots avec la documentation. C'est ce
que les embeddings et la fusion hybride doivent améliorer.

## Évaluation de l'agent

| Métrique | Définition | Résultat |
|---|---|---|
| Taux de citation pertinente | La réponse cite le passage annoté | _à compléter_ |
| Taux de réponses vérifiées | Toutes les citations ont été observées | _à compléter_ |
| Étapes moyennes | Nombre d'actions avant la réponse | _à compléter_ |
| Sélection d'outil | Le bon outil est appelé sur 4 cas de calcul | _à compléter_ |
| Exactitude des calculs | Outil appelé avec les bons arguments (bonne valeur) | _à compléter_ |

> `python -m rag_agents.evaluate agent --method hybrid`, réponses détaillées dans `reports/agent_answers.csv`.

## Installation et utilisation

```bash
git clone https://github.com/leg234-png/rag-agents-semantic-search.git
cd rag-agents-semantic-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # renseigner la clé OpenAI ou Mistral

python -m rag_agents.evaluate retrieval --methods bm25 dense hybrid
python -m rag_agents.evaluate agent --method hybrid

uvicorn rag_agents.api:app --reload        # API sur http://localhost:8000/docs
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "J ai payé 480 € et j annule 20 jours avant, combien je récupère ?"}'

pytest -q                                  # 17 tests sans réseau (faux embedder, faux LLM)
```

## Structure

```
rag_agents/
  retrieval.py   découpage par section, BM25, recherche dense (cache), hybride RRF
  tools.py       outils de l'agent + validation des arguments
  agent.py       agent ReAct LangGraph : decide → act → ... → finish (vérification des citations)
  evaluate.py    métriques de recherche (Recall@k, MRR) et de l'agent (citations, outils)
  api.py         API FastAPI : /search, /ask, /health
  factory.py     assemblage des composants
  llm.py         client OpenAI / Mistral : chat JSON validé + embeddings
data/docs/       base documentaire (Markdown)
data/eval_questions.jsonl   30 questions annotées
tests/           tests unitaires
```

## Limites et pistes

- Corpus de petite taille et fictif : le protocole reste valable sur une vraie base documentaire,
  avec plus de questions annotées.
- Pistes : reranking par cross-encoder, réécriture de requête (HyDE), évaluation de la fidélité des
  réponses par un LLM juge, streaming des réponses dans l'API.

## Auteur

Emmanuel Wandji – ENSTA, Institut Polytechnique de Paris ·
[LinkedIn](https://www.linkedin.com/in/emmanuel-wandji) · [GitHub](https://github.com/leg234-png)
