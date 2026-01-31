
## README

DEMO: http://localhost:8501/
DOCS: http://localhost:8000/docs
STATS: http://localhost:8000/stats
HEALTH: http://localhost:8000/health
OLLAMA STATUS: http://localhost:11434/

readme for the assignment
retrieval from 10k finance/economics wikipedia articles, rank via semantic search, cross encoder reranking, use local phi4-mini:3.8b to answer.

demo: run ./setup_docker.sh.
access on http://localhost:8501

The original data has been downloaded manually and truncated to 10k articles using utils/truncate_data.py
https://www.kaggle.com/datasets/akhiltheerthala/wikipedia-finance


## Architecture
user query -> streamlit UI (port 8501) -> http request -> fastAPI backend (port 8000) -> retriever -> reranker -> generator -> answer served

Retriever uses semantic search. For embeddings, paraphrase-MiniLM-L3-v2 is used, for vector store FAISS.
For the reranker, ms-marco-MiniLM-L-6-v2 is used. 
For the generator, ollama phi4-mini (3.8B params) is used.

evaluator: evaluate via 10 LLM generated test queries. report retrieval metrics (e.g. did we retrieve the relevant chunks) and answer quality metrics (is LLMs answer similarto ground truth answer?)

## Justifications

`sentence-transformers/paraphrase-MiniLM-L3-v2` is used for embedding because it is small for demo and fast for real-time retrieval. It is trained on paraphrase data, embeds using cos-sim search and runs on CPU for this demo.

`FAISS`is used for vector store because it is industry standard, fast, uses ANN which should be effective, memory efficient, scalable, and is self-contained with no external dependencies. Alternatives such as Pinecone require external services

`ms-marco-MiniLM-L-6-v2` is used as a cross-encoder reranker. the bi-encoder is good, and the cross encoder should increase precision to capture relevance by scoring query-document pairs jointly. from the 20 candidates from the bi-encored, top-k chunks are reranked with cross-encoder. I chose this model because it is trained on MS MARCO and sufficient for local demo.

`phi4-mini:3.8b` Ollama is used for the llm. It is chosen because ollama FOSS, local (therefore private), makes it easy to integrate with the application and models can be swapped. Phi4-mini is chosen because its small and fast therefore sufficient for the demo, has sufficient quality and has low resource use for the demo.

FastAPI is used for the API framework. it is fast, modern, and has automatic Swagger UI

Streamlit is the UI framework because it is fast, python-native, good for demos to sketch up a minimal proof-of-concept, and it auto-loads on code changes.

## Prerequisites

1. Docker & Docker Compose installed
2. Ollama installed locally with model pulled:
   ```bash
   ollama pull phi4-mini:3.8b
   ollama serve
   ```

3. start docker: 
   ```bash
   ./setup_docker.sh
   ```

**Streamlit UI:** http://localhost:8501
**FastAPI Docs:** http://localhost:8000/docs
**API Health:** http://localhost:8000/health


## Usage

### Via Streamlit UI

1. Open http://localhost:8501
2. Adjust settings in sidebar:
   - Number of chunks to retrieve (1-10)
   - Enable/disable cross-encoder reranking
3. Enter question in the text box, click "Get Answer"


**CLI:**
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 3,
    "use_reranking": true
  }'
```

**Expected response:**
```json
{
  "query": "What is Python?",
  "answer": "Python is a high-level programming language...",
  "sources": [
    {"title": "Python (programming language)", "url": "https://..."},
    {"title": "Guido van Rossum", "url": "https://..."}
  ],
  "num_chunks_retrieved": 3
}
```

## Evaluation

For this demo, evaluation is on 10 ChatGPT-generated "ground-truth" queries. For a production system, a separate ground-truth dataset should be used.

```bash
# Ensure Ollama is running and vector store exists
python evaluation.py
```

Since evaluation takes a while, results are available on data/evaluation_results.json


Metrics calculated:
**retrieval**
Recall@k (retrieved relevant docs / all relevant docs)
Precision@k (retrieved relevant docs / all retrieved docs)
MRR (position of first relevant result)
**answer quality**
- ROUGE-L F1 (overlap between generated and reference answers (longest common subsequence))

Interpretations:
good retrieval + bad answer qual: low answer quality, might benefit from a better model to generate answer
bad retrieval + good answer qual: model might hallucinate from parametric knowledge

tuning is possible relatively separately in the 2 metric categories (retrieval can be tuned with top-k value and reranking, as well as increasing train data, answer quality via prompting, model choice, temperature)

## suggestions for end-user accessibility

- deploy as a cloud-hosted web app on a cloud-platform with auth, so its accessible without local setup, automatically scales and restricts access to authenticated users. Docker containers could be displayed to a cloud service, an auth layer like OAuth2 could be implemented, using a better LLM (e.g. a managed API like Claude)
- a chatbot interface could be implemented in Slack, Teams, etc, for internal users and employees. Could be deployed on internal documentation. a Slack bot or an MSFT Teams bot, connecting it to fastAPI backend, maybe adding slash commands or a @mention command.
- providing a RESTAPI. The fastAPI could be used as a library for applications to embed into existing workflows and enable programmatic access. FastAPI could be published, and a python/JS/TS SDK created. Could be useful for internal work e.g. for dashboarding, statistics, or to automate knowledge base updates.
- implement it production-ready, as in this demo, vector store is in memory, it is not cached, the vector database is not a professionally managed one like Pinecone or Qdrant etc. caching for frequent queries could be added, as well as a load-balancer. Model could be maybe quantized, depending on needs and constraints. security should be enhanced to protect against prompt injections and such.

## Project Structure
```
kpmg_takehome/
├── api.py                  # FastAPI backend application
├── streamlit_app.py        # Streamlit web interface
├── pipeline.py             # RAG pipeline orchestration
├── retriever.py            # Document retrieval & reranking
├── ingest.py               # Data ingestion from Wikipedia
├── evaluation.py           # Evaluation metrics
├── requirements.txt
├── Dockerfile 
├── docker-compose.yml
├── setup_docker.sh         # Docker setup helper
├── README.md
└── data/
    ├── wiki_sample.json       # processed wikipedia articles
    ├── vector_store/          # FAISS index & embeddings
    │   ├── index.faiss
    │   └── index.pkl
    ├── test_set.json          # AI-generated test-set for demo
    └── evaluation_results.json # Evaluation output (already ran for demo)
```

---

- ingest.py processes the truncated data: stores 10k articles as json, keeps title, text, url for each article.
- retriever.py splits text into 256 tokens per chunk with a 20 token overlap, builds embedding with paraphrase-MiniLM-L3-v2 model, uses FAISS index for vector storage with cosine-similarity. For this demo run, this resulted in 38k chunks.
- in retriever.py, FAISS first retrieves the top20 candidates then the cross encoder reranks the top-k chunks
- pipeline.py constructs the prompt, calls ollama and returns the answer.