
## README

DEMO: http://localhost:8501/

DOCS: http://localhost:8000/docs

STATS: http://localhost:8000/stats

HEALTH: http://localhost:8000/health

OLLAMA STATUS: http://localhost:11434/

If it is working:

- http://localhost:8501/ should load
- `curl http://localhost:8000/health` should show `"status":"healthy","pipeline_loaded":true,"ollama_connected":true`
- `curl http://localhost:11434/api/tags` should list phi4-mini


## Architecture

- ingest.py processes the truncated data: stores 10k articles as json, keeps title, text, url for each article.
- retriever.py splits text into 256 tokens per chunk with a 20 token overlap, builds embedding with paraphrase-MiniLM-L3-v2 model, uses FAISS index for vector storage with cosine-similarity.
- in retriever.py, FAISS first retrieves the top20 candidates then the cross encoder reranks the top-k chunks
- pipeline.py constructs the prompt, calls ollama and returns the answer.

Retriever uses semantic search. For embeddings, paraphrase-MiniLM-L3-v2 is used, for vector store FAISS.
For the reranker, ms-marco-MiniLM-L-6-v2 is used. 
For the generator, ollama phi4-mini (3.8B params) is used.

`evaluator.py`: evaluate via 10 LLM generated test queries. report retrieval metrics (e.g. did we retrieve the relevant chunks) and answer quality metrics (is LLMs answer similar to ground truth answer?)

## NOTES

**Requirements:**
- Local Ollama with phi4-mini:3.8b model (setup script handles pulling automatically)
- Source data from Google Drive (see Data Setup section)

**Data Source:**
- Original dataset: [Kaggle - Wikipedia Finance](https://www.kaggle.com/datasets/akhiltheerthala/wikipedia-finance)
- "Manually" truncated to 10k articles using `utils/truncate_data.py`
- Pre-processed files available via Google Drive for convenience



## Justifications

`sentence-transformers/paraphrase-MiniLM-L3-v2` is used for embedding because it is small and fast for real-time retrieval. It is trained on paraphrase data and runs efficiently on CPU, so it is easy to setup locally without GPU requirements

`FAISS`is used for vector store because it is industry standard, fast, uses ANN which should be effective, memory efficient, scalable, and is self-contained with no external dependencies. Alternatives such as Pinecone require external services.

`ms-marco-MiniLM-L-6-v2` is used as a cross-encoder reranker. the bi-encoder is good, and the cross encoder should increase precision to capture relevance by scoring query-document pairs jointly. from the 20 candidates from the bi-encored, top-k chunks are reranked with cross-encoder. I chose this model because it is trained on MS MARCO and sufficient for local demo.

`phi4-mini:3.8b` Ollama is used for the llm. It is chosen because ollama FOSS, local (therefore private), makes it easy to integrate with the application and models can be swapped. Phi4-mini is chosen because its small and fast therefore sufficient for the demo, has sufficient quality and has low resource use for the demo.

FastAPI is used for the API framework. it is fast, modern, and has automatic Swagger UI

Streamlit is the UI framework because it is fast, python-native, good for demos to sketch up a minimal proof-of-concept.

## Data Setup

The source dataset and pre-built artifacts are available via Google Drive for convenience:

**Google Drive:** `https://drive.google.com/drive/folders/1CUTd7Gsog19155mEWyaWmCfPte--s-4m?usp=sharing`

It is recommended to download pre-built files to skip processing time:

1. Download from Google Drive:
   - `wikipedia_finance_trunc.jsonl` → place in `data/`
   - `wiki_sample.json` → place in `data/`
   - `vector_store/` folder → place in `data/`

2. Your directory structure should look like:
   ```
   data/
   ├── wikipedia_finance_trunc.jsonl
   ├── wiki_sample.json
   ├── vector_store/
   │   ├── index.faiss
   │   └── index.pkl
   ├── test_set.json
   └── evaluation_results.json
   ```

3. Then proceed to Setup section below

Alternatively, building from scratch shows the full pipeline but building the vector store adds ~10 minutes on the first run.

In this case, place `wikipedia_finance_trunc.jsonl` in `data/` and run the setup script.

The `setup_docker.sh` script will automatically:
1. Convert `wikipedia_finance_trunc.jsonl` → `wiki_sample.json`
2. Build the vector store from scratch


## Setup

1. **Install prerequisites:**
   - Docker & Docker Compose
   - [Ollama](https://ollama.ai/) installed locally

2. **Get the data** (see Data Setup section above)

3. **Start Ollama server** (keep this running in a terminal):
   ```bash
   ollama serve
   ```

4. **Run setup script** (in a new terminal):
   ```bash
   ./setup_docker.sh
   ```
   This script will:
   - Build and start Docker containers
   - Automatically pull phi4-mini:3.8b model if needed
   - Process data if needed (Option B)
   - Set up the API and Streamlit UI

5. **Access the demo:**
   - **Streamlit UI:** http://localhost:8501
   - **API docs:** http://localhost:8000/docs


## Troubleshooting

- Check API health: `curl http://localhost:8000/health`
- Check container logs: `docker logs rag-api` or `docker logs rag-streamlit`
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- If not, start it: `ollama serve` (in a separate terminal)
- If model not found, pull manually: `ollama pull phi4-mini:3.8b`

**If data processing takes too long or fails:**
- Use Option A from Data Setup section (download pre-built files from Google Drive)
- This skips the 5-10 minute vector store build process

**To restart everything:**
```bash
docker compose down
./setup_docker.sh
```

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
{"query":"What is machine learning?",
"answer":"Machine learning is a subset of artificial intelligence where...",
"sources": [
  {"title":"Credit card fraud","url":"https://en.wikipedia.org/wiki/Credit_card_fraud"},
  {"title":"Google DeepMind","url":"https://en.wikipedia.org/wiki/Google_DeepMind"},
  {"title":"Caper AI","url":"https://en.wikipedia.org/wiki/Caper_AI"}
],
"num_chunks_retrieved":3
}

## Evaluation

For this demo, evaluation is on 10 ChatGPT-generated "ground-truth" queries. For a production system, a separate ground-truth dataset should be used.

```bash
python evaluation.py
```

Since evaluation takes a while, results are available on data/evaluation_results.json


Metrics calculated:

**retrieval**

- Recall@k (retrieved relevant docs / all relevant docs)

- Precision@k (retrieved relevant docs / all retrieved docs)

- MRR (average of 1/rank of the first relevant doc across all queries)

**answer quality**

- ROUGE-L F1 (overlap between generated and reference answers using longest common subsequence)

tuning is possible relatively separately in the 2 metric categories (retrieval can be tuned with top-k value and reranking, as well as increasing train data, answer quality via prompting, model choice, temperature)

## suggestions for end-user accessibility

- deploy as a cloud-hosted web app on a cloud-platform with auth, so its accessible without local setup, automatically scales and restricts access to authenticated users. An auth layer like OAuth2 could be implemented, using a better LLM (e.g. a managed API like Claude) would be useful.
- a chatbot interface could be implemented in Slack, Teams, etc, for internal users and employees. Could be deployed on internal documentation. a Slack bot or an MSFT Teams bot, connecting it to backend, adding slash commands or a @mention command.
- the fastAPI could be used as a library for applications to embed into existing workflows and enable programmatic access. FastAPI could be published, and an SDK created. Could be useful for internal work e.g. for dashboarding, statistics.
- implement it production-ready. This demo uses local FAISS files (not a managed vector database service), has no query caching, and runs the LLM locally via Ollama. For production, a managed vector DB (e.g. Pinecone), query caching, a more capable LLM API, rate limiting, monitoring/logging, and security enhancements against attacks like prompt injection, should be implemented.

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
└── utils/
    ├── truncate_data.py       # util script to truncate original data
```