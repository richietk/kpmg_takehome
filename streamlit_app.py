"""
streamlit frontend
"""
import streamlit as st
import requests
from typing import Dict


API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Q&A demo kpmg",
    layout="wide"
)


st.title("Demo q&a for kpmg")

def check_api_health() -> Dict:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def query_api(query: str, k: int, use_reranking: bool) -> Dict:
    """send query to API"""
    try:
        response = requests.post(
            f"{API_URL}/answer",
            json={
                "query": query,
                "k": k,
                "use_reranking": use_reranking
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


# Sidebar for settings
with st.sidebar:
    health = check_api_health()
    if health.get("status") == "healthy":
        st.success("API Connected")
    elif health.get("status") == "degraded":
        st.warning("API Degraded")
        if not health.get("pipeline_loaded"):
            st.error("Pipeline not loaded")
        if not health.get("ollama_connected"):
            st.error("Ollama not connected")
    else:
        st.error("❌ API Offline")
        st.stop()

    num_docs = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="How many relevant chunks to retrieve for context"
    )

    use_reranking = st.checkbox(
        "Use reranking",
        value=True,
        help="Use cross-encoder to rerank retrieved documents"
    )

# Main query interface
st.header("Type your question")

# Query input
query = st.text_input(
    "Your question:",
    placeholder="Type your question here..."
)

# Submit button
if st.button("Submit", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Generating answer..."):
            result = query_api(query, num_docs, use_reranking)

        if result:
            st.subheader("Answer:")
            st.markdown(result["answer"])

            # Optionally display sources, but it hallucinates often
            # st.subheader("Sources")
            # sources = result["sources"]
            # if sources:
            #     for i, source in enumerate(sources, 1):
            #         with st.expander(f"{i}. {source['title']}"):
            #             if source.get('url'):
            #                 st.markdown(f"[Link for Wikipedia]({source['url']})")
            # else:
            #     st.info("No sources found")
