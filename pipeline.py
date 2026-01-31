"""
rag pipeline using Ollama
"""
from typing import List, Dict
import requests


class RAGPipeline:
    """RAG pipeline using a local Ollama instance for generation"""
    def __init__(self, retriever, model_name="phi4-mini:3.8b", ollama_host="localhost"):
        """
        init rag pipeline
        retriever: DocumentRetriever instance
        model_name: Ollama model to use
        ollama_host: Ollama host
        """
        self.retriever = retriever
        self.model_name = model_name
        self.ollama_url = f"http://{ollama_host}:11434/api/generate"

        # verify ollama is running
        try:
            requests.get(f"http://{ollama_host}:11434/api/tags", timeout=5)
            print(f"ollama connected at {ollama_host}, using {model_name}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"ollama not running at {ollama_host}:11434. Run: ollama serve")

    def _create_prompt(self, query: str, context_docs: List) -> str:
        """
        create prompt with context
        query: str, question
        context_docs: list, documents to use for RAG context
        """
        context_parts = []
        for doc in context_docs:
            title = doc.metadata.get('title', '')
            content = doc.page_content.strip()
            context_parts.append(f"{title}: {content}")
        context = "\n\n".join(context_parts) # newlines are for debugging readability, doesnt affect RAG quality
        
        prompt = f"""Use the following context to answer the question. Be concise but informative. Only use the provided context. If unsure, say so. Only use the provided context. If unsure, say so.

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def answer(self, query: str, k: int = 5, use_reranking: bool = True) -> Dict:
        """
        generate answer using RAG
        query: str, question to answer
        k: int, k documents to use
        use_reranking, bool, use reranking or just get docs from FAISS
        """
        if use_reranking and self.retriever.use_reranker:
            # retrieve more initially, then rerank to top k
            # details in func retriever.retrieve_and_rerank
            retrieved_docs = self.retriever.retrieve_and_rerank(query, initial_k=k*4, top_k=k)
        else:
            retrieved_docs = self.retriever.retrieve(query, k=k)

        if not retrieved_docs:
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "sources": [],
                "num_chunks_retrieved": 0
            }

        prompt = self._create_prompt(query, retrieved_docs)

        # call ollama
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False, #https://docs.ollama.com/api/streaming
                "options": {
                    "temperature": 0.3,
                    "num_predict": 200 # max tokens to generate
                }
            },
            timeout=120
        )
        response.raise_for_status()
        answer = response.json()["response"]

        # collect sources
        # note: with small models and small sample, this is can be inaccurate and lowers perceieved quality
        # this can happen e.g. when the model answers correctly using its parametric knowledge
        #   instead of from retrieved docs
        sources = []
        used_titles = set()
        for doc in retrieved_docs:
            title = doc.metadata.get('title', 'Unknown title')
            if title not in used_titles: # no dupes
                sources.append({
                    "title": title,
                    "url": doc.metadata.get('url', ''),
                })
                used_titles.add(title)

        return {
            "query": query,
            "answer": answer.strip(),
            "sources": sources,
            "num_chunks_retrieved": len(retrieved_docs) #just for debug/info
        }
