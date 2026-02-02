"""
embed docs and retrieve
"""
from typing import List, Dict, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from sentence_transformers import CrossEncoder

class DocumentRetriever:
    """retrieve using FAISS"""

    def __init__(self,
                 embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
                 use_reranker=True,
                 reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        init retriever
        embedding_model: str, HF model for embeddings
        use_reranker: bool, whether to use cross-encoder reranking
        reranker_model: str, cross-encoder model for reranking
        """
        print("loading embedding model")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # or cuda if there's nvidia gpu
            encode_kwargs={'normalize_embeddings': True}
            # normalize embeddings is a sentence-transformers parameter for normalizing vectors
            # makes cosine-sim == dot-product
            # make math easier for computer and avoids favoring longer chunks which have larger vectors
        )
        self.vector_store = None
        self.text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=20)  # ~1k chars with small overlap
        # 256 tokens should be enough for the model to identify what the chunk is about while keeping it resource effective
        # small overlap ensures answers are not cut off by chunking
        # example: "What does Tesla sell?" "Tesla designs, manufactures, and [CUT TO NEXT CHUNK] sells cars."
        # the first chunk doesn't say what Tesla sells, the second chunk doesn't say which company.

        # init cross-encoder for reranking
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            print(f"loading cross-encoder reranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model, max_length=512) #bert models have maximum token limit 512
            print("reranker loaded")

        print("retriever init success")
    
    def index_documents(self, articles: List[Dict], batch_size: int = 500,
                        checkpoint_path: str = None) -> None:
        """
        index and create vector store for doc objects. added checkpoints for robustness

        articles: list of dicts, expects 'id', 'title', 'text', 'url' keys
        batch_size: int, number of articles to process before checkpoint
        checkpoint_path: path to save checkpoints
        """
        total_articles = len(articles)
        total_chunks = 0

        for batch_start in range(0, total_articles, batch_size):
            batch_end = min(batch_start + batch_size, total_articles)
            batch = articles[batch_start:batch_end]

            # convert batch to langchain doc format
            documents = []
            for article in batch:
                # we add title to text
                # maybe improves quality by disambiguating https://arxiv.org/abs/2601.11863v1
                full_text = f"Title: {article['title']}\n\n{article['text']}"
                chunks = self.text_splitter.split_text(full_text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "chunk_id": i,
                            "source": article["id"],
                            "title": article["title"],
                            "url": article["url"]
                        }
                    )
                    documents.append(doc)

            total_chunks += len(documents)

            # add to vector store
            if self.vector_store is None:
                # create index based on docs and embedding
                # official docs seem to be missing?
                # https://www.myscale.com/blog/efficient-vector-stores-from-documents-using-faiss/
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)

            print(f"indexed {batch_end}/{total_articles} articles ({total_chunks} chunks)")

            if checkpoint_path:
                self.save(checkpoint_path)

        print(f"faiss vector store created with {total_chunks} total chunks")
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]: #TODO WHAT IS THIS
        """
        retrieve top k most relevant docs
        query: str, user input question
        k: int, number of docs to retrieve
        
        return: list of retrieved docs
        """
        if self.vector_store is None:
            raise ValueError("vector store not initialized, call index_documents().")
        return self.vector_store.similarity_search(query, k=k)

    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """
        rerank documents using cross-encoder
        query: str, user query
        documents: list of doc objects to rerank
        top_k: int, return top k documents (none: return all)

        return: list of (document, score) tuples sorted by relevance score
        """
        if not self.reranker:
            raise ValueError("reranker not initialized. Set use_reranker=True")

        if not documents:
            print("no documents found")
            return []

        # prepare query-document pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]

        # get relevance scores from cross-encoder
        scores = self.reranker.predict(pairs)

        # combine documents with scores and sort by score descending
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # return top k if specified
        if top_k:
            doc_score_pairs = doc_score_pairs[:top_k]

        return doc_score_pairs

    def retrieve_and_rerank(self, query: str, initial_k: int = 20, top_k: int = 5) -> List[Document]:
        """
        retrieve documents and rerank them using cross-encoder
        FAISS is good approximate but inaccurate, cross-encoding is more precise but expensive
        Get more docs than needed cheaply from FAISS
            then rerank those with cross-encoding and keep the best k

        query: str, user query
        initial_k: int, number of documents to retrieve initially before reranking
        top_k: int, number of documents to return after reranking

        return: list of top k reranked documents
        """
        # retrieve more documents than needed
        initial_docs = self.retrieve(query, k=initial_k)

        # step 2: rerank if reranker is available
        if self.use_reranker and self.reranker:
            reranked = self.rerank(query, initial_docs, top_k=top_k)
            # extract just documents
            return [doc for doc, score in reranked]
        else:
            # return top k docs from retrieval
            return initial_docs[:top_k]
    
    def save(self, path: str = "data/vector_store"):
        """save vector store"""
        if self.vector_store is None:
            raise ValueError("no vector store to store")
        self.vector_store.save_local(path)
        print(f"saved vector score to {path}")
    
    def load(self, path: str = "data/vector_store"):
        """load vector store from disk"""
        self.vector_store = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True # need for faiss 
            # https://stackoverflow.com/questions/78120202/the-de-serialization-relies-loading-a-pickle-file
        )
        print(f"loaded vector store from {path}")