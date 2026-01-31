"""
evaluate RAG pipeline quality
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from rouge_score import rouge_scorer
from retriever import DocumentRetriever
from pipeline import RAGPipeline


class RAGEvaluator:
    """evaluate retrieval and answer quality for RAG pipeline"""

    def __init__(self, pipeline: RAGPipeline):
        """
        init evaluator
        pipeline: RAGPipeline instance to evaluate
        """
        self.pipeline = pipeline
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        # https://en.wikipedia.org/wiki/ROUGE_(metric). use_stemmer enables Porter stemmer

    def evaluate_retrieval(self, query: str, retrieved_docs: List,
                          relevant_doc_titles: List[str], k: int = 5) -> Dict:
        """
        evaluate retrieval quality for a single query
        query: str, user query
        retrieved_docs: list of retrieved doc objects
        relevant_doc_titles: list of ground truth relevant document titles
        k: int, number of documents to consider

        return: dict with retrieval metrics
        """
        # get titles of retrieved docs
        retrieved_titles = [doc.metadata.get('title', '').lower()
                          for doc in retrieved_docs[:k]]
        
        relevant_titles_norm = [title.lower() for title in relevant_doc_titles]

        # calculate metrics
        # recall@k: fraction of relevant docs that were retrieved
        # this uses substring match, which works but is error prone (e.g. "apple" and "pineapple" match)
        # for demo, this works, for actual evaluation test_set should have doc_id for each relevant doc
        # count how many relevant docs were found (not how many matches)
        found_relevant = set()
        for rel in relevant_titles_norm:
            if any(rel in title or title in rel for title in retrieved_titles):
                found_relevant.add(rel)

        retrieved_relevant = len(found_relevant)
        recall_at_k = retrieved_relevant / len(relevant_doc_titles) if relevant_doc_titles else 0

        # precision@k: fraction of retrieved docs that are relevant
        # count how many retrieved docs match at least one relevant doc
        num_relevant_retrieved = sum(1 for title in retrieved_titles
                                    if any(rel in title or title in rel
                                          for rel in relevant_titles_norm))
        precision_at_k = num_relevant_retrieved / k if k > 0 else 0

        # MRR: mean reciprocal rank. position of first relevant doc
        # similar substring match as recall@k, fragile but works for demo
        first_relevant_pos = None
        for i, title in enumerate(retrieved_titles, 1):
            if any(rel in title or title in rel for rel in relevant_titles_norm):
                first_relevant_pos = i
                break

        mrr = 1.0 / first_relevant_pos if first_relevant_pos else 0.0

        return {
            'recall_at_k': recall_at_k,
            'precision_at_k': precision_at_k,
            'mrr': mrr,
            'retrieved_relevant_count': retrieved_relevant,
            'total_relevant': len(relevant_doc_titles)
        }

    def evaluate_answer_quality(self, generated_answer: str,
                                reference_answer: str) -> Dict:
        """
        evaluate generated answer quality using rouge-l f1 score.
        ROUGE-L score. longest common subsequence
        https://en.wikipedia.org/wiki/ROUGE_(metric)
        f1: 2x (precision*recall)/(precision+recall)
        generated_answer: str, answer from RAG pipeline
        reference_answer: str, ground truth reference answer

        return: dict with answer quality metrics
        """
        rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
        # rouge scores should report rouge-l recall, rouge-l precision, rouge-l f1
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure

        return {
            'rouge_l_f1': rouge_l_f1,
        }

    def evaluate_test_set(self, test_set_path: str = "data/test_set.json",
                         k: int = 5) -> Dict:
        """
        evaluate pipeline on test set
        test_set_path: path to test set JSON
        k: number of documents to retrieve

        return: dict with eval metrics
        """
        # load test set
        with open(test_set_path, 'r', encoding='utf-8') as f:
            test_set = json.load(f)

        print(f"evaluating on {len(test_set)} test queries...")

        retrieval_metrics = []
        answer_metrics = []

        for i, test_case in enumerate(test_set, 1):
            query = test_case['query']
            relevant_docs = test_case['relevant_docs']
            reference_answer = test_case.get('reference_answer', '')

            print(f"\n[{i}/{len(test_set)}] Query: {query}:")

            # get RAG answer
            result = self.pipeline.answer(query, k=k)
            generated_answer = result['answer']
            retrieved_docs = self.pipeline.retriever.retrieve_and_rerank(
                query, initial_k=k*4, top_k=k
            )

            # evaluate retrieval
            ret_metrics = self.evaluate_retrieval(
                query, retrieved_docs, relevant_docs, k=k
            )
            retrieval_metrics.append(ret_metrics)

            print(f"Recall@{k}={ret_metrics['recall_at_k']:.2f}, "
                  f"Precision@{k}={ret_metrics['precision_at_k']:.2f}, "
                  f"MRR={ret_metrics['mrr']:.2f}")

            # evaluate answer quality if reference exists
            if reference_answer:
                ans_metrics = self.evaluate_answer_quality(
                    generated_answer, reference_answer
                )
                answer_metrics.append(ans_metrics)
                print(f"ROUGE-L F1={ans_metrics['rouge_l_f1']:.2f}")

        # aggregate metrics
        print("Aggregated Results:")

        avg_retrieval = {
            'avg_recall_at_k': np.mean([m['recall_at_k'] for m in retrieval_metrics]),
            'avg_precision_at_k': np.mean([m['precision_at_k'] for m in retrieval_metrics]),
            'avg_mrr': np.mean([m['mrr'] for m in retrieval_metrics]),
        }

        avg_answer = {}
        if answer_metrics:
            avg_answer = {
                'avg_rouge_l_f1': np.mean([m['rouge_l_f1'] for m in answer_metrics]),
            }

        print(f"\n Aggregated Metrics (k={k}):")
        print(f"Average Recall@{k}: {avg_retrieval['avg_recall_at_k']:.3f}")
        print(f"Average Precision@{k}: {avg_retrieval['avg_precision_at_k']:.3f}")
        print(f"Average MRR: {avg_retrieval['avg_mrr']:.3f}")

        if avg_answer:
            print(f"Average ROUGE-L F1: {avg_answer['avg_rouge_l_f1']:.3f}")

        return {
            'retrieval_metrics': avg_retrieval,
            'answer_metrics': avg_answer,
            'num_queries': len(test_set)
        }


def main():
    """run evaluation on test set"""

    vector_store_path = "data/vector_store"
    test_set_path = "data/test_set.json"

    if not Path(vector_store_path).exists():
        print(f"Vector store not found at {vector_store_path}.")
        sys.exit(1)

    if not Path(test_set_path).exists():
        print(f"Test set not found at {test_set_path}")
        sys.exit(1)

    # load pipeline
    print("\nLoading pipeline...")
    retriever = DocumentRetriever(use_reranker=True)
    retriever.load(vector_store_path)
    pipeline = RAGPipeline(retriever)
    print("Pipeline loaded\n")

    # run evaluation
    print("Running evaluator...\n")
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.evaluate_test_set(test_set_path, k=5)

    # save results
    results_path = "data/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
