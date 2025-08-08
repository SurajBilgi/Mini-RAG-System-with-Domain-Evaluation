import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import asyncio
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import textstat
import re
from utils import calculate_weighted_score, normalize_score, safe_divide


class RAGEvaluator:
    """Comprehensive evaluation system for RAG pipeline performance"""

    def __init__(self, similarity_model: str = "all-MiniLM-L6-v2"):
        self.similarity_model = SentenceTransformer(similarity_model)
        self.evaluation_history = []

    def evaluate_response(
        self,
        query: str,
        answer: str,
        source_documents: List[Document],
        context: str = "",
        ground_truth: str = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a RAG response

        Args:
            query: User question
            answer: Generated answer
            source_documents: Retrieved documents
            context: Additional context if available
            ground_truth: Reference answer for comparison (optional)

        Returns:
            Dictionary containing all evaluation metrics
        """
        start_time = datetime.now()

        # Calculate individual metrics
        relevance = self.calculate_relevance(query, answer, source_documents)
        faithfulness = self.calculate_faithfulness(answer, source_documents)
        context_precision = self.calculate_context_precision(query, source_documents)
        context_recall = self.calculate_context_recall(
            query, source_documents, ground_truth
        )
        readability = self.calculate_readability_metrics(answer)
        completeness = self.calculate_completeness(query, answer)
        coherence = self.calculate_coherence(answer)

        # Calculate composite scores
        retrieval_score = (context_precision + context_recall) / 2
        generation_score = (relevance + faithfulness + coherence) / 3
        overall_score = (retrieval_score + generation_score) / 2

        # Ground truth comparison if available
        ground_truth_metrics = {}
        if ground_truth:
            ground_truth_metrics = self.compare_with_ground_truth(answer, ground_truth)

        evaluation_result = {
            "timestamp": start_time.isoformat(),
            "query": query,
            "answer": answer,
            "metrics": {
                "relevance": relevance,
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "completeness": completeness,
                "coherence": coherence,
                "retrieval_score": retrieval_score,
                "generation_score": generation_score,
                "overall_score": overall_score,
                **readability,
                **ground_truth_metrics,
            },
            "source_count": len(source_documents),
            "answer_length": len(answer.split()),
            "processing_time": (datetime.now() - start_time).total_seconds(),
        }

        self.evaluation_history.append(evaluation_result)
        return evaluation_result

    def calculate_relevance(
        self, query: str, answer: str, source_documents: List[Document]
    ) -> float:
        """
        Calculate relevance between query and answer using semantic similarity
        Enhanced with context awareness
        """
        try:
            # Primary relevance: query-answer similarity
            query_embedding = self.similarity_model.encode([query])
            answer_embedding = self.similarity_model.encode([answer])

            primary_relevance = np.dot(query_embedding[0], answer_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(answer_embedding[0])
            )

            # Secondary relevance: query-sources similarity (context awareness)
            if source_documents:
                source_texts = [doc.page_content for doc in source_documents]
                source_embeddings = self.similarity_model.encode(source_texts)

                source_similarities = []
                for source_embedding in source_embeddings:
                    similarity = np.dot(query_embedding[0], source_embedding) / (
                        np.linalg.norm(query_embedding[0])
                        * np.linalg.norm(source_embedding)
                    )
                    source_similarities.append(similarity)

                context_relevance = (
                    np.mean(source_similarities) if source_similarities else 0.0
                )

                # Weighted combination
                relevance = 0.7 * primary_relevance + 0.3 * context_relevance
            else:
                relevance = primary_relevance

            return float(normalize_score(relevance, -1, 1))

        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return 0.0

    def calculate_faithfulness(
        self, answer: str, source_documents: List[Document]
    ) -> float:
        """
        Calculate faithfulness by measuring how well the answer is grounded in source documents
        Uses fine-grained sentence-level analysis
        """
        try:
            if not source_documents:
                return 0.0

            # Split answer into sentences
            answer_sentences = self._split_into_sentences(answer)
            if not answer_sentences:
                return 0.0

            # Get source text
            source_text = " ".join([doc.page_content for doc in source_documents])
            source_sentences = self._split_into_sentences(source_text)

            if not source_sentences:
                return 0.0

            # Calculate faithfulness for each answer sentence
            faithfulness_scores = []

            answer_embeddings = self.similarity_model.encode(answer_sentences)
            source_embeddings = self.similarity_model.encode(source_sentences)

            for answer_embedding in answer_embeddings:
                # Find best match in source documents
                similarities = []
                for source_embedding in source_embeddings:
                    similarity = np.dot(answer_embedding, source_embedding) / (
                        np.linalg.norm(answer_embedding)
                        * np.linalg.norm(source_embedding)
                    )
                    similarities.append(similarity)

                max_similarity = max(similarities) if similarities else 0.0
                faithfulness_scores.append(max_similarity)

            # Average faithfulness across all sentences
            overall_faithfulness = np.mean(faithfulness_scores)
            return float(normalize_score(overall_faithfulness, -1, 1))

        except Exception as e:
            print(f"Error calculating faithfulness: {e}")
            return 0.0

    def calculate_context_precision(
        self, query: str, source_documents: List[Document]
    ) -> float:
        """
        Calculate context precision: how relevant are the retrieved documents to the query
        """
        try:
            if not source_documents:
                return 0.0

            query_embedding = self.similarity_model.encode([query])
            source_texts = [doc.page_content for doc in source_documents]
            source_embeddings = self.similarity_model.encode(source_texts)

            relevance_scores = []
            for source_embedding in source_embeddings:
                similarity = np.dot(query_embedding[0], source_embedding) / (
                    np.linalg.norm(query_embedding[0])
                    * np.linalg.norm(source_embedding)
                )
                # Consider a document relevant if similarity > threshold
                relevance_scores.append(1.0 if similarity > 0.5 else 0.0)

            precision = np.mean(relevance_scores)
            return float(precision)

        except Exception as e:
            print(f"Error calculating context precision: {e}")
            return 0.0

    def calculate_context_recall(
        self, query: str, source_documents: List[Document], ground_truth: str = None
    ) -> float:
        """
        Calculate context recall: how well does the retrieved context cover the information needed
        """
        try:
            if not source_documents:
                return 0.0

            # If ground truth is available, use it for more accurate recall calculation
            if ground_truth:
                ground_truth_embedding = self.similarity_model.encode([ground_truth])
                source_texts = [doc.page_content for doc in source_documents]
                source_embeddings = self.similarity_model.encode(source_texts)

                # Check how much of the ground truth information is covered by sources
                coverage_scores = []
                for source_embedding in source_embeddings:
                    similarity = np.dot(ground_truth_embedding[0], source_embedding) / (
                        np.linalg.norm(ground_truth_embedding[0])
                        * np.linalg.norm(source_embedding)
                    )
                    coverage_scores.append(similarity)

                recall = max(coverage_scores) if coverage_scores else 0.0
            else:
                # Approximate recall using query-context coverage
                query_embedding = self.similarity_model.encode([query])
                source_texts = [doc.page_content for doc in source_documents]
                combined_context = " ".join(source_texts)
                context_embedding = self.similarity_model.encode([combined_context])

                recall = np.dot(query_embedding[0], context_embedding[0]) / (
                    np.linalg.norm(query_embedding[0])
                    * np.linalg.norm(context_embedding[0])
                )

            return float(normalize_score(recall, -1, 1))

        except Exception as e:
            print(f"Error calculating context recall: {e}")
            return 0.0

    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate comprehensive readability metrics"""
        try:
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "automated_readability_index": textstat.automated_readability_index(
                    text
                ),
                "coleman_liau_index": textstat.coleman_liau_index(text),
                "gunning_fog": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
                "reading_time": textstat.reading_time(text, ms_per_char=14.69),
            }
        except Exception as e:
            print(f"Error calculating readability: {e}")
            return {"flesch_reading_ease": 0.0}

    def calculate_completeness(self, query: str, answer: str) -> float:
        """
        Calculate how completely the answer addresses the query
        Based on question type analysis and answer coverage
        """
        try:
            # Identify question type and expected elements
            question_types = {
                "what": ["definition", "description", "explanation"],
                "how": ["process", "method", "steps"],
                "why": ["reason", "cause", "explanation"],
                "when": ["time", "date", "period"],
                "where": ["location", "place"],
                "who": ["person", "people", "entity"],
            }

            query_lower = query.lower()
            answer_lower = answer.lower()

            # Check for question words and expected answer types
            completeness_score = 0.5  # Base score

            for question_word, expected_elements in question_types.items():
                if question_word in query_lower:
                    # Check if answer contains expected elements
                    element_coverage = sum(
                        1 for element in expected_elements if element in answer_lower
                    )
                    completeness_score += (
                        element_coverage / len(expected_elements)
                    ) * 0.3
                    break

            # Check answer length appropriateness
            query_words = len(query.split())
            answer_words = len(answer.split())

            # Ideal answer length should be proportional to query complexity
            ideal_length = max(20, query_words * 3)
            length_score = min(1.0, answer_words / ideal_length)
            completeness_score += length_score * 0.2

            return float(min(1.0, completeness_score))

        except Exception as e:
            print(f"Error calculating completeness: {e}")
            return 0.5

    def calculate_coherence(self, answer: str) -> float:
        """
        Calculate coherence of the answer using linguistic features
        """
        try:
            sentences = self._split_into_sentences(answer)
            if len(sentences) < 2:
                return 1.0  # Single sentence is coherent by default

            # Calculate sentence-to-sentence similarity
            sentence_embeddings = self.similarity_model.encode(sentences)
            coherence_scores = []

            for i in range(len(sentence_embeddings) - 1):
                similarity = np.dot(
                    sentence_embeddings[i], sentence_embeddings[i + 1]
                ) / (
                    np.linalg.norm(sentence_embeddings[i])
                    * np.linalg.norm(sentence_embeddings[i + 1])
                )
                coherence_scores.append(similarity)

            # Check for transition words and logical flow
            transition_words = [
                "however",
                "therefore",
                "moreover",
                "furthermore",
                "additionally",
                "consequently",
                "meanwhile",
                "similarly",
                "in contrast",
                "for example",
            ]

            transition_score = 0.0
            answer_lower = answer.lower()
            for word in transition_words:
                if word in answer_lower:
                    transition_score += 0.1

            transition_score = min(1.0, transition_score)

            # Combine scores
            semantic_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
            overall_coherence = (
                0.7 * normalize_score(semantic_coherence, -1, 1)
                + 0.3 * transition_score
            )

            return float(overall_coherence)

        except Exception as e:
            print(f"Error calculating coherence: {e}")
            return 0.5

    def compare_with_ground_truth(
        self, answer: str, ground_truth: str
    ) -> Dict[str, float]:
        """Compare generated answer with ground truth reference"""
        try:
            answer_embedding = self.similarity_model.encode([answer])
            ground_truth_embedding = self.similarity_model.encode([ground_truth])

            # Semantic similarity
            semantic_similarity = np.dot(
                answer_embedding[0], ground_truth_embedding[0]
            ) / (
                np.linalg.norm(answer_embedding[0])
                * np.linalg.norm(ground_truth_embedding[0])
            )

            # BLEU-like n-gram overlap (simplified)
            answer_words = set(answer.lower().split())
            ground_truth_words = set(ground_truth.lower().split())

            word_overlap = len(answer_words.intersection(ground_truth_words)) / len(
                ground_truth_words.union(answer_words)
            )

            return {
                "ground_truth_semantic_similarity": float(
                    normalize_score(semantic_similarity, -1, 1)
                ),
                "ground_truth_word_overlap": float(word_overlap),
            }

        except Exception as e:
            print(f"Error comparing with ground truth: {e}")
            return {
                "ground_truth_semantic_similarity": 0.0,
                "ground_truth_word_overlap": 0.0,
            }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex"""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def get_performance_summary(
        self, time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive performance summary

        Args:
            time_window: Number of recent evaluations to consider (None for all)
        """
        if not self.evaluation_history:
            return {"message": "No evaluations available"}

        # Filter by time window if specified
        evaluations = (
            self.evaluation_history[-time_window:]
            if time_window
            else self.evaluation_history
        )

        # Extract metrics
        metrics_data = [eval_result["metrics"] for eval_result in evaluations]
        df = pd.DataFrame(metrics_data)

        # Calculate summary statistics
        summary = {
            "total_evaluations": len(evaluations),
            "time_period": {
                "start": evaluations[0]["timestamp"],
                "end": evaluations[-1]["timestamp"],
            },
            "average_scores": {
                "overall_score": df["overall_score"].mean(),
                "relevance": df["relevance"].mean(),
                "faithfulness": df["faithfulness"].mean(),
                "context_precision": df["context_precision"].mean(),
                "context_recall": df["context_recall"].mean(),
                "completeness": df["completeness"].mean(),
                "coherence": df["coherence"].mean(),
                "retrieval_score": df["retrieval_score"].mean(),
                "generation_score": df["generation_score"].mean(),
            },
            "score_distribution": {
                "excellent": len(df[df["overall_score"] >= 0.8]),
                "good": len(
                    df[(df["overall_score"] >= 0.6) & (df["overall_score"] < 0.8)]
                ),
                "fair": len(
                    df[(df["overall_score"] >= 0.4) & (df["overall_score"] < 0.6)]
                ),
                "poor": len(df[df["overall_score"] < 0.4]),
            },
            "readability": {
                "avg_flesch_reading_ease": df["flesch_reading_ease"].mean(),
                "avg_reading_time": df.get("reading_time", pd.Series([0])).mean(),
            },
            "performance_trends": self._calculate_trends(df),
        }

        return summary

    def _calculate_trends(self, df: pd.DataFrame) -> Dict[str, str]:
        """Calculate performance trends over time"""
        if len(df) < 3:
            return {"overall": "insufficient_data"}

        # Simple trend calculation using linear regression slope
        x = np.arange(len(df))
        trends = {}

        key_metrics = ["overall_score", "relevance", "faithfulness"]

        for metric in key_metrics:
            if metric in df.columns:
                slope = np.polyfit(x, df[metric], 1)[0]
                if slope > 0.01:
                    trends[metric] = "improving"
                elif slope < -0.01:
                    trends[metric] = "declining"
                else:
                    trends[metric] = "stable"

        return trends

    def export_evaluation_report(self, filepath: str, format: str = "json"):
        """Export comprehensive evaluation report"""
        report = {
            "evaluation_summary": self.get_performance_summary(),
            "detailed_evaluations": self.evaluation_history,
            "export_timestamp": datetime.now().isoformat(),
        }

        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        elif format == "csv":
            # Export flattened metrics to CSV
            metrics_data = []
            for eval_result in self.evaluation_history:
                row = {
                    "timestamp": eval_result["timestamp"],
                    "query": eval_result["query"],
                    "answer_length": eval_result["answer_length"],
                    "source_count": eval_result["source_count"],
                    **eval_result["metrics"],
                }
                metrics_data.append(row)

            df = pd.DataFrame(metrics_data)
            df.to_csv(filepath, index=False)

        return filepath
