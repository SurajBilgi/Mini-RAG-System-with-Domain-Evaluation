import os
import hashlib
import json
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path


# Document processing utilities
def get_file_hash(file_content: bytes) -> str:
    """Generate MD5 hash for file content to detect duplicates"""
    return hashlib.md5(file_content).hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Validate file extension against allowed types"""
    extension = filename.split(".")[-1].lower()
    return extension in allowed_types


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters but keep punctuation
    text = "".join(char for char in text if char.isprintable())
    return text.strip()


def split_text_smart(text: str, max_length: int, overlap: int = 0) -> List[str]:
    """Smart text splitting that respects sentence boundaries"""
    if len(text) <= max_length:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_length

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Find the last sentence boundary within the chunk
        chunk = text[start:end]
        last_period = chunk.rfind(".")
        last_newline = chunk.rfind("\n")

        boundary = max(last_period, last_newline)
        if boundary > 0 and boundary > len(chunk) * 0.5:  # Don't split too early
            end = start + boundary + 1

        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# Evaluation utilities
def calculate_weighted_score(
    scores: Dict[str, float], weights: Dict[str, float]
) -> float:
    """Calculate weighted average of evaluation scores"""
    weighted_sum = sum(
        scores.get(metric, 0) * weight for metric, weight in weights.items()
    )
    total_weight = sum(weights.values())
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to 0-1 range"""
    return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))


def categorize_readability(flesch_score: float) -> str:
    """Categorize readability score into human-readable labels"""
    if flesch_score >= 90:
        return "Very Easy"
    elif flesch_score >= 80:
        return "Easy"
    elif flesch_score >= 70:
        return "Fairly Easy"
    elif flesch_score >= 60:
        return "Standard"
    elif flesch_score >= 50:
        return "Fairly Difficult"
    elif flesch_score >= 30:
        return "Difficult"
    else:
        return "Very Difficult"


def calculate_response_quality(
    relevance: float, faithfulness: float, readability: float
) -> Dict[str, Any]:
    """Calculate overall response quality metrics"""
    # Normalize readability (Flesch Reading Ease) to 0-1 scale
    normalized_readability = normalize_score(readability, 0, 100)

    # Calculate weighted average
    weights = {"relevance": 0.4, "faithfulness": 0.4, "readability": 0.2}
    overall_score = calculate_weighted_score(
        {
            "relevance": relevance,
            "faithfulness": faithfulness,
            "readability": normalized_readability,
        },
        weights,
    )

    # Determine quality category
    if overall_score >= 0.8:
        quality = "Excellent"
    elif overall_score >= 0.6:
        quality = "Good"
    elif overall_score >= 0.4:
        quality = "Fair"
    else:
        quality = "Poor"

    return {
        "overall_score": overall_score,
        "quality_category": quality,
        "readability_category": categorize_readability(readability),
    }


# Data persistence utilities
def save_chat_history(chat_history: List[Dict], filepath: str):
    """Save chat history to JSON file"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving chat history: {e}")


def load_chat_history(filepath: str) -> List[Dict]:
    """Load chat history from JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading chat history: {e}")
    return []


def export_performance_data(performance_data: List[Dict], format: str = "csv") -> str:
    """Export performance data to file"""
    if not performance_data:
        return None

    df = pd.DataFrame(performance_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "csv":
        filename = f"rag_performance_{timestamp}.csv"
        df.to_csv(filename, index=False)
    elif format == "json":
        filename = f"rag_performance_{timestamp}.json"
        df.to_json(filename, orient="records", indent=2, date_format="iso")
    else:
        raise ValueError(f"Unsupported format: {format}")

    return filename


# UI helper utilities
def format_timestamp(timestamp) -> str:
    """Format timestamp for display"""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_sources(sources: List[str]) -> str:
    """Format source list for display"""
    if not sources:
        return "No sources"

    unique_sources = list(set(sources))
    if len(unique_sources) == 1:
        return unique_sources[0]
    elif len(unique_sources) <= 3:
        return ", ".join(unique_sources)
    else:
        return f"{', '.join(unique_sources[:2])}, and {len(unique_sources)-2} more"


def create_performance_summary(performance_data: List[Dict]) -> Dict[str, Any]:
    """Create summary statistics from performance data"""
    if not performance_data:
        return {}

    df = pd.DataFrame(performance_data)

    return {
        "total_queries": len(df),
        "avg_relevance": df["relevance"].mean(),
        "avg_faithfulness": df["faithfulness"].mean(),
        "avg_latency": df["latency"].mean(),
        "min_latency": df["latency"].min(),
        "max_latency": df["latency"].max(),
        "queries_per_hour": (
            len(df)
            / ((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600)
            if len(df) > 1
            else 0
        ),
    }


# Error handling utilities
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default


def validate_openai_response(response: Dict) -> bool:
    """Validate OpenAI API response structure"""
    required_fields = ["choices"]
    return all(field in response for field in required_fields)


def handle_file_processing_error(error: Exception, filename: str) -> str:
    """Generate user-friendly error message for file processing errors"""
    error_type = type(error).__name__

    error_messages = {
        "FileNotFoundError": f"File '{filename}' not found.",
        "PermissionError": f"Permission denied to access '{filename}'.",
        "UnicodeDecodeError": f"Unable to read '{filename}'. File encoding not supported.",
        "ValueError": f"Invalid content in '{filename}'. Please check the file format.",
        "MemoryError": f"File '{filename}' is too large to process.",
    }

    return error_messages.get(
        error_type, f"Error processing '{filename}': {str(error)}"
    )


# Domain-specific utilities
def extract_art_metadata(text: str) -> Dict[str, Any]:
    """Extract art-specific metadata from text"""
    metadata = {}

    # Common art-related keywords to look for
    art_keywords = {
        "artist": ["artist", "painter", "sculptor", "creator"],
        "date": ["date", "year", "created", "painted"],
        "medium": ["oil", "watercolor", "bronze", "marble", "canvas"],
        "style": ["baroque", "renaissance", "modern", "contemporary", "impressionist"],
        "dimensions": ["cm", "inches", "feet", "meters"],
    }

    text_lower = text.lower()

    for category, keywords in art_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Extract context around the keyword
                idx = text_lower.find(keyword)
                context = text[max(0, idx - 50) : idx + 100]
                metadata[category] = metadata.get(category, []) + [context.strip()]

    return metadata


def generate_domain_questions(domain: str, content_keywords: List[str]) -> List[str]:
    """Generate domain-specific sample questions based on content"""
    base_questions = {
        "art": [
            f"What artistic techniques are mentioned in relation to {', '.join(content_keywords[:2])}?",
            f"How do the artworks described relate to their historical period?",
            f"What influences can be identified in the artistic styles discussed?",
        ],
        "literature": [
            f"What themes emerge from the analysis of {', '.join(content_keywords[:2])}?",
            f"How do the literary techniques contribute to the overall meaning?",
            f"What is the significance of the historical context mentioned?",
        ],
        "technical": [
            f"How do the systems described in the documentation work?",
            f"What are the key implementation details for {', '.join(content_keywords[:2])}?",
            f"What best practices are recommended in this technical content?",
        ],
    }

    return base_questions.get(
        domain,
        [
            "What are the main topics covered?",
            "Can you summarize the key points?",
            "How are the concepts related?",
        ],
    )
