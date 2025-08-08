import os
from typing import Dict, Any


class Config:
    """Configuration class for the RAG system"""

    # Default model settings
    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
    DEFAULT_TEMPERATURE = 0.1

    # Chunking settings
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    MAX_CHUNK_SIZE = 2000
    MIN_CHUNK_SIZE = 500

    # Retrieval settings
    DEFAULT_K_DOCUMENTS = 3
    MAX_K_DOCUMENTS = 10

    # Memory settings
    MEMORY_WINDOW_SIZE = 10

    # File upload settings
    SUPPORTED_FILE_TYPES = ["pdf", "txt", "csv", "doc", "docx"]
    MAX_FILE_SIZE_MB = 100

    # Vector store settings
    VECTOR_STORE_PERSIST_DIR = "./chroma_db"

    # Evaluation settings
    SIMILARITY_MODEL = "all-MiniLM-L6-v2"

    # UI settings
    PAGE_TITLE = "Mini RAG System"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"

    # Performance settings
    STREAMING_ENABLED = True
    MAX_RESPONSE_TIME = 30  # seconds

    # Available OpenAI models
    AVAILABLE_MODELS = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
    ]

    # Available embedding models
    AVAILABLE_EMBEDDING_MODELS = [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]

    @staticmethod
    def get_openai_api_key() -> str:
        """Get OpenAI API key from environment variables"""
        return os.getenv("OPENAI_API_KEY", "")

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate OpenAI API key format"""
        return api_key.startswith("sk-") and len(api_key) > 20

    @staticmethod
    def get_sample_questions() -> Dict[str, list]:
        """Get sample questions for different domains"""
        return {
            "general": [
                "What are the main topics covered in these documents?",
                "Can you summarize the key findings?",
                "What are the most important points mentioned?",
                "How are the concepts in these documents related?",
                "What conclusions can be drawn from this information?",
            ],
            "art_history": [
                "What artistic techniques are described in these documents?",
                "How did this artist's style evolve over time?",
                "What influences can be seen in this artwork?",
                "What is the historical context of this piece?",
                "How does this work compare to other pieces from the same period?",
            ],
            "literature": [
                "What are the major themes in this literary work?",
                "How do the characters develop throughout the story?",
                "What literary devices are employed by the author?",
                "What is the significance of the setting?",
                "How does this work relate to its historical context?",
            ],
            "technical": [
                "How does this system work?",
                "What are the implementation details?",
                "What are the requirements and dependencies?",
                "How can I configure this feature?",
                "What are the best practices mentioned?",
            ],
            "research": [
                "What methodologies were used in this study?",
                "What were the main findings and results?",
                "What are the limitations of this research?",
                "How does this study compare to previous work?",
                "What are the implications of these findings?",
            ],
        }

    @staticmethod
    def get_readability_thresholds() -> Dict[str, Dict[str, float]]:
        """Get readability score thresholds for evaluation"""
        return {
            "flesch_reading_ease": {
                "very_easy": 90.0,
                "easy": 80.0,
                "fairly_easy": 70.0,
                "standard": 60.0,
                "fairly_difficult": 50.0,
                "difficult": 30.0,
                "very_difficult": 0.0,
            },
            "flesch_kincaid_grade": {
                "elementary": 6.0,
                "middle_school": 9.0,
                "high_school": 13.0,
                "college": 16.0,
                "graduate": 20.0,
            },
        }

    @staticmethod
    def get_evaluation_weights() -> Dict[str, float]:
        """Get weights for different evaluation metrics"""
        return {"relevance": 0.4, "faithfulness": 0.4, "readability": 0.2}
