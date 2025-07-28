import os

# Project configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")

# Model configuration
MODEL_NAME = "gemma-3-1b-it"
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_NAME)

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval settings
TOP_K = 3
SIMILARITY_THRESHOLD = 0.5

# Generation settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.95
REPETITION_PENALTY = 1.05

# File paths
PDF_FILE = os.path.join(DATA_DIR, "CELEX_32016R0679_EN_TXT.pdf") 