# Legal Assistant Project

A legal document assistant that uses retrieval-augmented generation to answer questions about legal documents.

## Project Structure

```
Project/
├── data/                    # Data files (PDFs, documents)
│   └── CELEX_32016R0679_EN_TXT.pdf
├── embeddings/              # Embedding-related files
│   ├── __init__.py
│   └── embedding_manager.py
├── generator/               # Text generation components
│   ├── __init__.py
│   └── generator.py
├── logger/                  # Logging functionality
│   ├── __init__.py
│   └── logger.py
├── logs/                    # Log files
├── metrics/                 # Evaluation metrics
│   ├── __init__.py
│   └── evaluator.py
├── retriever/               # Document retrieval components
│   ├── __init__.py
│   └── retriever.py
├── gemma-3-1b-it/          # Model files
├── config.py               # Configuration file
├── legal_assistant.py      # Main assistant module
├── runner.py               # Main execution script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the assistant:
```bash
python runner.py
```

## Usage

The legal assistant will load the legal document and allow you to ask questions about it. The system will:

1. Retrieve relevant sections from the document
2. Generate comprehensive answers with specific article references
3. Stream the response in real-time

Type 'quit' to exit the application.

## Components

- **Retriever**: Handles document processing and similarity search
- **Generator**: Manages text generation with the language model
- **Embeddings**: Manages document embeddings for similarity search
- **Logger**: Provides logging functionality
- **Metrics**: Tracks and evaluates response quality 
