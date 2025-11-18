# AmbedkarGPT - Intern Task

This is a simple RAG (Retrieval-Augmented Generation) Q&A system using:

- LangChain
- ChromaDB
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Ollama + Mistral 7B

## How It Works
1. Loads the speech from `speech.txt`
2. Splits into chunks
3. Converts chunks into embeddings
4. Stores embeddings in ChromaDB
5. Retrieves relevant chunks based on a question
6. Uses Mistral via Ollama to generate answers

## Setup Instructions

### 1. Create virtual environment
### 2. Install dependencies
### 3. Install Ollama
### 4. Run the program
## Notes
- Ollama cannot run on Android/iPhone. Use a desktop/laptop.
- All code is self-contained for evaluation.
