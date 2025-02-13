# QA Bot
# Chatbot System

## Overview
The **QA Bot** is a solution designed to process e-commerce sales data, generate insights, and assist with decision-making using AI-powered tools. It uses a flexible architecture that supports both Azure OpenAI and OpenRouter for LLM capabilities, along with multiple embedding options:
- **Azure OpenAI** (optional) for embeddings and chat completions
- **SentenceTransformers** (free alternative) for embeddings
- **OpenRouter** with Mistral-7B (free tier) for chat completions
- **Pinecone DB** for storing vectors

## Features
- Dual LLM support: Use either Azure OpenAI or OpenRouter's free tier
- Flexible embedding generation using either Azure OpenAI or free SentenceTransformers
- Vector storage with Pinecone DB
- Real-time interaction with a chatbot for general questions, tips, or advice
- Graceful fallback between services

## Architecture
The system uses a flexible **Data-driven architecture** that supports multiple LLM and embedding providers:
- Primary: Azure OpenAI (if configured)
- Secondary/Free: OpenRouter + SentenceTransformers
- Vector Database: Pinecone DB
- Method: **Retrieval-Augmented Generation (RAG)**

## Tools, Libraries and Programming Languages
- **LLM Providers**:
  - Azure OpenAI API (optional)
  - OpenRouter API (free tier with Mistral-7B)
- **Embedding Models**:
  - Azure OpenAI (optional)
  - SentenceTransformers (free alternative)
- **Vector Database**:
  - Pinecone DB
- **Python Libraries**:
  - `flask`: Web framework
  - `openai`: Azure OpenAI integration
  - `sentence-transformers`: Free embedding model
  - `pinecone-client`: Vector database
  - `pandas`: Data manipulation
  - `requests`: API calls
  - `numpy`: Numerical operations
- **Frontend**:
  - HTML/CSS/JavaScript
  - Bootstrap 5

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/QA_Bot.git
cd QA_Bot
```

### 2. Set up environment
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables
1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit the `.env` file with your credentials:
- For Azure OpenAI (optional):
  - `AZURE_API_KEY`
  - `AZURE_API_VERSION`
  - `AZURE_ENDPOINT`
- For OpenRouter (required if not using Azure):
  - `OPENROUTER_API_KEY` from [OpenRouter](https://openrouter.ai/)
- For Pinecone:
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_NAME`

### 4. Run the application
```bash
python app.py
```

## How It Works
1. The system attempts to use Azure OpenAI if configured
2. If Azure is not configured or fails:
   - Falls back to SentenceTransformers for embeddings
   - Uses OpenRouter's Mistral-7B for chat completions
3. All embeddings are stored in Pinecone DB
4. The chatbot uses RAG to provide context-aware responses

## License
This project is licensed under the MIT License - see the LICENSE file for details





