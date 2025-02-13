import openai  # Ensure you have installed the `openai` package
from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
import pandas as pd
from openai import AzureOpenAI
from datetime import datetime
import tiktoken
import re
import secrets
import pinecone  # Pinecone for vector database
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import requests
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))

# Initialize SentenceTransformer model for multilingual support
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
EMBEDDING_DIMENSION = 1024  # dimension for multilingual-e5-large model

# Azure OpenAI Configuration (Optional)
azure_client = None
if os.getenv("AZURE_API_KEY"):
    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT")
    )

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # Your OpenRouter API key
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Pinecone Configuration
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get index name from environment variables
index_name = os.getenv("PINECONE_INDEX_NAME")
if not index_name:
    raise ValueError("PINECONE_INDEX_NAME environment variable is not set")

try:
    # Try to get the index first
    index = pc.Index(index_name)
    print(f"Successfully connected to existing index '{index_name}'")
except Exception as e:
    print(f"Creating new index '{index_name}' with dimension {EMBEDDING_DIMENSION}")
    # Create a new index
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric='cosine',
        spec={"serverless": {"cloud": "aws", "region": "us-west-2"}}
    )
    time.sleep(10)  # Wait for index to be ready
    index = pc.Index(index_name)

print(f"Index '{index_name}' is ready.")

@app.route('/')
def home():
   session.clear()
   return render_template('index.html')

@app.route('/embed')
def embed():
    """Serve the embed code page"""
    server_url = os.getenv('SERVER_URL', 'http://localhost:5000')
    return render_template('embed.html', server_url=server_url)

@app.route('/ask', methods=['POST'])
def ask_route():
    data = request.get_json()       # Gets the JSON data sent in the request body.
    user_query = data.get('query')  # Extracts the query key from the JSON data.

    response_message = ask(user_query, token_budget=4096 - 100, print_message=False)
    return jsonify({"response": response_message})


# Cleans text by removing HTML tags and extra whitespace.
def clean_text(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'[\t\r\n]+', '', cleaned_text)
    return cleaned_text

def generate_embeddings(text, model="text-embedding-3-large-model"):
    """Generate embeddings using either Azure OpenAI or SentenceTransformer"""
    if azure_client:
        try:
            return azure_client.embeddings.create(input=[text], model=model).data[0].embedding
        except Exception as e:
            print(f"Azure embedding failed: {e}")
    
    # Fallback to SentenceTransformer
    embeddings = embedding_model.encode([text])[0]
    return embeddings.tolist()

def cosine_similarity(a, b):
    # np.dot(a, b): Computes the dot product between the two vectors.
    # np.linalg.norm(a): Computes the Euclidean norm (magnitude) of vector a.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


import ast  # for converting embeddings saved as strings back to arrays
import numpy as np

def strings_ranked_by_relatedness(query: str, top_n: int = 100):
    

    """Search Pinecone for similar embeddings based on the query"""
    query_embedding = generate_embeddings(query)
    
    # Upsert the query embedding into Pinecone
    # We need a unique ID for the vector (e.g., a UUID or simple incrementing counter)
    vector_id = str(hash(query))  # Simple hash to generate a unique ID for the query
    metadata = {"text": query}
    
    # Upsert the vector into Pinecone
    index.upsert(vectors=[(vector_id, query_embedding, metadata)])

    # Query Pinecone for similar embeddings
    results = index.query(
        vector=query_embedding,
        top_k=top_n,
        include_metadata=True
    )

    # Extract text and scores
    strings = [item["metadata"]["text"] for item in results["matches"]]
    relatednesses = [item["score"] for item in results["matches"]]

    return strings, relatednesses

#  Returns the number of tokens in a string based on the model being used (e.g., GPT-4).
def num_tokens(text: str, model: str = "gpt-4") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, _ = strings_ranked_by_relatedness(query)
    introduction = 'You are a customer assistant that answers questions or give information about text entered by the user from the given data. The Characters before the fisrt space are the Campaign Ids.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nConcat:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    model: str = "gpt-4",
    token_budget: int = 4096 - 100,
    print_message: bool = False,
) -> str:
    """Answers a query using either Azure OpenAI or OpenRouter"""
    message = query_message(query, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You are a customer assistant that answers based on the questions that are ask."},
        {"role": "user", "content": message},
    ]

    if azure_client:
        try:
            response = azure_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )
            response_message = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Azure chat completion failed: {e}")
            response_message = None
    
    # If Azure fails or is not configured, use OpenRouter
    if not azure_client or not response_message:
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://your-site.com",  # Required for OpenRouter
                "X-Title": "QA Bot"  # Optional, but helps OpenRouter understand your app
            }
            
            data = {
                "model": "mistralai/mistral-7b-instruct",  # Free tier model
                "messages": messages,
                "temperature": 0
            }
            
            response = requests.post(OPENROUTER_URL, headers=headers, json=data)
            response.raise_for_status()
            response_message = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"OpenRouter chat completion failed: {e}")
            return ["I apologize, but I'm having trouble processing your request at the moment. Please try again later."]

    # Split the response into meaningful paragraphs
    formatted_response = response_message.split('\n\n')
    return formatted_response

if __name__ == '__main__':
    # Get port from environment variable or default to 10000
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
