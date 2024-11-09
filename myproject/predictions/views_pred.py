from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import csv
import torch
import chromadb
import tiktoken
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.INFO)
device = 'cpu'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Initialize ChromaDB client with persistent storage
persist_directory = "chapter_embeddings_persist"
client1 = chromadb.Client(settings=chromadb.config.Settings(persist_directory=persist_directory))
collection = client1.get_or_create_collection(name="chapter_embeddings")

    # Initialize Hugging Face question-answering pipeline for CPU only
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)


class OpenAIClient:
    def __init__(self, api_key):
        openai.api_key = api_key  # Set the API key for the OpenAI library

    def rephrase(self, answer, question, context):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that rephrases and expands answers."},
            {"role": "user", "content": (
                f"Based on the question and context, rephrase and expand brieffly the answer to be more detailed and directly relevant.\n"
                f"Question: {question}\n"
                f"Context: {context}\n"
                f"Answer: {answer}\n"
                f"Refined Answer:")}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Change to "gpt-4" if using GPT-4
                messages=messages,
                max_tokens=100,
                temperature=0.7
            )
            refined_answer = response.choices[0].message['content'].strip()
            return refined_answer
        except Exception as e:
            print(f"Error generating refined answer: {e}")
            return answer  # Return the original answer if there's an error
openai.api_key="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"
def split_text_with_overlap(text, max_tokens=150, overlap_tokens=50):
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = start + max_tokens - overlap_tokens

    return chunks
def clean_text(text):
    """Clean text by removing unnecessary characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\*\n]', ' ', text)   # Remove asterisks and newlines
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text
def create_embedding(text_chunk):
    try:
        embedding = embedding_model.encode(text_chunk, convert_to_tensor=True, device=device)
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Error creating embedding for chunk: {text_chunk[:50]}... : {e}")
        return None



def save_embeddings_batch(embeddings_data, collection):
    """Save embeddings batch to ChromaDB collection with debug logging."""
    if embeddings_data:
        try:
            documents = [item['document'] for item in embeddings_data]
            embeddings = [item['embedding'] for item in embeddings_data]
            ids = [item['id'] for item in embeddings_data]
            metadatas = [item['metadata'] for item in embeddings_data]
                
            collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
            logging.debug(f"Batch of {len(embeddings_data)} documents added to ChromaDB.")
        except Exception as e:
            logging.error(f"Error saving batch to ChromaDB: {e}")
        finally:
            embeddings_data.clear()  # Clear batch to avoid memory buildup
def retrieve_and_rank_relevant_chunks(query, top_k=5):
    query_embedding = create_embedding(clean_text(query))
    if query_embedding is None:
        print("Failed to create embedding for query.")
        return [], []

    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k * 2)  # Retrieve more results initially
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

            # Rank chunks by similarity to the query
        chunk_embeddings = [create_embedding(doc) for doc in documents]
        similarity_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
        ranked_results = sorted(zip(similarity_scores, documents, metadatas), reverse=True, key=lambda x: x[0])

            # Return top-ranked chunks only
        top_chunks = [doc for _, doc, _ in ranked_results[:top_k]]
        top_metadata = [meta for _, _, meta in ranked_results[:top_k]]

        return top_chunks, top_metadata

    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        return [], []

def generate_huggingface_answer(query, context):
    try:
        result = qa_model(question=query, context=context)
        return result['answer']
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Could not generate answer."          
@api_view(['POST'])
def predict(request):
    # Initialize OpenAI client
    openai_client = OpenAIClient(api_key="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")

    query = request.data.get('input')
    file_path = r'C:\Users\Dell\Desktop\react\myproject\DataPreprocessed.csv'
    data = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',', on_bad_lines="skip", usecols=['title', 'cleaned_content'])

    print("Starting embedding creation")

    # Retrieve relevant chunks and metadata
    chunks, metadata = retrieve_and_rank_relevant_chunks(query, top_k=5)
    
    # Provide a default context if no relevant chunks are found
    if not chunks:
        default_context = "This is the default context to help provide relevant information if no specific context is found in the data."
        combined_context = default_context
    else:
        combined_context = "\n\n".join(chunks)

    # Generate answer using QA model only if context is available
    initial_answer = generate_huggingface_answer(query, combined_context)

    # Refine the answer using OpenAI rephrase
    refined_answer = openai_client.rephrase(initial_answer, query, combined_context)

    return JsonResponse({"predictions": refined_answer})


