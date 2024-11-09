from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
# Step 1: Set up Models with CPU

device = 'cpu'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Initialize ChromaDB client with persistent storage
persist_directory = "chapter_embeddings_persist"
client = chromadb.Client(settings=chromadb.config.Settings(persist_directory=persist_directory))
collection = client.get_or_create_collection(name="chapter_embeddings")

# Initialize Hugging Face question-answering pipeline for CPU only
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)

import openai
import time
import os

class OpenAIClient:
    def __init__(self, api_key):
        openai.api_key = api_key

    def summarize_prompt(self, long_prompt):
        # Summarize the prompt to make it fit within the 1000-character limit
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize the following prompt to be under 1000 characters:\n\n{long_prompt}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()

    def generate_image(self, prompt, retries=3, delay=5):
        # Shorten the prompt if it's over 1000 characters
        if len(prompt) > 1000:
            prompt = self.summarize_prompt(prompt)

        for attempt in range(retries):
            try:
                response = openai.Image.create(
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                image_url = response['data'][0]['url']
                return image_url
            except Exception as e:
                print(f"Error generating image on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print("Max retries reached. Please try again later or contact support.")
                    return None
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
            print(f"Error generating json code: {e}")
            return answer  # Return the original answer if there's an error
    def pipline(self, answer, question, recommendation):
        messages = [
          {"role": "system", "content": "You are a helpful assistant that prepares JSON code for a pipeline."},
          {"role": "user", "content": (
              f"Based on the question, answer, and recommendation, create JSON code that can be used to prepare a pipeline in my React app.\n"
              f"Each step in the pipeline should follow this format:\n"
              f"{{\n"
              f'  "pipeline": [\n'
              f'    {{ "step": "", "description": "", "relation": "" }},\n'
              f'    {{ "step": "", "description": "", "relation": "" }},\n'
              f'    ...\n'
              f'  ]\n'
              f"}}\n"
              f"The 'relation' field should indicate the flow direction to other steps (just write the step name).\n\n"
              f"Question: {question}\n"
              f"Answer: {answer}\n"
              f"Recommendation: {recommendation}\n"
              f"Provide the JSON code for the steps with flow direction relations.")
          }
      ]


        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Change to "gpt-4" if using GPT-4
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            refined_answer = response.choices[0].message['content'].strip()
            return refined_answer
        except Exception as e:
            print(f"Error generating json code: {e}")
            return answer  # Return the original answer if there's an error
client = OpenAIClient(api_key="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")  # Replace with your actual API key

import networkx as nx
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import csv
# Load data
file_path = r'C:\Users\Dell\Desktop\react\myproject\DataPreprocessed.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',', on_bad_lines="skip")

# Initialize Sentence Transformer model for embeddings
device = 'cpu'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Step 1: Build the NetworkX graph with SPO triples
# Step 1: Build the NetworkX graph with SPO triples
def build_graph_from_triples(df):
    graph = nx.Graph()
    for _, row in df.iterrows():
        triples_str = row['SPO_triples']

        # Define general node attributes based on the row data
        node_attributes = {
            "title": row["title"],
            "content": row["cleaned_content"],
            "summary": row["summary"],
            "lemmatized_summary": row["lemmatized_summary"],
            "text": row["cleaned_content"]  # Ensure 'text' is set to use later
        }

        # Check that SPO_triples is in the expected format
        if isinstance(triples_str, str):
            try:
                triples = eval(triples_str)  # Convert string to list of triples
            except Exception as e:
                print(f"Error evaluating triples for row {row['title']}: {e}")
                continue

            # Add nodes and edges based on the SPO triples
            for s, p, o in triples:
                # Add subject and object nodes with attributes if they don’t exist
                if s not in graph:
                    graph.add_node(s, **node_attributes)
                if o not in graph:
                    graph.add_node(o, **node_attributes)

                # Add edge between subject and object with the predicate as an attribute
                graph.add_edge(s, o, predicate=p)

    return graph

# Build the graph
graph = build_graph_from_triples(data)

# Step 2: Diagnostics - Check if graph is populated correctly
print(f"Total Nodes: {graph.number_of_nodes()}")
print(f"Total Edges: {graph.number_of_edges()}")

embeddings = []
target_embeddings = []
for node in graph.nodes():
    # Generate the embedding for 'content'
    content_text = graph.nodes[node].get('content', '')
    embedding = embedding_model.encode(content_text, convert_to_tensor=True).cpu().numpy()
    graph.nodes[node]['embedding'] = embedding
    embeddings.append(embedding)
    target_embeddings.append(embedding)  # Set as target

# Convert NetworkX graph to PyTorch Geometric Data object
data = from_networkx(graph)

# Populate data.x with node embeddings and set data.y as target embeddings
data.x = torch.tensor(embeddings, dtype=torch.float)
data.y = torch.tensor(target_embeddings, dtype=torch.float)

# Check dimensions to confirm they match
print("data.x shape:", data.x.shape)
print("data.y shape:", data.y.shape)

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score

# Optimized GNN Model with GATConv, Dropout, and Early Stopping
class OptimizedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=384, dropout_rate=0.3):
        super(OptimizedGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)  # Dropout after first layer
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)  # Dropout after second layer
        x = self.fc(x)
        return x

# Initialize Model, Optimizer, and Loss Function
input_dim = data.x.shape[1]
model = OptimizedGNN(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Optimized learning rate
criterion = nn.MSELoss()
print("GNN model initialized successfully.")

# Load GNN model and weights
input_dim = 384  # Set according to your model's input dimension
model = OptimizedGNN(input_dim=input_dim)
model.load_state_dict(torch.load('C:/Users/Dell/Desktop/projet/final/optimized_gnn.pth'))
model.eval()

from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Download for sentence tokenization

def generate_gnn_answer_with_refined_recommendations(query, data, model, graph, top_k=10, similarity_threshold=0.6):
    # Step 1: Generate Answer
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu()
    model.eval()

    with torch.no_grad():
        node_embeddings = model(data)

    # Calculate similarity scores between query and node embeddings
    similarities = torch.matmul(node_embeddings, query_embedding)
    top_k_nodes = similarities.topk(top_k, largest=True).indices

    # Retrieve nodes with top similarity scores and build context for the answer
    node_list = list(graph.nodes)
    context = []
    top_node_ids = []
    total_similarity = 0  # Track the total similarity for fallback handling
    for node in top_k_nodes:
        node_index = node.item()
        if node_index < len(node_list):
            node_id = node_list[node_index]
            similarity_score = similarities[node_index].item()
            total_similarity += similarity_score  # Accumulate similarity for checking later
            context.append(graph.nodes[node_id].get('content', ''))
            top_node_ids.append(node_id)

    # Calculate average similarity to detect off-topic queries
    avg_similarity = total_similarity / len(top_k_nodes)
    combined_context = "\n\n".join(context)

    # Generate answer using a QA model on the combined context
    answer = qa_model(question=query, context=combined_context) if combined_context else "Context not found."
    refined_answer = client.rephrase(answer["answer"], query, combined_context)

    # Step 2: Generate Refined Recommendations based on neighbors of top nodes
    recommendations = []
    seen_titles = set()  # Track unique titles to avoid duplicates
    query_embedding_np = query_embedding.numpy()  # Convert to numpy for similarity calculation

    for node_id in top_node_ids:
        # Check if node exists before fetching neighbors
        if node_id not in graph:
            continue

        # Fetch neighbors for recommendations
        neighbors = graph.neighbors(node_id)
        for neighbor in neighbors:
            if neighbor != node_id:  # Avoid self-loops
                neighbor_data = graph.nodes[neighbor]
                title = neighbor_data.get("title", neighbor)

                # Only add unique titles
                if title not in seen_titles:
                    content = neighbor_data.get("content", "")
                    content_embedding = embedding_model.encode(content, convert_to_tensor=True).cpu().numpy()
                    similarity_score = cosine_similarity([query_embedding_np], [content_embedding])[0][0]

                    # Include recommendations above the similarity threshold
                    if similarity_score > similarity_threshold:
                        # Extract key sentences from the content
                        sentences = sent_tokenize(content)
                        key_sentences = " ".join(sentences[:2]) if len(sentences) > 2 else content

                        recommendations.append({
                            "title": title,
                            "key_sentences": key_sentences,
                            "similarity_score": similarity_score
                        })
                        seen_titles.add(title)  # Mark as seen to avoid duplicates

    # Sort recommendations by similarity score in descending order
    recommendations = sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)

    # Display the answer with formatting
    print("\n" + "="*40)
    print("Answer:")
    print("="*40)
    print(refined_answer)
    print("="*40)

    # Display refined recommendations or a default message if none found
    if avg_similarity < similarity_threshold:
        # Fallback for ambiguous/off-topic queries
        print("\nIt seems that your query may not be related to the core topics covered by this system.")
        print("This system is optimized for queries on risk management and related topics.")
    elif recommendations:
        print("\nRelated Recommendations:")
        top_recommendation = recommendations[0]  # Get the first recommendation
        print("\n" + "-"*30)
        print(f"Recommendation:")
        print("Title:", top_recommendation["title"])
        print("Key Sentences:", top_recommendation["key_sentences"])
        print("Similarity Score:", top_recommendation["similarity_score"])
        print("-"*30)
    else:
        print("\nNo related recommendations found.")
    print("\n" + "="*40 + "\n")
    output = client.pipline(refined_answer, query, top_recommendation["title"] + top_recommendation["key_sentences"])
    print("code json :",output)
    import json
    import re

    # Assume `client.pipline` returns a JSON-like string

    # Extract JSON content between the first '{' and the last '}'
    json_match = re.search(r"\{.*\}", output, re.DOTALL)

    if json_match:
        json_content = json_match.group(0)  # Get the JSON part from the match

        try:
            json_data = json.loads(json_content)  # Convert string to JSON (dict)
        except json.JSONDecodeError:
            print("Error: Extracted content is not in JSON format.")
            json_data = {}
    else:
        print("Error: No JSON object found in the output.")
        json_data = {}

    print("JSON file 'pipeline_output.json' created successfully.")
    return refined_answer,top_recommendation["title"],top_recommendation["key_sentences"],json_data

@api_view(['POST'])
def predict(request):
    try:
        query = request.data.get('input')
        print("Received query:", query)  # Debugging line

        # Assuming generate_gnn_answer_with_refined_recommendations is defined elsewhere
        answer, rec_title, rec, json = generate_gnn_answer_with_refined_recommendations(query, data, model, graph)
        print("Generated answer and recommendations")  # Debugging line

        return JsonResponse({"answer": answer, "rec_title": rec_title, "rec": rec, "json": json})
    except Exception as e:
        print("Error in predict view:", e)  # Print error details
        return JsonResponse({"error": "An error occurred", "details": str(e)}, status=500)



