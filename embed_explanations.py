from openai import OpenAI
from typing import List
import json
from pydantic import BaseModel

client = OpenAI()

class G2GEmbeddingTerm(BaseModel):
    term: str
    snippet: str
    definition: str
    explanation: str
    snippet_embedding: List[float]

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model)\
       .data[0].embedding

# get_embedding("We are lucky to live in an age in which we are still making discoveries.")

# open g2g-response.json and read the file

with open("g2g-response.json", "r", encoding="utf-8") as file:
    response = json.load(file)
    embedding_terms = []
    for i,term in enumerate(response):
        print(f"Embedding term {i + 1}/{len(response)}")
        snippet_embedding = get_embedding(term["snippet"])
        term["snippet_embedding"] = snippet_embedding
        embedding_term = G2GEmbeddingTerm(**term)
        embedding_terms.append(embedding_term)

    with open("g2g-response-embeddings.json", "w", encoding="utf-8") as file:
        track_dicts = [dict(track) for track in embedding_terms]
        file.write(json.dumps(track_dicts, indent=4))