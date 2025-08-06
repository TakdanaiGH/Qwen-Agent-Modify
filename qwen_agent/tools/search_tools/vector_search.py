import json
import os
from typing import List, Tuple

from qwen_agent.tools.base import register_tool
from qwen_agent.tools.doc_parser import Record
from qwen_agent.tools.search_tools.base_search import BaseSearch


@register_tool('vector_search')
class VectorSearch(BaseSearch):
    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, str, float]]:
        try:
            from langchain.schema import Document
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please install langchain: `pip install langchain`')

        try:
            from sentence_transformers import SentenceTransformer, util
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please install sentence-transformers: `pip install sentence-transformers`')

        # Load embedding model
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        # Extract query string if it's a JSON object
        try:
            query_json = json.loads(query)
            if 'text' in query_json:
                query = query_json['text']
        except json.decoder.JSONDecodeError:
            pass

        # Prepare document chunks (one per JSON record)
        all_documents = []
        texts = []
        metadatas = []

        for doc in docs:
            for record in doc.raw:
                content = record.content
                metadata = {
                    "source": record.metadata.get("source", record.metadata.get("url", "")),
                    "chunk_id": record.metadata.get("chunk_id", record.metadata.get("id", "")),
                    "url": record.metadata.get("url", ""),
                    "id": record.metadata.get("id", "")
                }
                texts.append(content)
                metadatas.append(metadata)
                all_documents.append(Document(page_content=content, metadata=metadata))

        # Embed documents
        doc_embeddings = model.encode(texts, convert_to_tensor=True)
        # Embed query with prompt
        query_embedding = model.encode(query, prompt_name="query", convert_to_tensor=True)

        # Compute cosine similarities
        from torch import topk

        scores = util.cos_sim(query_embedding, doc_embeddings)[0]  # shape: [num_docs]
        top_k = min(3, len(scores))  # get top 3 matches or less
        top_results = topk(scores, k=top_k)

        top_matches = []
        for idx, score in zip(top_results.indices, top_results.values):
            metadata = metadatas[idx]
            top_matches.append((metadata["source"], metadata["chunk_id"], float(score)))

        return top_matches
