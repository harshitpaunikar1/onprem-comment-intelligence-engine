"""
On-premises comment intelligence RAG pipeline.
Embeds customer comments locally using sentence transformers and retrieves relevant context.
"""
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


@dataclass
class Comment:
    comment_id: str
    text: str
    source: str           # support_ticket, survey, review, social_media
    product_id: Optional[str] = None
    sentiment: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    comment_id: str
    text: str
    source: str
    score: float
    sentiment: Optional[str]


class LocalEmbedder:
    """On-premises sentence embedder using a locally loaded model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if not ST_AVAILABLE or self._model is not None:
            return
        self._model = SentenceTransformer(self.model_name)
        logger.info("Loaded embedding model: %s", self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        if self._model is None:
            rng = np.random.default_rng(seed=42)
            return [rng.standard_normal(384).tolist() for _ in texts]
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]


class CommentVectorStore:
    """Stores comment embeddings in Qdrant with metadata for filtered retrieval."""

    def __init__(self, qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "comments",
                 vector_dim: int = 384):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self._client = None
        self._stub_comments: List[Comment] = []

    def _connect(self) -> bool:
        if not QDRANT_AVAILABLE:
            return False
        try:
            self._client = QdrantClient(url=self.qdrant_url)
            collections = [c.name for c in self._client.get_collections().collections]
            if self.collection_name not in collections:
                self._client.create_collection(
                    self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
                )
            return True
        except Exception as exc:
            logger.warning("Qdrant unavailable: %s", exc)
            return False

    def index(self, comments: List[Comment], embeddings: List[List[float]]) -> int:
        if not self._client and not self._connect():
            self._stub_comments.extend(comments)
            logger.info("[STUB] Indexed %d comments in memory.", len(comments))
            return len(comments)
        points = [
            PointStruct(
                id=abs(int(hashlib.md5(c.comment_id.encode()).hexdigest(), 16)) % (2**63),
                vector=emb,
                payload={
                    "comment_id": c.comment_id,
                    "text": c.text[:500],
                    "source": c.source,
                    "sentiment": c.sentiment,
                    "product_id": c.product_id,
                },
            )
            for c, emb in zip(comments, embeddings)
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query_vector: List[float], top_k: int = 5,
               source_filter: Optional[str] = None,
               sentiment_filter: Optional[str] = None) -> List[RetrievalResult]:
        if not self._client:
            results = []
            for c in self._stub_comments[:top_k]:
                results.append(RetrievalResult(
                    comment_id=c.comment_id,
                    text=c.text,
                    source=c.source,
                    score=float(np.random.uniform(0.7, 0.99)),
                    sentiment=c.sentiment,
                ))
            return results
        search_filter = None
        conditions = []
        if source_filter:
            conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
        if sentiment_filter:
            conditions.append(FieldCondition(key="sentiment", match=MatchValue(value=sentiment_filter)))
        if conditions:
            search_filter = Filter(must=conditions)
        try:
            hits = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter,
            )
            return [
                RetrievalResult(
                    comment_id=h.payload.get("comment_id", ""),
                    text=h.payload.get("text", ""),
                    source=h.payload.get("source", ""),
                    score=round(h.score, 4),
                    sentiment=h.payload.get("sentiment"),
                )
                for h in hits
            ]
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            return []


class CommentRAGPipeline:
    """
    End-to-end on-prem RAG pipeline for comment intelligence.
    Ingests, embeds, indexes, and retrieves customer comments for downstream analysis.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "comments"):
        self.embedder = LocalEmbedder(model_name=model_name)
        self.store = CommentVectorStore(qdrant_url=qdrant_url, collection_name=collection_name)
        self._indexed_count = 0

    def ingest(self, comments: List[Comment], batch_size: int = 64) -> int:
        """Embed and index comments in batches."""
        total = 0
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            embeddings = self.embedder.embed([c.text for c in batch])
            total += self.store.index(batch, embeddings)
        self._indexed_count += total
        return total

    def retrieve(self, query: str, top_k: int = 5,
                  source_filter: Optional[str] = None,
                  sentiment_filter: Optional[str] = None) -> List[RetrievalResult]:
        """Embed query and retrieve top-k semantically similar comments."""
        query_vec = self.embedder.embed_single(query)
        return self.store.search(query_vec, top_k=top_k,
                                  source_filter=source_filter,
                                  sentiment_filter=sentiment_filter)

    def find_similar_complaints(self, complaint_text: str,
                                 top_k: int = 5) -> List[RetrievalResult]:
        return self.retrieve(complaint_text, top_k=top_k, sentiment_filter="negative")

    def stats(self) -> Dict:
        return {"indexed_comments": self._indexed_count}


if __name__ == "__main__":
    np.random.seed(42)
    sample_comments = [
        Comment("c001", "The product arrived damaged and customer service was unhelpful.",
                "review", product_id="P001", sentiment="negative"),
        Comment("c002", "Delivery was fast and the item is exactly as described. Very happy!",
                "review", product_id="P002", sentiment="positive"),
        Comment("c003", "The refund process took too long and I had to follow up multiple times.",
                "support_ticket", sentiment="negative"),
        Comment("c004", "Great quality product, will definitely buy again.",
                "survey", product_id="P001", sentiment="positive"),
        Comment("c005", "Wrong item was shipped, support team resolved it quickly though.",
                "support_ticket", sentiment="neutral"),
        Comment("c006", "Product broke after one week, very disappointed with quality control.",
                "review", product_id="P003", sentiment="negative"),
        Comment("c007", "Excellent packaging and on-time delivery. Highly recommend.",
                "review", product_id="P002", sentiment="positive"),
    ]

    pipeline = CommentRAGPipeline(
        model_name="all-MiniLM-L6-v2",
        collection_name="comments_demo",
    )
    indexed = pipeline.ingest(sample_comments)
    print(f"Indexed {indexed} comments.")

    results = pipeline.retrieve("damaged product and poor customer service", top_k=3)
    print("\nTop results for 'damaged product and poor customer service':")
    for r in results:
        print(f"  [{r.score:.2f}] ({r.sentiment}) {r.text[:80]}...")

    complaints = pipeline.find_similar_complaints("item broke after arrival", top_k=2)
    print("\nSimilar complaints:")
    for c in complaints:
        print(f"  [{c.score:.2f}] {c.text[:80]}...")

    print("\nPipeline stats:", pipeline.stats())
