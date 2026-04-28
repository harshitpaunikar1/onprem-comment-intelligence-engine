"""
On-premises comment intelligence engine.
Analyzes ingested comments for sentiment, topic clustering, trend detection, and insight generation.
"""
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rag_pipeline import Comment, CommentRAGPipeline

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


@dataclass
class TopicCluster:
    cluster_id: int
    top_terms: List[str]
    sample_texts: List[str]
    size: int
    dominant_sentiment: str


@dataclass
class InsightReport:
    total_comments: int
    sentiment_distribution: Dict[str, int]
    top_topics: List[TopicCluster]
    trend_signals: List[str]
    recommended_actions: List[str]


class SentimentAnalyzer:
    """
    Rule-based sentiment analyzer for customer comments.
    Uses lexicon matching for offline operation without external APIs.
    """

    POSITIVE_WORDS = {"great", "excellent", "amazing", "love", "perfect", "happy",
                      "fast", "recommend", "satisfied", "quality", "helpful"}
    NEGATIVE_WORDS = {"damaged", "broken", "wrong", "delay", "late", "disappointed",
                      "refund", "terrible", "awful", "poor", "unhelpful", "missing"}
    NEUTRAL_WORDS = {"okay", "average", "acceptable", "fine", "neutral"}

    def analyze(self, text: str) -> Tuple[str, float]:
        """Return (sentiment_label, confidence) for a text."""
        words = set(re.findall(r"\b\w+\b", text.lower()))
        pos = len(words & self.POSITIVE_WORDS)
        neg = len(words & self.NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return "neutral", 0.5
        score = (pos - neg) / total
        if score > 0.2:
            return "positive", min(1.0, 0.5 + score)
        elif score < -0.2:
            return "negative", min(1.0, 0.5 - score)
        return "neutral", 0.5

    def batch_analyze(self, comments: List[Comment]) -> List[Comment]:
        for c in comments:
            if c.sentiment is None:
                sentiment, _ = self.analyze(c.text)
                c.sentiment = sentiment
        return comments


class TopicExtractor:
    """Extracts topics from comments using TF-IDF + K-Means clustering."""

    def __init__(self, n_topics: int = 5, max_features: int = 200):
        self.n_topics = n_topics
        self.max_features = max_features

    def extract(self, comments: List[Comment]) -> List[TopicCluster]:
        if not SKLEARN_AVAILABLE or len(comments) < self.n_topics:
            return []
        texts = [c.text for c in comments]
        vectorizer = TfidfVectorizer(max_features=self.max_features,
                                      stop_words="english", min_df=1)
        try:
            X = vectorizer.fit_transform(texts)
        except Exception:
            return []
        n_clusters = min(self.n_topics, len(texts))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = km.fit_predict(X)
        terms = vectorizer.get_feature_names_out()
        clusters = []
        for cid in range(n_clusters):
            mask = labels == cid
            cluster_comments = [c for c, m in zip(comments, mask) if m]
            center = km.cluster_centers_[cid]
            top_indices = center.argsort()[-8:][::-1]
            top_terms = [terms[i] for i in top_indices]
            sentiments = [c.sentiment or "neutral" for c in cluster_comments]
            dominant = Counter(sentiments).most_common(1)[0][0]
            samples = [c.text[:100] for c in cluster_comments[:3]]
            clusters.append(TopicCluster(
                cluster_id=cid,
                top_terms=top_terms,
                sample_texts=samples,
                size=int(mask.sum()),
                dominant_sentiment=dominant,
            ))
        return sorted(clusters, key=lambda x: x.size, reverse=True)


class TrendDetector:
    """Detects emerging trends from time-series comment data."""

    def detect(self, comments: List[Comment], window_days: int = 7) -> List[str]:
        """Return trend signal strings for the most recent time window."""
        import time
        now = time.time()
        cutoff = now - window_days * 86400
        recent = [c for c in comments if c.created_at >= cutoff]
        if not recent:
            return []
        signals = []
        neg_recent = sum(1 for c in recent if c.sentiment == "negative")
        neg_all = sum(1 for c in comments if c.sentiment == "negative")
        all_count = max(len(comments), 1)
        recent_count = max(len(recent), 1)
        neg_rate_recent = neg_recent / recent_count
        neg_rate_all = neg_all / all_count
        if neg_rate_recent > neg_rate_all * 1.3:
            signals.append(f"Negative sentiment spike: {neg_rate_recent:.0%} in last {window_days}d vs {neg_rate_all:.0%} overall")
        if len(recent) / all_count > 0.4:
            signals.append(f"High comment volume surge: {len(recent)} comments in last {window_days} days")
        return signals


class CommentIntelligenceEngine:
    """
    Orchestrates sentiment analysis, topic extraction, trend detection,
    and LLM-powered insight generation for on-premises comment intelligence.
    """

    def __init__(self, rag_pipeline: CommentRAGPipeline,
                 llm_model: str = "llama3"):
        self.pipeline = rag_pipeline
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_extractor = TopicExtractor(n_topics=5)
        self.trend_detector = TrendDetector()
        self.llm_model = llm_model
        self._comments: List[Comment] = []

    def ingest(self, comments: List[Comment]) -> int:
        labeled = self.sentiment_analyzer.batch_analyze(comments)
        self._comments.extend(labeled)
        return self.pipeline.ingest(labeled)

    def generate_insight_report(self) -> InsightReport:
        if not self._comments:
            return InsightReport(0, {}, [], [], [])
        sentiments = Counter(c.sentiment or "neutral" for c in self._comments)
        topics = self.topic_extractor.extract(self._comments)
        trends = self.trend_detector.detect(self._comments)
        actions = self._recommend_actions(sentiments, topics)
        return InsightReport(
            total_comments=len(self._comments),
            sentiment_distribution=dict(sentiments),
            top_topics=topics,
            trend_signals=trends,
            recommended_actions=actions,
        )

    def _recommend_actions(self, sentiments: Counter,
                             topics: List[TopicCluster]) -> List[str]:
        actions = []
        total = max(sum(sentiments.values()), 1)
        neg_pct = sentiments.get("negative", 0) / total
        if neg_pct > 0.3:
            actions.append(f"Urgent: {neg_pct:.0%} negative sentiment - escalate to product/support teams.")
        neg_topics = [t for t in topics if t.dominant_sentiment == "negative"]
        for t in neg_topics[:2]:
            actions.append(f"Address recurring issues around: {', '.join(t.top_terms[:4])}")
        if not neg_topics:
            actions.append("Sentiment is healthy - focus on amplifying positive reviews.")
        actions.append("Set up weekly sentiment monitoring dashboard for early signal detection.")
        return actions

    def ask(self, question: str, top_k: int = 5) -> str:
        """Answer a question about comments using RAG + LLM."""
        results = self.pipeline.retrieve(question, top_k=top_k)
        context = "\n\n".join([f"Comment [{r.source}]: {r.text}" for r in results])
        prompt = f"Based on customer comments, answer: {question}\n\nContext:\n{context}\n\nAnswer:"
        if OLLAMA_AVAILABLE:
            try:
                resp = ollama.generate(model=self.llm_model, prompt=prompt)
                return resp.get("response", "")
            except Exception:
                pass
        if results:
            return f"Based on {len(results)} relevant comments: {results[0].text[:200]}..."
        return "No relevant comments found to answer the question."


if __name__ == "__main__":
    import time
    from rag_pipeline import CommentRAGPipeline

    pipeline = CommentRAGPipeline(collection_name="engine_demo")
    engine = CommentIntelligenceEngine(pipeline)

    now = time.time()
    comments = [
        Comment("e001", "Product quality has declined significantly this month.", "review", sentiment=None, created_at=now - 86400),
        Comment("e002", "Shipping was damaged again, third time this happens.", "support_ticket", sentiment=None, created_at=now - 3600),
        Comment("e003", "Love the new packaging, much better experience.", "survey", sentiment=None, created_at=now - 7200),
        Comment("e004", "Customer support resolved my issue very quickly.", "review", sentiment=None, created_at=now - 50000),
        Comment("e005", "Wrong item delivered, refund process is slow.", "support_ticket", sentiment=None, created_at=now - 1800),
        Comment("e006", "Great product, will recommend to friends.", "review", sentiment=None, created_at=now - 200000),
        Comment("e007", "Disappointed with quality, expected better for the price.", "review", sentiment=None, created_at=now - 3000),
    ]

    indexed = engine.ingest(comments)
    print(f"Indexed: {indexed} comments")

    report = engine.generate_insight_report()
    print(f"\nTotal comments: {report.total_comments}")
    print(f"Sentiment distribution: {report.sentiment_distribution}")
    print(f"\nTrend signals: {report.trend_signals}")
    print(f"\nRecommended actions:")
    for action in report.recommended_actions:
        print(f"  - {action}")

    answer = engine.ask("What are the main product quality issues?")
    print(f"\nQ: What are the main product quality issues?\nA: {answer}")
