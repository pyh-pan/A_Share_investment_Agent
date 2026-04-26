"""
Financial Decision Memory using ChromaDB.

Stores past trading decisions with their market context and outcomes,
enabling the system to learn from past mistakes and recall similar situations.

Architecture:
- Uses ChromaDB PersistentClient (survives restarts)
- Embeds structured feature summaries (not raw reports)
- Single collection with metadata filtering (not per-agent collections)
- SHA-256 hash IDs for deduplication
- Cosine similarity threshold of 0.7
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logging_config import setup_logger

logger = setup_logger("memory")

# ChromaDB is optional — system should work without it
_chromadb_available = False
_bge_available = False

try:
    import chromadb
    _chromadb_available = True
except ImportError:
    logger.debug("chromadb not installed, memory system disabled")

try:
    from chromadb.utils import embedding_functions
    _bge_available = True
except ImportError:
    logger.debug("chromadb embedding_functions not available")


class AShareDecisionMemory:
    """Financial decision memory with ChromaDB vector store.

    Key design decisions (improving on TradingAgents-CN):
    1. PersistentClient (not in-memory) — survives restarts
    2. Structured feature summary for embedding (not raw report text)
    3. Single collection + metadata filtering (not 5 collections)
    4. SHA-256 hash IDs (not auto-increment) — prevents duplicates
    5. Cosine similarity threshold 0.7 (not unfiltered top-K)
    6. bge-large-zh-v1.5 embedding model (local, free, excellent Chinese)
    """

    DEFAULT_PERSIST_DIR = "src/data/memory"
    COLLECTION_NAME = "ashare_decisions"
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, persist_dir: Optional[str] = None):
        if not _chromadb_available:
            self._client = None
            self._collection = None
            logger.warning("ChromaDB not available, memory system disabled")
            return

        persist_path = persist_dir or self.DEFAULT_PERSIST_DIR
        os.makedirs(persist_path, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(path=persist_path)
            embedding_fn = self._get_embedding_function()

            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB memory initialized at {persist_path}, "
                       f"existing memories: {self._collection.count()}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._client = None
            self._collection = None

    def _get_embedding_function(self):
        """Get the best available embedding function for Chinese financial text."""
        if _bge_available:
            try:
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="BAAI/bge-large-zh-v1.5"
                )
            except Exception:
                logger.warning("bge-large-zh-v1.5 not available, using default")

        # Fallback to ChromaDB default (all-MiniLM-L6-v2)
        return None

    def _create_situation_summary(self, state_data: dict) -> str:
        """Create structured situation text for embedding.

        This is CRITICAL: we embed a structured feature summary, NOT raw reports.
        This keeps embeddings short, focused, and produces better similarity matches.
        """
        ticker = state_data.get("ticker", "unknown")
        industry = state_data.get("industry_classification", "unknown")

        # Extract signals from agent reports (if available)
        tech = state_data.get("technical_report", {})
        fund = state_data.get("fundamentals_report", {})
        sent = state_data.get("sentiment_report", {})
        val = state_data.get("valuation_report", {})
        macro = state_data.get("macro_report", {})

        tech_signal = tech.get("signal", "N/A") if isinstance(tech, dict) else "N/A"
        fund_signal = fund.get("signal", "N/A") if isinstance(fund, dict) else "N/A"
        sent_signal = sent.get("signal", "N/A") if isinstance(sent, dict) else "N/A"
        val_signal = val.get("signal", "N/A") if isinstance(val, dict) else "N/A"
        macro_env = macro.get("macro_environment", "N/A") if isinstance(macro, dict) else "N/A"

        northbound = state_data.get("northbound_flow", {})
        nb_signal = northbound.get("signal", "N/A") if isinstance(northbound, dict) else "N/A"

        metrics = state_data.get("financial_metrics", [{}])
        latest_metrics = metrics[0] if metrics else {}

        summary = (
            f"股票:{ticker} 行业:{industry} "
            f"技术:{tech_signal} 基本面:{fund_signal} "
            f"情绪:{sent_signal} 估值:{val_signal} "
            f"宏观:{macro_env} 北向:{nb_signal} "
            f"ROE:{latest_metrics.get('return_on_equity', 'N/A')} "
            f"PE:{latest_metrics.get('pe_ratio', 'N/A')} "
            f"负债率:{latest_metrics.get('debt_to_equity', 'N/A')}"
        )
        return summary.strip()

    def _generate_id(self, ticker: str, date: str, situation: str) -> str:
        """Deterministic ID based on content hash to prevent duplicates."""
        content = f"{ticker}:{date}:{situation[:200]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def is_available(self) -> bool:
        """Check if memory system is operational."""
        return self._collection is not None

    def store_decision(
        self,
        state_data: dict,
        decision: str,
        confidence: float,
        reasoning: str,
        outcome: Optional[dict] = None,
    ) -> bool:
        """Store a trading decision with its context.

        Args:
            state_data: The agent state data dict at decision time
            decision: "buy", "sell", or "hold"
            confidence: Decision confidence (0-1)
            reasoning: LLM reasoning text
            outcome: Optional outcome data {"return_pct": float, "days_held": int}

        Returns:
            True if stored successfully
        """
        if not self.is_available:
            return False

        try:
            ticker = state_data.get("ticker", "unknown")
            date = datetime.now().strftime("%Y-%m-%d")
            situation = self._create_situation_summary(state_data)
            doc_id = self._generate_id(ticker, date, situation)

            metadata = {
                "ticker": ticker,
                "date": date,
                "industry": state_data.get("industry_classification", "unknown"),
                "decision": decision,
                "confidence": confidence,
                "reasoning_summary": reasoning[:500] if reasoning else "",
                "actual_return": outcome.get("return_pct", 0) if outcome else 0,
                "correct": self._was_correct(decision, outcome) if outcome else None,
                "has_outcome": outcome is not None,
            }

            self._collection.upsert(
                ids=[doc_id],
                documents=[situation],
                metadatas=[metadata],
            )

            logger.debug(f"Stored decision memory: {ticker} {decision} on {date}")
            return True
        except Exception as e:
            logger.error(f"Failed to store decision memory: {e}")
            return False

    def store_reflection(
        self,
        ticker: str,
        date: str,
        situation_summary: str,
        reflection: str,
        was_correct: bool,
        actual_return: float = 0,
    ) -> bool:
        """Store a post-hoc reflection on a past decision.

        This is called after the outcome is known, to update the memory
        with whether the decision was correct.
        """
        if not self.is_available:
            return False

        try:
            doc_id = self._generate_id(ticker, date, situation_summary)

            metadata = {
                "ticker": ticker,
                "date": date,
                "industry": "",
                "decision": "reflection",
                "confidence": 0,
                "reasoning_summary": reflection[:500],
                "actual_return": actual_return,
                "correct": was_correct,
                "has_outcome": True,
            }

            self._collection.upsert(
                ids=[doc_id],
                documents=[situation_summary],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store reflection: {e}")
            return False

    def recall_similar(
        self,
        state_data: dict,
        n: int = 3,
        ticker_filter: Optional[str] = None,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Recall similar past decisions.

        Args:
            state_data: Current agent state data
            n: Number of similar situations to retrieve
            ticker_filter: If set, only return decisions for this stock
            min_similarity: Minimum cosine similarity (0-1)

        Returns:
            List of similar past decisions with their outcomes
        """
        if not self.is_available:
            return []

        try:
            situation = self._create_situation_summary(state_data)

            where_clause = None
            if ticker_filter:
                where_clause = {"ticker": ticker_filter}

            results = self._collection.query(
                query_texts=[situation],
                n_results=n,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            # Cosine distance = 1 - cosine_similarity
            max_distance = 1.0 - min_similarity
            memories = []

            if not results["documents"] or not results["documents"][0]:
                return []

            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                if distance <= max_distance:
                    meta = results["metadatas"][0][i]
                    memories.append({
                        "situation": doc,
                        "decision": meta.get("decision", "unknown"),
                        "confidence": meta.get("confidence", 0),
                        "was_correct": meta.get("correct"),
                        "actual_return": meta.get("actual_return", 0),
                        "reasoning": meta.get("reasoning_summary", ""),
                        "date": meta.get("date", ""),
                        "similarity": round(1.0 - distance, 3),
                    })

            logger.debug(f"Recalled {len(memories)} similar past decisions "
                        f"(from {len(results['documents'][0])} candidates)")
            return memories
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return []

    def get_memory_prompt(self, state_data: dict, n: int = 3) -> str:
        """Get a formatted prompt section with past decision memories.

        This is designed to be injected into agent prompts.
        """
        memories = self.recall_similar(state_data, n=n)

        if not memories:
            return ""

        prompt = "📚 历史决策记忆（类似情况下的过去决策）：\n"
        for i, mem in enumerate(memories, 1):
            correct_str = "✅正确" if mem["was_correct"] else "❌错误" if mem["was_correct"] is False else "❓未知"
            prompt += (
                f"\n{i}. [{mem['date']}] {mem['situation'][:100]}\n"
                f"   决策: {mem['decision']} (置信度:{mem['confidence']}) "
                f"结果: {correct_str} 实际收益:{mem['actual_return']:+.1%}\n"
            )
            if mem["reasoning"]:
                prompt += f"   教训: {mem['reasoning'][:200]}\n"

        prompt += "\n⚠️ 请从上述历史经验中学习，避免重复犯同样的错误。\n"
        return prompt

    @staticmethod
    def _was_correct(decision: str, outcome: Optional[dict]) -> bool:
        """Determine if a decision was correct based on actual outcome."""
        if not outcome:
            return False
        ret = outcome.get("return_pct", 0)
        if decision == "buy":
            return ret > 0.02  # At least 2% gain to count as correct
        elif decision == "sell":
            return ret < -0.02  # Stock dropped after sell
        else:
            return -0.05 < ret < 0.05  # Stock stayed relatively flat

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self.is_available:
            return {"available": False}

        try:
            count = self._collection.count()
            return {
                "available": True,
                "total_memories": count,
                "persist_dir": self.DEFAULT_PERSIST_DIR,
            }
        except Exception:
            return {"available": False}

    def clear(self) -> bool:
        """Clear all memories (for testing)."""
        if not self.is_available:
            return False
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._get_embedding_function(),
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False


def _normalize_decision(decision: str) -> str:
    normalized = str(decision or "").strip().lower()
    aliases = {
        "买入": "buy",
        "buy": "buy",
        "卖出": "sell",
        "sell": "sell",
        "持有": "hold",
        "hold": "hold",
    }
    return aliases.get(normalized, normalized or "unknown")


def _was_outcome_correct(decision: str, actual_return: float) -> bool:
    if decision == "buy":
        return actual_return > 0.02
    if decision == "sell":
        return actual_return < -0.02
    if decision == "hold":
        return -0.05 < actual_return < 0.05
    return False


def build_outcome_reflection(
    decision: str,
    actual_return: float,
    days_held: int,
) -> Dict[str, Any]:
    """Build a deterministic post-hoc lesson for a completed decision."""
    normalized_decision = _normalize_decision(decision)
    return_pct = float(actual_return or 0)
    holding_days = int(days_held or 0)
    was_correct = _was_outcome_correct(normalized_decision, return_pct)

    if normalized_decision == "buy":
        if was_correct:
            lesson = "buy thesis was confirmed; similar signals can support future entries."
        elif return_pct < -0.02:
            lesson = "buy was wrong; tighten entry validation and downside risk controls."
        else:
            lesson = "buy was inconclusive; require stronger upside before committing capital."
    elif normalized_decision == "sell":
        if was_correct:
            lesson = "sell avoided a subsequent decline; respect similar exit warnings."
        elif return_pct > 0.02:
            lesson = "sell missed upside; review whether exit signals were too conservative."
        else:
            lesson = "sell was inconclusive; transaction timing added little value."
    elif normalized_decision == "hold":
        if was_correct:
            lesson = "hold was appropriate; market stayed range-bound and patience avoided churn."
        elif return_pct > 0.05:
            lesson = "hold missed upside; look for catalysts that warranted buying."
        else:
            lesson = "hold missed risk control; defensive signals should trigger reduction."
    else:
        lesson = "unknown decision type; verify decision normalization before reusing this memory."

    reflection = (
        f"{normalized_decision} outcome over {holding_days} days: "
        f"actual return {return_pct:+.1%}. Lesson: {lesson}"
    )
    return {
        "decision": normalized_decision,
        "actual_return": return_pct,
        "days_held": holding_days,
        "was_correct": was_correct,
        "reflection": reflection,
    }


def update_decision_outcome(
    ticker: str,
    date: str,
    situation_summary: str,
    decision: str,
    actual_return: float,
    days_held: int,
) -> bool:
    """Best-effort memory update after an outcome is known."""
    try:
        outcome = build_outcome_reflection(decision, actual_return, days_held)
        memory = get_memory()
        return bool(
            memory.store_reflection(
                ticker=ticker,
                date=date,
                situation_summary=situation_summary,
                reflection=outcome["reflection"],
                was_correct=outcome["was_correct"],
                actual_return=outcome["actual_return"],
            )
        )
    except Exception as e:
        logger.warning(f"Decision outcome memory update skipped: {e}")
        return False


_memory_instance: Optional[AShareDecisionMemory] = None

def get_memory() -> AShareDecisionMemory:
    """Get the singleton memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = AShareDecisionMemory()
    return _memory_instance
