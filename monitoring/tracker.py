import time
from typing import Dict, Any
from datetime import datetime



_metrics = {
    "total_queries": 0,
    "total_tokens": 0,
    "total_cost_usd": 0.0,
    "latencies": [],
    "failures": 0,
    "queries_log": []
}

# Groq pricing (approximate)
COST_PER_1K_TOKENS = 0.0001


def track_query(
    query: str,
    answer: str,
    latency_seconds: float,
    token_count: int = 0,
    success: bool = True
) -> Dict[str, Any]:
    """Track a query and its metrics."""
    
    cost = (token_count / 1000) * COST_PER_1K_TOKENS
    
    _metrics["total_queries"] += 1
    _metrics["total_tokens"] += token_count
    _metrics["total_cost_usd"] += cost
    _metrics["latencies"].append(latency_seconds)
    
    if not success:
        _metrics["failures"] += 1
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query[:100],
        "answer_length": len(answer),
        "latency_seconds": round(latency_seconds, 3),
        "token_count": token_count,
        "cost_usd": round(cost, 6),
        "success": success
    }
    
    _metrics["queries_log"].append(log_entry)
    
    # Keep only last 100 entries
    if len(_metrics["queries_log"]) > 100:
        _metrics["queries_log"] = _metrics["queries_log"][-100:]
    
    return log_entry


def get_metrics() -> Dict[str, Any]:
    """Get current metrics summary."""
    latencies = _metrics["latencies"]
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    
    return {
        "total_queries": _metrics["total_queries"],
        "total_tokens": _metrics["total_tokens"],
        "total_cost_usd": round(_metrics["total_cost_usd"], 4),
        "avg_latency_seconds": round(avg_latency, 3),
        "p95_latency_seconds": round(p95_latency, 3),
        "failure_rate": round(
            _metrics["failures"] / max(_metrics["total_queries"], 1), 3
        ),
        "recent_queries": _metrics["queries_log"][-10:]
    }


class QueryTimer:
    """Context manager for timing queries."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time