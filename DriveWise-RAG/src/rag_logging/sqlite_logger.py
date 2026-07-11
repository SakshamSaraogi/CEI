import sqlite3
import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "logs" / "rag_logs.db"

class SQLiteLogger:
    def __init__(self):
        # Ensure directory exists
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    rewritten_query TEXT,
                    brand TEXT,
                    model TEXT,
                    is_comparison INTEGER,
                    confidence TEXT,
                    latency_ms REAL,
                    tokens_used INTEGER,
                    response_json TEXT,
                    error_msg TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_id INTEGER NOT NULL,
                    rating INTEGER, -- 1 for thumbs up, -1 for thumbs down
                    feedback TEXT,
                    FOREIGN KEY(log_id) REFERENCES query_logs(id)
                )
            """)

    def log_query(self, query: str, rewritten_query: str = None, brand: str = None, model: str = None,
                  is_comparison: bool = False, confidence: str = "not_found", latency_ms: float = 0.0,
                  tokens_used: int = 0, response_json: str = None, error_msg: str = None) -> int:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO query_logs (
                    timestamp, query, rewritten_query, brand, model, is_comparison,
                    confidence, latency_ms, tokens_used, response_json, error_msg
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                query,
                rewritten_query,
                brand,
                model,
                1 if is_comparison else 0,
                confidence,
                latency_ms,
                tokens_used,
                response_json,
                error_msg
            ))
            return cursor.lastrowid

    def log_feedback(self, log_id: int, rating: int, feedback: str = None):
        with self.conn:
            self.conn.execute("""
                INSERT INTO ratings (log_id, rating, feedback)
                VALUES (?, ?, ?)
            """, (log_id, rating, feedback))

    def get_logs(self, limit: int = 50):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_analytics(self):
        cursor = self.conn.cursor()
        
        # 1. Total count
        cursor.execute("SELECT COUNT(*) FROM query_logs")
        total_queries = cursor.fetchone()[0]
        
        if total_queries == 0:
            return {
                "total_queries": 0,
                "avg_latency": 0.0,
                "confidence_breakdown": {},
                "top_models": [],
                "daily_volume": []
            }
            
        # 2. Avg latency
        cursor.execute("SELECT AVG(latency_ms) FROM query_logs")
        avg_latency = cursor.fetchone()[0] or 0.0
        
        # 3. Confidence breakdown
        cursor.execute("SELECT confidence, COUNT(*) as count FROM query_logs GROUP BY confidence")
        confidence_breakdown = {row["confidence"]: row["count"] for row in cursor.fetchall()}
        
        # 4. Top models
        cursor.execute("""
            SELECT model, COUNT(*) as count 
            FROM query_logs 
            WHERE model IS NOT NULL AND model != 'global' AND model != ''
            GROUP BY model 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_models = [dict(row) for row in cursor.fetchall()]
        
        # 5. Daily volume (past 30 days)
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count, AVG(latency_ms) as avg_lat
            FROM query_logs
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        """, (thirty_days_ago,))
        daily_volume = [dict(row) for row in cursor.fetchall()]
        
        return {
            "total_queries": total_queries,
            "avg_latency": round(avg_latency, 2),
            "confidence_breakdown": confidence_breakdown,
            "top_models": top_models,
            "daily_volume": daily_volume
        }
