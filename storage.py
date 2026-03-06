import sqlite3
import json
import threading
from config import DB_PATH


class StorageManager:
    """SQLite backed storage for analysis results."""

    def __init__(self):
        self.db_path = DB_PATH
        self.lock = threading.Lock()
        self._init_db()
        self._migrate_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER,
                timestamp REAL,
                frame_path TEXT,
                clean_frame_path TEXT,
                analysis_text TEXT,
                classification TEXT,
                person_ids TEXT,
                num_persons INTEGER,
                detections_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _migrate_db(self):
        """Add columns if they don't exist (safe for existing databases)."""
        conn = self._get_conn()
        cursor = conn.execute("PRAGMA table_info(analyses)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        if "detections_json" not in existing_cols:
            conn.execute(
                "ALTER TABLE analyses ADD COLUMN detections_json TEXT"
            )
            print("[DB] Migrated: added 'detections_json' column.")

        if "clean_frame_path" not in existing_cols:
            conn.execute(
                "ALTER TABLE analyses ADD COLUMN clean_frame_path TEXT"
            )
            print("[DB] Migrated: added 'clean_frame_path' column.")

        conn.commit()
        conn.close()

    def save_analysis(self, frame_id, timestamp, frame_path,
                      clean_frame_path, analysis_text, classification,
                      person_ids, num_persons, detections_json=None):
        with self.lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO analyses
                   (frame_id, timestamp, frame_path, clean_frame_path,
                    analysis_text, classification, person_ids, num_persons,
                    detections_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    frame_id, timestamp, frame_path, clean_frame_path,
                    analysis_text, classification,
                    json.dumps(person_ids), num_persons,
                    json.dumps(detections_json) if detections_json else None,
                ),
            )
            conn.commit()
            conn.close()

    def query_analyses(self, classification=None, start_ts=None,
                       end_ts=None, limit=50, offset=0):
        """Query saved analyses with optional filters."""
        with self.lock:
            conn = self._get_conn()
            conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM analyses WHERE 1=1"
            params = []

            if classification:
                sql += " AND classification = ?"
                params.append(classification)
            if start_ts:
                sql += " AND timestamp >= ?"
                params.append(start_ts)
            if end_ts:
                sql += " AND timestamp <= ?"
                params.append(end_ts)

            sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(sql, params).fetchall()
            conn.close()

            results = []
            for r in rows:
                d = dict(r)
                # Parse detections_json back to list for API consumers
                if d.get("detections_json"):
                    try:
                        d["detections_json"] = json.loads(d["detections_json"])
                    except (json.JSONDecodeError, TypeError):
                        d["detections_json"] = []
                else:
                    d["detections_json"] = []
                results.append(d)
            return results

    def get_stats(self):
        with self.lock:
            conn = self._get_conn()
            total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            suspicious = conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE classification='suspicious'"
            ).fetchone()[0]
            normal = conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE classification='normal'"
            ).fetchone()[0]
            conn.close()
            return {"total": total, "suspicious": suspicious, "normal": normal}