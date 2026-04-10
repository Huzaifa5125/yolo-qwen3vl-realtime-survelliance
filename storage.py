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
                vlm_latency REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_metrics (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_frames_seen INTEGER DEFAULT 0,
                yolo_runs INTEGER DEFAULT 0,
                yolo_total_latency REAL DEFAULT 0,
                yolo_avg_latency REAL DEFAULT 0,
                yolo_min_latency REAL,
                yolo_max_latency REAL,
                yolo_frames_with_detections INTEGER DEFAULT 0,
                buffer_pushes INTEGER DEFAULT 0,
                buffer_pops INTEGER DEFAULT 0,
                buffer_drops_before_vlm INTEGER DEFAULT 0,
                vlm_frames_processed INTEGER DEFAULT 0,
                vlm_total_latency REAL DEFAULT 0,
                vlm_avg_latency REAL DEFAULT 0,
                vlm_min_latency REAL,
                vlm_max_latency REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT OR IGNORE INTO pipeline_metrics (id) VALUES (1)")
        conn.commit()
        conn.close()

    def _migrate_db(self):
        """Add missing columns/tables if they don't exist (safe migrations)."""
        conn = self._get_conn()
        self._migrate_analyses_table(conn)
        self._migrate_pipeline_metrics_table(conn)

        conn.commit()
        conn.close()

    def _migrate_analyses_table(self, conn):
        cursor = conn.execute("PRAGMA table_info(analyses)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        if "detections_json" not in existing_cols:
            conn.execute("ALTER TABLE analyses ADD COLUMN detections_json TEXT")
            print("[DB] Migrated: added 'detections_json' column.")

        if "clean_frame_path" not in existing_cols:
            conn.execute("ALTER TABLE analyses ADD COLUMN clean_frame_path TEXT")
            print("[DB] Migrated: added 'clean_frame_path' column.")

        if "vlm_latency" not in existing_cols:
            conn.execute("ALTER TABLE analyses ADD COLUMN vlm_latency REAL")
            print("[DB] Migrated: added 'vlm_latency' column.")

    def _migrate_pipeline_metrics_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_metrics (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_frames_seen INTEGER DEFAULT 0,
                yolo_runs INTEGER DEFAULT 0,
                yolo_total_latency REAL DEFAULT 0,
                yolo_avg_latency REAL DEFAULT 0,
                yolo_min_latency REAL,
                yolo_max_latency REAL,
                yolo_frames_with_detections INTEGER DEFAULT 0,
                buffer_pushes INTEGER DEFAULT 0,
                buffer_pops INTEGER DEFAULT 0,
                buffer_drops_before_vlm INTEGER DEFAULT 0,
                vlm_frames_processed INTEGER DEFAULT 0,
                vlm_total_latency REAL DEFAULT 0,
                vlm_avg_latency REAL DEFAULT 0,
                vlm_min_latency REAL,
                vlm_max_latency REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT OR IGNORE INTO pipeline_metrics (id) VALUES (1)")
        cursor = conn.execute("PRAGMA table_info(pipeline_metrics)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        required_cols = {
            "total_frames_seen": "INTEGER DEFAULT 0",
            "yolo_runs": "INTEGER DEFAULT 0",
            "yolo_total_latency": "REAL DEFAULT 0",
            "yolo_avg_latency": "REAL DEFAULT 0",
            "yolo_min_latency": "REAL",
            "yolo_max_latency": "REAL",
            "yolo_frames_with_detections": "INTEGER DEFAULT 0",
            "buffer_pushes": "INTEGER DEFAULT 0",
            "buffer_pops": "INTEGER DEFAULT 0",
            "buffer_drops_before_vlm": "INTEGER DEFAULT 0",
            "vlm_frames_processed": "INTEGER DEFAULT 0",
            "vlm_total_latency": "REAL DEFAULT 0",
            "vlm_avg_latency": "REAL DEFAULT 0",
            "vlm_min_latency": "REAL",
            "vlm_max_latency": "REAL",
            "updated_at": "DATETIME DEFAULT CURRENT_TIMESTAMP",
        }
        for col_name, col_def in required_cols.items():
            if col_name not in existing_cols:
                conn.execute(
                    f"ALTER TABLE pipeline_metrics ADD COLUMN {col_name} {col_def}"
                )
                print(f"[DB] Migrated: added '{col_name}' to pipeline_metrics.")

    def save_analysis(self, frame_id, timestamp, frame_path,
                      clean_frame_path, analysis_text, classification,
                      person_ids, num_persons, detections_json=None, vlm_latency=None):
        with self.lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO analyses
                   (frame_id, timestamp, frame_path, clean_frame_path,
                    analysis_text, classification, person_ids, num_persons,
                    detections_json, vlm_latency)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    frame_id, timestamp, frame_path, clean_frame_path,
                    analysis_text, classification,
                    json.dumps(person_ids), num_persons,
                    json.dumps(detections_json) if detections_json else None,
                    vlm_latency,
                ),
            )
            conn.commit()
            conn.close()

    def upsert_pipeline_metrics(self, metrics):
        """Persist latest aggregate runtime metrics as a single snapshot row."""
        with self.lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO pipeline_metrics (
                    id,
                    total_frames_seen,
                    yolo_runs,
                    yolo_total_latency,
                    yolo_avg_latency,
                    yolo_min_latency,
                    yolo_max_latency,
                    yolo_frames_with_detections,
                    buffer_pushes,
                    buffer_pops,
                    buffer_drops_before_vlm,
                    vlm_frames_processed,
                    vlm_total_latency,
                    vlm_avg_latency,
                    vlm_min_latency,
                    vlm_max_latency,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    total_frames_seen=excluded.total_frames_seen,
                    yolo_runs=excluded.yolo_runs,
                    yolo_total_latency=excluded.yolo_total_latency,
                    yolo_avg_latency=excluded.yolo_avg_latency,
                    yolo_min_latency=excluded.yolo_min_latency,
                    yolo_max_latency=excluded.yolo_max_latency,
                    yolo_frames_with_detections=excluded.yolo_frames_with_detections,
                    buffer_pushes=excluded.buffer_pushes,
                    buffer_pops=excluded.buffer_pops,
                    buffer_drops_before_vlm=excluded.buffer_drops_before_vlm,
                    vlm_frames_processed=excluded.vlm_frames_processed,
                    vlm_total_latency=excluded.vlm_total_latency,
                    vlm_avg_latency=excluded.vlm_avg_latency,
                    vlm_min_latency=excluded.vlm_min_latency,
                    vlm_max_latency=excluded.vlm_max_latency,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    1,
                    int(metrics.get("total_frames_seen", 0)),
                    int(metrics.get("yolo_runs", 0)),
                    float(metrics.get("yolo_total_latency", 0.0)),
                    float(metrics.get("yolo_avg_latency", 0.0)),
                    metrics.get("yolo_min_latency"),
                    metrics.get("yolo_max_latency"),
                    int(metrics.get("yolo_frames_with_detections", 0)),
                    int(metrics.get("buffer_pushes", 0)),
                    int(metrics.get("buffer_pops", 0)),
                    int(metrics.get("buffer_drops_before_vlm", 0)),
                    int(metrics.get("vlm_frames_processed", 0)),
                    float(metrics.get("vlm_total_latency", 0.0)),
                    float(metrics.get("vlm_avg_latency", 0.0)),
                    metrics.get("vlm_min_latency"),
                    metrics.get("vlm_max_latency"),
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
            metrics_row = conn.execute(
                """
                SELECT
                    total_frames_seen,
                    yolo_runs,
                    yolo_total_latency,
                    yolo_avg_latency,
                    yolo_min_latency,
                    yolo_max_latency,
                    yolo_frames_with_detections,
                    buffer_pushes,
                    buffer_pops,
                    buffer_drops_before_vlm,
                    vlm_frames_processed,
                    vlm_total_latency,
                    vlm_avg_latency,
                    vlm_min_latency,
                    vlm_max_latency,
                    updated_at
                FROM pipeline_metrics
                WHERE id = 1
                """
            ).fetchone()
            conn.close()
            result = {"total": total, "suspicious": suspicious, "normal": normal}
            if metrics_row is not None:
                result.update({
                    "total_frames_seen": metrics_row[0],
                    "yolo_runs": metrics_row[1],
                    "yolo_total_latency": metrics_row[2],
                    "yolo_avg_latency": metrics_row[3],
                    "yolo_min_latency": metrics_row[4],
                    "yolo_max_latency": metrics_row[5],
                    "yolo_frames_with_detections": metrics_row[6],
                    "buffer_pushes": metrics_row[7],
                    "buffer_pops": metrics_row[8],
                    "buffer_drops_before_vlm": metrics_row[9],
                    "vlm_frames_processed": metrics_row[10],
                    "vlm_total_latency": metrics_row[11],
                    "vlm_avg_latency": metrics_row[12],
                    "vlm_min_latency": metrics_row[13],
                    "vlm_max_latency": metrics_row[14],
                    "metrics_updated_at": metrics_row[15],
                })
            return result
