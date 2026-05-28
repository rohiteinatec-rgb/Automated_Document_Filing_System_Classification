import psycopg2
from psycopg2 import pool
import time
import threading
from psycopg2.extras import DictCursor, RealDictCursor
from config import Config
from logger import adfs_logger
from schemas import ProcessResult
from contextlib import contextmanager

class DatabaseArchiver:
    _pool = None
    _pool_lock = threading.Lock()

    def __init__(self):
        self.enabled = bool(Config.DATABASE_URL)
        if self.enabled:
            self._initialize_pool()

    def _initialize_pool(self):
        """Creates a thread-safe connection pool with exponential backoff."""
        with self._pool_lock:
            if DatabaseArchiver._pool is None:
                max_retries = 3

                for attempt in range(max_retries):
                    try:
                        DatabaseArchiver._pool = psycopg2.pool.ThreadedConnectionPool(
                            minconn=1,
                            maxconn=10,           # Allows up to 10 concurrent API requests
                            dsn=Config.DATABASE_URL,
                            connect_timeout=5     # Prevents hanging if the DB is down
                        )
                        adfs_logger.info("Connected to PostgreSQL (Threaded Pool).", extra={"stage": "database"})
                        self._init_db()
                        break
                    except psycopg2.OperationalError as e:
                        adfs_logger.warning(f"DB Pool init failed (attempt {attempt + 1}): {e}", extra={"stage": "database"})
                        if attempt == max_retries - 1:
                            adfs_logger.error("FATAL: Could not connect to PostgreSQL.", extra={"stage": "database"})
                            self.enabled = False
                        time.sleep(2 ** attempt) # Sleeps 1s, then 2s, then 4s

    @contextmanager
    def get_connection(self):
        if not self.enabled or not DatabaseArchiver._pool:
            yield None
            return

        conn = None
        try:
            conn = DatabaseArchiver._pool.getconn()
            yield conn
        finally:
            if conn:
                DatabaseArchiver._pool.putconn(conn)

    def _init_db(self):
        """Creates the audit table automatically on first boot."""
        with self.get_connection() as conn:
            if not conn: return

            try:
                with conn.cursor() as cur:
                    cur.execute("""
                                CREATE TABLE IF NOT EXISTS filings (
                                                                       id SERIAL PRIMARY KEY,
                                                                       document_uuid UUID NOT NULL DEFAULT gen_random_uuid(),
                                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                            pdf_file VARCHAR(255) NOT NULL,
                                    tag VARCHAR(100),
                                    company VARCHAR(255),
                                    action VARCHAR(50),
                                    message TEXT,
                                    success BOOLEAN
                                    );

                                -- Create indexes for blazing fast analytics
                                CREATE INDEX IF NOT EXISTS idx_filings_tag ON filings(tag);
                                CREATE INDEX IF NOT EXISTS idx_filings_company ON filings(company);
                                CREATE INDEX IF NOT EXISTS idx_filings_action ON filings(action);
                                CREATE INDEX IF NOT EXISTS idx_filings_uuid ON filings(document_uuid);
                                """)
                conn.commit()
            except Exception as e:
                conn.rollback()
                adfs_logger.error(f"Failed to initialize schema: {e}", extra={"stage": "database"})

    def archive_filing(self, record: ProcessResult):
        """Inserts a single filing result into the database."""
        max_retries = 3
        for attempt in range(max_retries):
            with self.get_connection() as conn:
                if not conn: return None

                try:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute("""
                                    INSERT INTO filings
                                        (pdf_file, tag, company, action, message, success)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                        RETURNING document_uuid;
                                    """, (
                                        record.get("file", "unknown"),
                                        record.get("tag"),
                                        record.get("company"),
                                        record.get("action", "unknown"),
                                        record.get("message"),
                                        record.get("success", False)
                                    ))
                        result = cur.fetchone()

                    conn.commit()
                    return result['document_uuid'] # Return the UUID to the API!

                except psycopg2.OperationalError as e:
                    conn.rollback()
                    adfs_logger.warning(f"DB Insert failed (attempt {attempt + 1}): {e}", extra={"stage": "database"})
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(2 ** attempt)
                except Exception as e:
                    conn.rollback()
                    adfs_logger.error(f"Failed to archive filing to DB: {e}", extra={"stage": "database"})
                    return None