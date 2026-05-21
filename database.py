import psycopg2
from psycopg2.extras import DictCursor
from config import Config
from logger import adfs_logger
from schemas import ProcessResult

class DatabaseArchiver:
    def __init__(self):
        self.conn = None
        self.enabled = bool(Config.DATABASE_URL)

        if self.enabled:
            self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect(Config.DATABASE_URL)
            self._init_db()
            adfs_logger.info("Connected to PostgreSQL for archival.", extra={"stage": "database"})
        except Exception as e:
            adfs_logger.error(f"PostgreSQL connection failed: {e}", extra={"stage": "database"})
            self.enabled = False

    def _init_db(self):
        """Creates the audit table automatically on first boot."""
        with self.conn.cursor() as cur:
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS filings (
                                                               id SERIAL PRIMARY KEY,
                                                               timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
                        """)
            self.conn.commit()

    def archive_filing(self, record: ProcessResult):
        """Inserts a single filing result into the database."""
        if not self.enabled or not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                            INSERT INTO filings
                                (pdf_file, tag, company, action, message, success)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """, (
                                record.get("file", "unknown"),
                                record.get("tag"),
                                record.get("company"),
                                record.get("action", "unknown"),
                                record.get("message"),
                                record.get("success", False)
                            ))
            self.conn.commit()
        except Exception as e:
            adfs_logger.error(f"Failed to archive filing to DB: {e}", extra={"stage": "database"})
            self.conn.rollback() # Reset transaction state on failure