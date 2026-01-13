"""
Database layer for Job Application Co-Pilot.
Handles SQLite operations, deduplication, and job tracking.
"""

import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job processing status."""
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    READY = "READY"
    APPLIED = "APPLIED"
    SKIPPED = "SKIPPED"


@dataclass
class JobRecord:
    """Represents a job record in the database."""
    job_id: str
    company: str
    title: str
    content_hash: str
    status: JobStatus
    similarity_score: Optional[float] = None
    resume_path: Optional[str] = None
    apply_link: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DatabaseManager:
    """Manages SQLite database operations for job tracking."""
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT UNIQUE NOT NULL,
        company TEXT NOT NULL,
        title TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING',
        similarity_score REAL,
        resume_path TEXT,
        apply_link TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_job_id ON jobs(job_id);
    CREATE INDEX IF NOT EXISTS idx_content_hash ON jobs(content_hash);
    CREATE INDEX IF NOT EXISTS idx_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_company ON jobs(company);
    
    CREATE TABLE IF NOT EXISTS processing_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        action TEXT NOT NULL,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS run_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        jobs_discovered INTEGER DEFAULT 0,
        jobs_skipped_duplicate INTEGER DEFAULT 0,
        jobs_rejected_score INTEGER DEFAULT 0,
        jobs_processed INTEGER DEFAULT 0,
        jobs_ready INTEGER DEFAULT 0,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );
    """
    
    def __init__(self, db_path: Path):
        """Initialize database connection."""
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        conn.executescript(self.SCHEMA)
        conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute MD5 hash of job description."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def job_exists(self, job_id: str) -> bool:
        """Check if job_id already exists in database."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ? LIMIT 1",
            (job_id,)
        )
        return cursor.fetchone() is not None
    
    def content_hash_exists(self, content_hash: str) -> bool:
        """Check if content hash already exists (tombstone check)."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM jobs WHERE content_hash = ? LIMIT 1",
            (content_hash,)
        )
        return cursor.fetchone() is not None
    
    def is_duplicate(self, job_id: str, job_description: str) -> bool:
        """
        Check if job is a duplicate by job_id OR content hash.
        This is the Tombstone pattern - we check both identifiers.
        """
        content_hash = self.compute_content_hash(job_description)
        
        if self.job_exists(job_id):
            logger.debug(f"Duplicate found by job_id: {job_id}")
            return True
        
        if self.content_hash_exists(content_hash):
            logger.debug(f"Duplicate found by content_hash: {content_hash[:8]}...")
            return True
        
        return False
    
    def insert_job(
        self,
        job_id: str,
        company: str,
        title: str,
        job_description: str,
        status: JobStatus,
        similarity_score: Optional[float] = None,
        resume_path: Optional[str] = None,
        apply_link: Optional[str] = None
    ) -> bool:
        """Insert a new job record."""
        content_hash = self.compute_content_hash(job_description)
        
        try:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO jobs 
                (job_id, company, title, content_hash, status, similarity_score, resume_path, apply_link)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, company, title, content_hash, status.value, 
                 similarity_score, resume_path, apply_link)
            )
            conn.commit()
            logger.info(f"Inserted job: {job_id} ({company} - {title}) with status {status.value}")
            return True
        except sqlite3.IntegrityError as e:
            logger.warning(f"Failed to insert job {job_id}: {e}")
            return False
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        similarity_score: Optional[float] = None,
        resume_path: Optional[str] = None,
        apply_link: Optional[str] = None
    ):
        """Update job status and optional fields."""
        conn = self._get_connection()
        
        updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: List[Any] = [status.value]
        
        if similarity_score is not None:
            updates.append("similarity_score = ?")
            params.append(similarity_score)
        
        if resume_path is not None:
            updates.append("resume_path = ?")
            params.append(resume_path)
        
        if apply_link is not None:
            updates.append("apply_link = ?")
            params.append(apply_link)
        
        params.append(job_id)
        
        conn.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?",
            params
        )
        conn.commit()
        logger.info(f"Updated job {job_id} to status {status.value}")
    
    def log_action(self, job_id: Optional[str], action: str, details: Optional[str] = None):
        """Log a processing action."""
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO processing_log (job_id, action, details) VALUES (?, ?, ?)",
            (job_id, action, details)
        )
        conn.commit()
    
    def get_ready_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs ready for application."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT job_id, company, title, similarity_score, resume_path, apply_link, created_at
            FROM jobs 
            WHERE status = ?
            ORDER BY similarity_score DESC
            """,
            (JobStatus.READY.value,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        conn = self._get_connection()
        stats = {}
        
        cursor = conn.execute("SELECT COUNT(*) FROM jobs")
        stats['total_jobs'] = cursor.fetchone()[0]
        
        for status in JobStatus:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?",
                (status.value,)
            )
            stats[f'jobs_{status.value.lower()}'] = cursor.fetchone()[0]
        
        return stats
    
    def create_run_stats(self, run_id: str) -> int:
        """Create a new run stats record."""
        conn = self._get_connection()
        cursor = conn.execute(
            "INSERT INTO run_stats (run_id) VALUES (?)",
            (run_id,)
        )
        conn.commit()
        return cursor.lastrowid
    
    def update_run_stats(self, run_id: str, **kwargs):
        """Update run statistics."""
        conn = self._get_connection()
        
        updates = []
        params = []
        
        for key, value in kwargs.items():
            if key in ['jobs_discovered', 'jobs_skipped_duplicate', 
                       'jobs_rejected_score', 'jobs_processed', 'jobs_ready']:
                updates.append(f"{key} = ?")
                params.append(value)
        
        if 'completed' in kwargs and kwargs['completed']:
            updates.append("completed_at = CURRENT_TIMESTAMP")
        
        if updates:
            params.append(run_id)
            conn.execute(
                f"UPDATE run_stats SET {', '.join(updates)} WHERE run_id = ?",
                params
            )
            conn.commit()
