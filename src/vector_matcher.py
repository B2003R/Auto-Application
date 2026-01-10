"""
Vector matching module for Job Application Co-Pilot.
Uses sentence-transformers for embeddings and cosine similarity for matching.
"""

import logging
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# PDF support via PyMuPDF
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logger = logging.getLogger(__name__)


class VectorMatcher:
    """
    Handles vector embeddings and similarity matching.
    Implements transient vector matching - job vectors are computed in-memory
    and immediately discarded after comparison.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector matcher with specified model.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._resume_vector: Optional[np.ndarray] = None
        self._resume_text: Optional[str] = None
        
        logger.info(f"VectorMatcher initialized with model: {model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model
    
    def load_resume(self, resume_path: Path) -> bool:
        """
        Load and vectorize the user's resume from PDF.
        This is done once at startup and cached.
        
        Args:
            resume_path: Path to the resume PDF file.
            
        Returns:
            True if resume was loaded successfully.
        """
        try:
            file_ext = resume_path.suffix.lower()
            
            if file_ext != '.pdf':
                logger.error(f"Resume must be in PDF format. Got: {file_ext}")
                return False
            
            self._resume_text = self._extract_text_from_pdf(resume_path)
            
            if not self._resume_text or not self._resume_text.strip():
                logger.error(f"Resume PDF is empty or could not be parsed: {resume_path}")
                return False
            
            self._resume_vector = self.model.encode(
                self._resume_text, 
                convert_to_numpy=True,
                show_progress_bar=False
            ).reshape(1, -1)
            
            logger.info(f"Resume loaded and vectorized: {len(self._resume_text)} chars")
            return True
            
        except FileNotFoundError:
            logger.error(f"Resume file not found: {resume_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load resume: {e}")
            return False
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Extracted text content.
        """
        if not PDF_SUPPORT:
            raise ImportError(
                "PDF support requires PyMuPDF. Install with: pip install pymupdf"
            )
        
        text_parts = []
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
                logger.debug(f"Extracted {len(page_text)} chars from page {page_num + 1}")
        doc.close()
        
        full_text = "\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} chars from {num_pages} PDF pages")
        return full_text
    
    def load_resume_from_text(self, resume_text: str) -> bool:
        """
        Load and vectorize resume from text directly.
        
        Args:
            resume_text: The resume content as string.
            
        Returns:
            True if resume was vectorized successfully.
        """
        try:
            self._resume_text = resume_text
            self._resume_vector = self.model.encode(
                resume_text,
                convert_to_numpy=True,
                show_progress_bar=False
            ).reshape(1, -1)
            
            logger.info(f"Resume vectorized: {len(resume_text)} chars")
            return True
            
        except Exception as e:
            logger.error(f"Failed to vectorize resume: {e}")
            return False
    
    @property
    def resume_text(self) -> Optional[str]:
        """Get the loaded resume text."""
        return self._resume_text
    
    @property
    def is_resume_loaded(self) -> bool:
        """Check if resume is loaded and vectorized."""
        return self._resume_vector is not None
    
    def compute_similarity(self, job_description: str) -> Tuple[float, bool]:
        """
        Compute cosine similarity between resume and job description.
        
        This implements TRANSIENT vector matching:
        - Job vector is computed in-memory
        - Similarity is calculated
        - Job vector is immediately eligible for garbage collection
        
        Args:
            job_description: The job description text.
            
        Returns:
            Tuple of (similarity_score, is_above_threshold)
            Score is between 0 and 1, where 1 is perfect match.
            
        Raises:
            ValueError: If resume is not loaded.
        """
        if not self.is_resume_loaded:
            raise ValueError("Resume not loaded. Call load_resume() first.")
        
        # Compute job vector (transient - will be GC'd after function returns)
        job_vector = self.model.encode(
            job_description,
            convert_to_numpy=True,
            show_progress_bar=False
        ).reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(self._resume_vector, job_vector)[0][0]
        
        # Normalize to 0-1 range (cosine similarity is already in this range for normalized vectors)
        score = float(max(0.0, min(1.0, similarity)))
        
        logger.debug(f"Similarity score: {score:.4f}")
        
        # job_vector goes out of scope here and will be garbage collected
        return score
    
    def matches_threshold(self, job_description: str, threshold: float = 0.75) -> Tuple[float, bool]:
        """
        Check if job description meets similarity threshold.
        
        Args:
            job_description: The job description text.
            threshold: Minimum similarity score to pass (default: 0.75).
            
        Returns:
            Tuple of (similarity_score, passes_threshold).
        """
        score = self.compute_similarity(job_description)
        passes = score >= threshold
        
        logger.info(f"Similarity: {score:.4f} | Threshold: {threshold} | Pass: {passes}")
        
        return score, passes
    
    def batch_compute_similarities(self, job_descriptions: list[str]) -> list[float]:
        """
        Compute similarities for multiple job descriptions efficiently.
        Uses batch encoding for better performance.
        
        Args:
            job_descriptions: List of job description texts.
            
        Returns:
            List of similarity scores.
        """
        if not self.is_resume_loaded:
            raise ValueError("Resume not loaded. Call load_resume() first.")
        
        if not job_descriptions:
            return []
        
        # Batch encode all job descriptions
        job_vectors = self.model.encode(
            job_descriptions,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Compute all similarities at once
        similarities = cosine_similarity(self._resume_vector, job_vectors)[0]
        
        # Normalize and convert to list
        scores = [float(max(0.0, min(1.0, s))) for s in similarities]
        
        logger.info(f"Batch computed {len(scores)} similarities")
        
        return scores
    
    def preload_model(self):
        """
        Explicitly preload the model.
        Useful for ensuring model is loaded before processing begins.
        """
        _ = self.model
        logger.info("Model preloaded successfully")


class ResumeJobMatcher:
    """
    High-level interface for resume-job matching.
    Encapsulates the matching logic with threshold handling.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75
    ):
        """
        Initialize the matcher.
        
        Args:
            model_name: Sentence transformer model name.
            similarity_threshold: Minimum score to accept a job.
        """
        self.matcher = VectorMatcher(model_name)
        self.threshold = similarity_threshold
    
    def initialize(self, resume_path: Path) -> bool:
        """
        Initialize matcher with resume.
        
        Args:
            resume_path: Path to resume text file.
            
        Returns:
            True if initialization successful.
        """
        self.matcher.preload_model()
        return self.matcher.load_resume(resume_path)
    
    def evaluate_job(self, job_description: str) -> dict:
        """
        Evaluate a job description against the resume.
        
        Args:
            job_description: Job description text.
            
        Returns:
            Dictionary with 'score', 'passes', and 'recommendation'.
        """
        score, passes = self.matcher.matches_threshold(
            job_description, 
            self.threshold
        )
        
        if passes:
            if score >= 0.90:
                recommendation = "EXCELLENT_MATCH"
            elif score >= 0.85:
                recommendation = "STRONG_MATCH"
            else:
                recommendation = "GOOD_MATCH"
        else:
            if score >= 0.70:
                recommendation = "BORDERLINE"
            elif score >= 0.60:
                recommendation = "WEAK_MATCH"
            else:
                recommendation = "NO_MATCH"
        
        return {
            "score": score,
            "passes": passes,
            "recommendation": recommendation,
            "threshold": self.threshold
        }
