"""
Configuration management for Job Application Co-Pilot.
Loads settings from environment variables and provides defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Central configuration for the application."""
    
    # Paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    GENERATED_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "generated")
    TEMP_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "temp")
    
    # Database
    DB_PATH: Path = field(default_factory=lambda: Path(__file__).parent.parent / "job_history.db")
    
    # API Keys
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    BRIGHT_DATA_API_KEY: str = field(default_factory=lambda: os.getenv("BRIGHT_DATA_API_KEY", ""))
    BRIGHT_DATA_DATASET_ID: str = field(default_factory=lambda: os.getenv("BRIGHT_DATA_DATASET_ID", ""))
    
    # Vector Matching
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD: float = 0.50
    
    # Scraping
    SEARCH_KEYWORDS: List[str] = field(
        default_factory=lambda: [
            "Junior Data Engineer",
            "Business Intelligence Analyst",
            "Data Analyst",
            "ML Engineer (Entry-Level)",
            "Analytics Engineer",
            "ETL Developer",
            "Financial Data Analyst",
            "Data Quality Engineer",
            "Data Scientist",
            "Cloud Data Engineer (Azure/AWS)",
            "Software Engineering",
            "Database Developer",
            "ML Ops Engineer",
            "Python Developer (Data-focused)",
            "Healthcare Data Analyst",
            "Business Analyst",
            "Reporting Analyst",
            "Market Intelligence Analyst",
            "AI Data Engineer",
            "Prompt Engineer",
            "Data Operations Analyst",
            "Research Assistant",
            "Analytics Consultant",
            "Data Integration Engineer",
            "Risk Analytics Analyst",
            "Product Analytics Analyst",
            "Supply Chain Analyst",
            "Marketing Analytics Analyst",
            "Fraud Analytics Analyst",
            "Data Governance Analyst",
            "Pricing Analyst",
            "Customer Analytics Analyst"
            ]
    )
    TIME_RANGE: str = "past_24h"  # Options: past_24h, past_week, past_month, past_year
    DEFAULT_LOCATION: str = field(default_factory=lambda: os.getenv("DEFAULT_LOCATION", "United States"))
    DEFAULT_COUNTRY: str = field(default_factory=lambda: os.getenv("DEFAULT_COUNTRY", "US"))
    
    # OpenAI
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_MAX_TOKENS: int = 4000
    OPENAI_TEMPERATURE: float = 0.3
    
    # LaTeX
    PDFLATEX_PATH: str = field(default_factory=lambda: os.getenv("PDFLATEX_PATH", "pdflatex"))
    
    def __post_init__(self):
        """Ensure all directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")
        
        if not self.BRIGHT_DATA_API_KEY:
            errors.append("BRIGHT_DATA_API_KEY is not set")
        
        companies_file = self.DATA_DIR / "companies.json"
        if not companies_file.exists():
            errors.append(f"Companies file not found: {companies_file}")
        
        # Check for resume PDF file
        resume_pdf = self.DATA_DIR / "Resume.pdf"
        if not resume_pdf.exists():
            errors.append(f"Resume PDF not found: {resume_pdf}")
        
        return errors


# Global config instance
config = Config()
