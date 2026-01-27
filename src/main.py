"""
Job Application Co-Pilot - Main Orchestrator

This is the main entry point that coordinates all phases:
- Phase 1: Discovery (Job Sourcing)
- Phase 2: Intelligence (Filtering & Matching)
- Phase 3: Factory (Resume Tailoring & PDF Generation)
"""

import sys
import uuid
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

# Local imports
from .config import Config, config
from .database import DatabaseManager, JobStatus
from .vector_matcher import VectorMatcher, ResumeJobMatcher
from .scraper import BrightDataJobScraper, MockJobScraper, Job, load_companies
from .tailor import ResumeTailor, LaTeXTemplateManager
from .compiler import PDFCompiler, LaTeXValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('job_copilot.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics for a pipeline run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    jobs_discovered: int = 0
    jobs_skipped_duplicate: int = 0
    jobs_rejected_score: int = 0
    jobs_processed: int = 0
    jobs_ready: int = 0
    jobs_failed: int = 0
    
    total_api_cost_estimate: float = 0.0
    
    def summary(self) -> str:
        """Generate a summary string."""
        duration = ""
        if self.completed_at:
            delta = self.completed_at - self.started_at
            duration = f" (Duration: {delta.total_seconds():.1f}s)"
        
        return f"""
+==============================================================+
|              JOB APPLICATION CO-PILOT - RUN SUMMARY          |
+==============================================================+
|  Run ID: {self.run_id:<52} |
|  Started: {self.started_at.strftime('%Y-%m-%d %H:%M:%S'):<51} |
+==============================================================+
|  PHASE 1 - DISCOVERY                                         |
|    Jobs Discovered:     {self.jobs_discovered:>5}                               |
+==============================================================+
|  PHASE 2 - INTELLIGENCE                                      |
|    Skipped (Duplicate): {self.jobs_skipped_duplicate:>5}                               |
|    Rejected (Score):    {self.jobs_rejected_score:>5}                               |
+==============================================================+
|  PHASE 3 - FACTORY                                           |
|    Processed:           {self.jobs_processed:>5}                               |
|    Ready for Apply:     {self.jobs_ready:>5}                               |
|    Failed:              {self.jobs_failed:>5}                               |
+==============================================================+
|  Est. API Cost: ${self.total_api_cost_estimate:<8.4f}                              |
+==============================================================+{duration}
"""


class JobApplicationCoPilot:
    """
    Main orchestrator for the Job Application Co-Pilot.
    
    Implements the three-phase pipeline:
    1. Discovery - Source jobs from Bright Data API
    2. Intelligence - Filter and match jobs to resume
    3. Factory - Tailor resumes and generate PDFs
    """
    
    def __init__(self, config: Config, use_mock_scraper: bool = False):
        """
        Initialize the Co-Pilot with configuration.
        
        Args:
            config: Configuration object.
            use_mock_scraper: Use mock scraper instead of Bright Data API.
        """
        self.config = config
        self.use_mock_scraper = use_mock_scraper
        
        # Initialize components
        self.db = DatabaseManager(config.DB_PATH)
        self.matcher = VectorMatcher(config.EMBEDDING_MODEL)
        
        if use_mock_scraper:
            self.scraper = MockJobScraper()
            logger.info("Using MOCK scraper for testing")
        else:
            self.scraper = BrightDataJobScraper(
                api_key=config.BRIGHT_DATA_API_KEY,
                dataset_id=config.BRIGHT_DATA_DATASET_ID,
                default_country=config.DEFAULT_COUNTRY,
                default_location=config.DEFAULT_LOCATION
            )
        
        self.tailor = ResumeTailor(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            max_tokens=config.OPENAI_MAX_TOKENS,
            temperature=config.OPENAI_TEMPERATURE
        )
        
        self.compiler = PDFCompiler(
            pdflatex_path=config.PDFLATEX_PATH,
            temp_dir=config.TEMP_DIR,
            output_dir=config.GENERATED_DIR
        )
        
        self.template_manager = LaTeXTemplateManager(template_type="minimal")
        
        # Stats tracking
        self.stats = PipelineStats()
        
        logger.info("JobApplicationCoPilot initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the pipeline - load resume and preload models.
        
        Returns:
            True if initialization successful.
        """
        logger.info("Initializing pipeline...")
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            for error in errors:
                logger.error(f"Config error: {error}")
            return False
        
        # Load resume PDF
        resume_path = self.config.DATA_DIR / "Resume.pdf"
        if not self.matcher.load_resume(resume_path):
            logger.error("Failed to load resume PDF")
            return False
        
        # Preload embedding model
        self.matcher.preload_model()
        
        logger.info("Pipeline initialization complete")
        return True
    
    def run(self, dry_run: bool = False) -> PipelineStats:
        """
        Execute the full pipeline.
        
        Args:
            dry_run: If True, don't make API calls to OpenAI or Bright Data.
            
        Returns:
            Pipeline statistics.
        """
        self.stats = PipelineStats()
        self.db.create_run_stats(self.stats.run_id)
        
        logger.info(f"Starting pipeline run: {self.stats.run_id}")
        
        try:
            # Phase 1: Discovery
            jobs = self._phase_discovery()
            self.stats.jobs_discovered = len(jobs)
            
            if not jobs:
                logger.warning("No jobs discovered")
                self.stats.completed_at = datetime.now()
                return self.stats
            
            # Phase 2: Intelligence
            qualified_jobs = self._phase_intelligence(jobs)
            
            if not qualified_jobs:
                logger.info("No qualified jobs after filtering")
                self.stats.completed_at = datetime.now()
                return self.stats
            
            # Phase 3: Factory
            if not dry_run:
                self._phase_factory(qualified_jobs)
            else:
                logger.info(f"DRY RUN: Would process {len(qualified_jobs)} jobs")
                self.stats.jobs_processed = len(qualified_jobs)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
        
        finally:
            self.stats.completed_at = datetime.now()
            self._update_run_stats()
        
        logger.info(self.stats.summary())
        return self.stats
    
    def _phase_discovery(self) -> List[Job]:
        """
        Phase 1: Discovery - Source jobs from API.
        
        Returns:
            List of discovered Job objects.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: DISCOVERY - Casting the Net")
        logger.info("=" * 60)
        
        # Load target companies
        companies_file = self.config.DATA_DIR / "companies.json"
        companies = load_companies(companies_file)
        
        if not companies:
            logger.error("No target companies loaded")
            return []
        
        logger.info(f"Targeting {len(companies)} companies")
        logger.info(f"Search keywords: {self.config.SEARCH_KEYWORDS}")
        logger.info(f"Time range: {self.config.TIME_RANGE}")
        
        # Scrape jobs
        jobs = self.scraper.scrape_jobs(
            companies=companies,
            keywords=self.config.SEARCH_KEYWORDS,
            time_range=self.config.TIME_RANGE
        )
        
        logger.info(f"Discovered {len(jobs)} jobs")
        self.db.log_action(None, "DISCOVERY_COMPLETE", f"Found {len(jobs)} jobs")
        
        return jobs
    
    def _phase_intelligence(self, jobs: List[Job]) -> List[Dict[str, Any]]:
        """
        Phase 2: Intelligence - Filter and match jobs.
        
        Implements the Gatekeeper logic:
        1. Deduplication (Tombstone pattern)
        2. Transient Vector Matching
        
        Args:
            jobs: List of discovered jobs.
            
        Returns:
            List of qualified jobs with their scores.
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: INTELLIGENCE - The Gatekeeper")
        logger.info("=" * 60)
        
        qualified = []
        
        for job in jobs:
            # Step 1: Deduplication Check (Tombstone)
            if self.db.is_duplicate(job.job_id, job.description):
                logger.info(f"SKIP (already processed): {job.company} - {job.title}")
                self.stats.jobs_skipped_duplicate += 1
                continue
            
            # Step 2: Vector Matching
            score = self.matcher.compute_similarity(job.description)
            
            if score < self.config.SIMILARITY_THRESHOLD:
                # Reject and tombstone
                logger.info(f"REJECT: {job.company} - {job.title} (score: {score:.3f})")
                
                self.db.insert_job(
                    job_id=job.job_id,
                    company=job.company,
                    title=job.title,
                    job_description=job.description,
                    status=JobStatus.REJECTED,
                    similarity_score=score
                )
                
                self.stats.jobs_rejected_score += 1
                
                # IMMEDIATELY DELETE from memory (Python GC will handle)
                # The job object goes out of scope after this iteration
                continue
            
            # Job passes - add to qualified list
            logger.info(f"QUALIFIED: {job.company} - {job.title} (score: {score:.3f})")
            
            qualified.append({
                "job": job,
                "score": score
            })
        
        logger.info(f"Qualified jobs: {len(qualified)}/{len(jobs)}")
        self.db.log_action(
            None, 
            "INTELLIGENCE_COMPLETE", 
            f"Qualified: {len(qualified)}, Rejected: {self.stats.jobs_rejected_score}, Duplicates: {self.stats.jobs_skipped_duplicate}"
        )
        
        return qualified
    
    def _phase_factory(self, qualified_jobs: List[Dict[str, Any]]):
        """
        Phase 3: Factory - Tailor resumes and generate PDFs.
        
        Args:
            qualified_jobs: List of qualified jobs with scores.
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: FACTORY - Resume Tailoring")
        logger.info("=" * 60)
        
        resume_text = self.matcher.resume_text
        
        for item in qualified_jobs:
            job: Job = item["job"]
            score: float = item["score"]
            
            logger.info(f"Processing: {job.company} - {job.title}")
            
            try:
                # Step 1: Tailor resume with GPT-4o
                result = self.tailor.tailor_resume(
                    resume_text=resume_text,
                    job_description=job.description,
                    company=job.company,
                    job_title=job.title
                )
                
                if not result["success"]:
                    logger.error(f"Tailoring failed: {result['error']}")
                    self.db.insert_job(
                        job_id=job.job_id,
                        company=job.company,
                        title=job.title,
                        job_description=job.description,
                        status=JobStatus.PENDING,
                        similarity_score=score,
                        apply_link=job.apply_link
                    )
                    self.stats.jobs_failed += 1
                    continue
                
                # Track API cost
                if result.get("usage"):
                    cost = self.tailor.estimate_cost(resume_text, job.description)
                    self.stats.total_api_cost_estimate += cost["estimated_cost_usd"]
                
                # Step 2: Wrap content with template
                latex_content = result["latex_content"]
                full_document = self.template_manager.wrap_content(latex_content)
                
                # Step 3: Validate LaTeX
                validation = LaTeXValidator.validate(full_document)
                if not validation["valid"]:
                    logger.warning(f"LaTeX validation errors: {validation['errors']}")
                    # Try to continue anyway - pdflatex will catch real errors
                
                # Step 4: Compile to PDF
                compile_result = self.compiler.compile(
                    latex_content=full_document,
                    company=job.company,
                    role=job.title,
                    job_id=job.job_id
                )
                
                if not compile_result["success"]:
                    logger.error(f"PDF compilation failed: {compile_result['error']}")
                    
                    # Save the .tex file for debugging
                    debug_tex = self.config.TEMP_DIR / f"debug_{job.job_id}.tex"
                    debug_tex.write_text(full_document, encoding='utf-8')
                    logger.info(f"Debug .tex saved to: {debug_tex}")
                    
                    self.db.insert_job(
                        job_id=job.job_id,
                        company=job.company,
                        title=job.title,
                        job_description=job.description,
                        status=JobStatus.PENDING,
                        similarity_score=score,
                        apply_link=job.apply_link
                    )
                    self.stats.jobs_failed += 1
                    continue
                
                # Step 5: Success - Store in database
                self.db.insert_job(
                    job_id=job.job_id,
                    company=job.company,
                    title=job.title,
                    job_description=job.description,
                    status=JobStatus.READY,
                    similarity_score=score,
                    resume_path=compile_result["pdf_path"],
                    apply_link=job.apply_link
                )
                
                self.stats.jobs_processed += 1
                self.stats.jobs_ready += 1
                
                logger.info(f"[OK] Resume ready: {compile_result['pdf_path']}")
                
            except Exception as e:
                logger.error(f"Error processing {job.job_id}: {e}", exc_info=True)
                self.stats.jobs_failed += 1
        
        self.db.log_action(
            None,
            "FACTORY_COMPLETE",
            f"Ready: {self.stats.jobs_ready}, Failed: {self.stats.jobs_failed}"
        )
    
    def _update_run_stats(self):
        """Update run statistics in database."""
        self.db.update_run_stats(
            self.stats.run_id,
            jobs_discovered=self.stats.jobs_discovered,
            jobs_skipped_duplicate=self.stats.jobs_skipped_duplicate,
            jobs_rejected_score=self.stats.jobs_rejected_score,
            jobs_processed=self.stats.jobs_processed,
            jobs_ready=self.stats.jobs_ready,
            completed=True
        )
    
    def get_ready_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all jobs ready for application.
        
        Returns:
            List of ready job records.
        """
        return self.db.get_ready_jobs()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall database statistics.
        
        Returns:
            Statistics dictionary.
        """
        return self.db.get_stats()
    
    def mark_applied(self, job_id: str):
        """
        Mark a job as applied (human action).
        
        Args:
            job_id: The job ID to mark.
        """
        self.db.update_job_status(job_id, JobStatus.APPLIED)
        self.db.log_action(job_id, "MARKED_APPLIED", "Human confirmed application submitted")
        logger.info(f"Job {job_id} marked as APPLIED")
    
    def mark_skipped(self, job_id: str, reason: Optional[str] = None):
        """
        Mark a job as skipped (human decided not to apply).
        
        Args:
            job_id: The job ID to mark.
            reason: Optional reason for skipping.
        """
        self.db.update_job_status(job_id, JobStatus.SKIPPED)
        self.db.log_action(job_id, "MARKED_SKIPPED", reason or "Human decided to skip")
        logger.info(f"Job {job_id} marked as SKIPPED")
    
    def cleanup(self):
        """Clean up resources."""
        self.db.close()
        logger.info("Co-Pilot cleaned up")


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Job Application Co-Pilot - Automate your job search"
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making API calls'
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock scraper instead of Bright Data'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics and exit'
    )
    
    parser.add_argument(
        '--ready',
        action='store_true',
        help='List jobs ready for application'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.75,
        help='Similarity threshold (default: 0.75)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Update config with CLI args
    if args.threshold != 0.75:
        config.SIMILARITY_THRESHOLD = args.threshold
    
    # Initialize Co-Pilot
    copilot = JobApplicationCoPilot(config, use_mock_scraper=args.mock)
    
    try:
        if args.stats:
            # Just show statistics
            stats = copilot.get_statistics()
            print("\n📊 DATABASE STATISTICS")
            print("-" * 40)
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return
        
        if args.ready:
            # List ready jobs
            ready = copilot.get_ready_jobs()
            print(f"\n[JOBS READY FOR APPLICATION] ({len(ready)} total)")
            print("-" * 60)
            for job in ready:
                print(f"\n  {job['company']} - {job['title']}")
                print(f"     Score: {job['similarity_score']:.2%}")
                print(f"     Resume PDF: {job['resume_path']}")
                print(f"     Apply Link: {job['apply_link']}")
            return
        
        # Run the pipeline
        if not copilot.initialize():
            logger.error("Failed to initialize pipeline")
            sys.exit(1)
        
        stats = copilot.run(dry_run=args.dry_run)
        
        # Show ready jobs after run
        if stats.jobs_ready > 0:
            print("\n🎯 JOBS READY FOR YOUR REVIEW:")
            for job in copilot.get_ready_jobs()[:5]:  # Show top 5
                print(f"  • {job['company']} - {job['title']} ({job['similarity_score']:.2%})")
        
    finally:
        copilot.cleanup()


if __name__ == "__main__":
    main()
