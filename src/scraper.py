"""
Job scraper module for Job Application Co-Pilot.
Integrates with Bright Data API for job sourcing.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents a scraped job posting."""
    job_id: str
    company: str
    title: str
    description: str
    location: Optional[str] = None
    apply_link: Optional[str] = None
    salary: Optional[str] = None
    posted_date: Optional[str] = None
    job_type: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_bright_data(cls, data: Dict[str, Any]) -> "Job":
        """
        Create Job instance from Bright Data LinkedIn Jobs API response.
        
        LinkedIn API field names:
        - job_posting_id: unique job identifier
        - job_title: position title
        - company_name: employer name
        - job_summary: job description text
        - job_location: city/state
        - apply_link: application URL
        - url: LinkedIn job page URL
        - job_base_pay_range: salary info
        - job_posted_date: ISO timestamp
        - job_employment_type: Full-time, Part-time, etc.
        """
        # Extract job ID from URL if job_posting_id not present
        job_id = data.get("job_posting_id") or data.get("url", "").split("/")[-1].split("?")[0]
        
        # Use apply_link if available, otherwise use LinkedIn job URL
        apply_link = data.get("apply_link") or data.get("url")
        
        return cls(
            job_id=str(job_id),
            company=data.get("company_name", "Unknown Company"),
            title=data.get("job_title", "Unknown Position"),
            description=data.get("job_summary", ""),
            location=data.get("job_location"),
            apply_link=apply_link,
            salary=data.get("job_base_pay_range"),
            posted_date=data.get("job_posted_date"),
            job_type=data.get("job_employment_type"),
            raw_data=data
        )


class BrightDataJobScraper:
    """
    Job scraper using Bright Data's LinkedIn Job Scraping API.
    
    Implements the "Broad Net" strategy:
    - Search across multiple keywords
    - Filter by time range
    - Target specific companies
    
    API Reference: Bright Data LinkedIn Jobs Scraper
    Endpoint: https://api.brightdata.com/datasets/v3/scrape
    """
    
    BASE_URL = "https://api.brightdata.com/datasets/v3/scrape"
    
    # Mapping from our time_range values to Bright Data's expected values
    # Valid LinkedIn time ranges: Past 24 hours, Past week, Past month
    TIME_RANGE_MAP = {
        "past_24h": "Past 24 hours",
        "past_week": "Past week",
        "past_month": "Past month",
        "any": ""  # No filter
    }
    
    # Valid job types for LinkedIn
    JOB_TYPE_MAP = {
        "full_time": "Full-time",
        "part_time": "Part-time", 
        "contract": "Contract",
        "temporary": "Temporary",
        "internship": "Internship",
        "volunteer": "Volunteer",
        "other": "Other"
    }
    
    # Experience levels
    EXPERIENCE_MAP = {
        "internship": "Internship",
        "entry": "Entry level",
        "associate": "Associate",
        "mid_senior": "Mid-Senior level",
        "director": "Director",
        "executive": "Executive"
    }
    
    # Remote options
    REMOTE_MAP = {
        "onsite": "On-site",
        "remote": "Remote",
        "hybrid": "Hybrid"
    }
    
    def __init__(
        self,
        api_key: str,
        dataset_id: str,
        timeout: int = 300,
        default_country: str = "US",
        default_location: str = "United States"
    ):
        """
        Initialize the Bright Data scraper.
        
        Args:
            api_key: Bright Data API key.
            dataset_id: Dataset ID for LinkedIn job scraping.
            timeout: Request timeout in seconds.
            default_country: Default country code (e.g., "US", "FR").
            default_location: Default location string.
        """
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.timeout = timeout
        self.default_country = default_country
        self.default_location = default_location
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def _build_input_queries(
        self,
        companies: List[str],
        keywords: List[str],
        time_range: str = "past_24h",
        location: Optional[str] = None,
        country: Optional[str] = None,
        job_type: Optional[str] = None,
        experience_level: Optional[str] = None,
        remote: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build the input queries array for Bright Data API.
        
        Args:
            companies: List of company names to search.
            keywords: Search keywords.
            time_range: Time filter (past_24h, past_week, past_month).
            location: Location string (e.g., "New York", "San Francisco").
            country: Country code (e.g., "US", "FR").
            job_type: Job type filter.
            experience_level: Experience level filter.
            remote: Remote work option.
            
        Returns:
            List of input query dictionaries.
        """
        queries = []
        
        # Convert time_range to Bright Data format
        bd_time_range = self.TIME_RANGE_MAP.get(time_range, "Past 24 hours")
        
        # Convert other filters if provided
        bd_job_type = self.JOB_TYPE_MAP.get(job_type, "") if job_type else ""
        bd_remote = self.REMOTE_MAP.get(remote, "") if remote else ""
        
        # Fixed location for United States
        loc = "United States of America"
        ctry = "US"
        
        # Experience levels to search: Entry level and Internship
        experience_levels = ["Entry level", "Internship"]
        
        # Build queries: company in company field, keyword in keyword field
        # Create queries for each combination of company, keyword, and experience level
        for company in companies:
            for keyword in keywords:
                for exp_level in experience_levels:
                    query = {
                        "location": loc,
                        "keyword": keyword,
                        "country": ctry,
                        "time_range": bd_time_range,
                        "job_type": bd_job_type,
                        "experience_level": exp_level,
                        "remote": bd_remote,
                        "company": company,  # Company name in company filter
                        "location_radius": ""
                    }
                    queries.append(query)
        
        return queries
    
    def _trigger_collection(
        self,
        companies: List[str],
        keywords: List[str],
        time_range: str = "past_24h",
        location: Optional[str] = None,
        country: Optional[str] = None,
        job_type: Optional[str] = None,
        experience_level: Optional[str] = None,
        remote: Optional[str] = None
    ) -> Optional[str]:
        """
        Trigger a new data collection job using Bright Data's scrape endpoint.
        
        Returns:
            Snapshot ID for tracking the collection, or None on failure.
        """
        # Build input queries
        input_queries = self._build_input_queries(
            companies=companies,
            keywords=keywords,
            time_range=time_range,
            location=location,
            country=country,
            job_type=job_type,
            experience_level=experience_level,
            remote=remote
        )
        
        # Build the request payload
        payload = {
            "input": input_queries
        }
        
        # Build URL with query parameters
        url = (
            f"{self.BASE_URL}"
            f"?dataset_id={self.dataset_id}"
            f"&notify=false"
            f"&include_errors=true"
            f"&type=discover_new"
            f"&discover_by=keyword"
        )
        
        try:
            logger.info(f"Triggering collection with {len(input_queries)} queries")
            logger.debug(f"Request URL: {url}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = self.session.post(
                url,
                data=json.dumps(payload),
                timeout=300
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            # 200 = immediate response with data, 202 = accepted for async processing
            if response.status_code in (200, 202):
                # Handle both JSON and NDJSON responses
                content = response.text
                
                # Parse response - could be single JSON or NDJSON
                results = []
                try:
                    # Try parsing as single JSON object
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        results = parsed
                    else:
                        results = [parsed]
                except json.JSONDecodeError:
                    # Parse as NDJSON (newline-delimited JSON)
                    for line in content.strip().split('\n'):
                        if line.strip():
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                
                # Check if we got a snapshot_id (async) or immediate data
                if results and "snapshot_id" in results[0]:
                    snapshot_id = results[0].get("snapshot_id")
                    logger.info(f"Collection triggered (async). Snapshot ID: {snapshot_id}")
                    return {"type": "async", "snapshot_id": snapshot_id}
                else:
                    # Immediate data returned - these are the job results
                    logger.info(f"Collection completed (sync). Got {len(results)} records immediately.")
                    return {"type": "sync", "data": results}
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {e}")
            logger.error(f"Response content: {response.text[:500]}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger collection: {e}")
            return None
    
    def _check_status(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Check the status of a collection job.
        
        Bright Data status values:
        - "running" : Collection in progress
        - "ready" : Collection completed, data available
        - "failed" : Collection failed
        
        Args:
            snapshot_id: The snapshot ID to check.
            
        Returns:
            Status information dictionary.
        """
        try:
            # Use the progress endpoint
            url = f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
            response = self.session.get(url, timeout=30)
            
            logger.debug(f"Status check response: {response.status_code} - {response.text[:200]}")
            
            if response.status_code == 200:
                data = response.json()
                # Normalize status field
                status = data.get("status", "unknown").lower()
                return {
                    "status": status,
                    "progress": data.get("progress", 0),
                    "records": data.get("records", 0),
                    "errors": data.get("errors", []),
                    "raw": data
                }
            else:
                logger.warning(f"Status check returned {response.status_code}: {response.text}")
                return {"status": "running", "progress": 0}  # Assume still running
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check status: {e}")
            return {"status": "error", "error": str(e)}
    
    def _download_results(self, snapshot_id: str) -> List[Dict[str, Any]]:
        """
        Download results from a completed collection.
        
        Args:
            snapshot_id: The snapshot ID to download.
            
        Returns:
            List of job data dictionaries.
        """
        try:
            url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Response might be NDJSON (newline-delimited JSON)
            content = response.text
            
            # Try parsing as JSON array first
            try:
                data = response.json()
                if isinstance(data, list):
                    return data
                return [data]
            except json.JSONDecodeError:
                # Parse as NDJSON
                jobs = []
                for line in content.strip().split('\n'):
                    if line:
                        jobs.append(json.loads(line))
                return jobs
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download results: {e}")
            return []
    
    def scrape_jobs(
        self,
        companies: List[str],
        keywords: List[str],
        time_range: str = "past_2hrs",
        location: Optional[str] = None,
        country: Optional[str] = None,
        job_type: Optional[str] = None,
        experience_level: Optional[str] = None,
        remote: Optional[str] = None,
        poll_interval: int = 30,
        max_wait: int = 600
    ) -> List[Job]:
        """
        Execute the full scraping workflow.
        
        Args:
            companies: List of target company names.
            keywords: Search keywords.
            time_range: Time filter ("past_24h", "past_week", "past_month").
            location: Location string (e.g., "New York").
            country: Country code (e.g., "US").
            job_type: Job type ("full_time", "part_time", etc.).
            experience_level: Experience level ("entry", "mid_senior", etc.).
            remote: Remote option ("onsite", "remote", "hybrid").
            poll_interval: Seconds between status checks.
            max_wait: Maximum seconds to wait for completion.
            
        Returns:
            List of Job objects.
        """
        logger.info(f"Starting job scrape for {len(companies)} companies with {len(keywords)} keywords")
        
        # Step 1: Trigger collection
        result = self._trigger_collection(
            companies=companies,
            keywords=keywords,
            time_range=time_range,
            location=location,
            country=country,
            job_type=job_type,
            experience_level=experience_level,
            remote=remote
        )
        
        if not result:
            logger.error("Failed to trigger collection")
            return []
        
        # Check if we got immediate data or need to poll
        if result.get("type") == "sync":
            # Data returned immediately
            raw_jobs = result.get("data", [])
            logger.info(f"Processing {len(raw_jobs)} immediately returned records")
        else:
            # Async - need to poll for completion
            snapshot_id = result.get("snapshot_id")
            if not snapshot_id:
                logger.error("No snapshot_id received for async collection")
                return []
            
            # Step 2: Poll for completion
            logger.info(f"Waiting for collection to complete (polling every {poll_interval}s, max {max_wait}s)...")
            elapsed = 0
            while elapsed < max_wait:
                status = self._check_status(snapshot_id)
                current_status = status.get("status", "unknown")
                
                if current_status == "ready":
                    records = status.get("records", 0)
                    logger.info(f"Collection completed successfully! Records: {records}")
                    break
                elif current_status in ("failed", "error"):
                    logger.error(f"Collection failed: {status.get('error', status.get('errors', 'Unknown error'))}")
                    return []
                elif current_status in ("running", "unknown"):
                    progress = status.get("progress", 0)
                    records = status.get("records", 0)
                    logger.info(f"Collection in progress... Progress: {progress}% | Records so far: {records} | Elapsed: {elapsed}s")
                
                time.sleep(poll_interval)
                elapsed += poll_interval
            else:
                logger.error(f"Collection timed out after {max_wait} seconds")
                logger.info(f"You can manually check snapshot: {snapshot_id}")
                return []
            
            # Step 3: Download results
            raw_jobs = self._download_results(snapshot_id)
        logger.info(f"Downloaded {len(raw_jobs)} raw job records")
        
        # Step 4: Deduplicate by job_posting_id (API may return same job from multiple queries)
        seen_ids = set()
        unique_raw_jobs = []
        for raw_job in raw_jobs:
            job_id = raw_job.get("job_posting_id") or raw_job.get("url", "").split("/")[-1].split("?")[0]
            if job_id and job_id not in seen_ids:
                seen_ids.add(job_id)
                unique_raw_jobs.append(raw_job)
        
        if len(unique_raw_jobs) < len(raw_jobs):
            logger.info(f"Deduplicated: {len(raw_jobs)} -> {len(unique_raw_jobs)} unique jobs")
        
        # Step 5: Convert to Job objects
        jobs = []
        for raw_job in unique_raw_jobs:
            try:
                job = Job.from_bright_data(raw_job)
                if job.description:  # Only include jobs with descriptions
                    jobs.append(job)
            except Exception as e:
                logger.warning(f"Failed to parse job: {e}")
        
        logger.info(f"Parsed {len(jobs)} valid jobs with descriptions")
        return jobs
    
    def scrape_jobs_sync(
        self,
        companies: List[str],
        keywords: List[str],
        time_range: str = "past_2hrs"
    ) -> List[Job]:
        """
        Synchronous version that triggers and waits for results.
        Wrapper around scrape_jobs with default settings.
        """
        return self.scrape_jobs(
            companies=companies,
            keywords=keywords,
            time_range=time_range
        )


class MockJobScraper:
    """
    Mock scraper for testing without API calls.
    Returns sample job data for development.
    """
    
    def __init__(self):
        self.sample_jobs = [
            Job(
                job_id="mock_001",
                company="TechCorp",
                title="Senior Software Engineer",
                description="""
                We are looking for a Senior Software Engineer to join our team.
                
                Requirements:
                - 5+ years of experience in Python
                - Experience with microservices architecture
                - Strong knowledge of databases (PostgreSQL, Redis)
                - Experience with cloud platforms (AWS/GCP)
                
                Responsibilities:
                - Design and implement scalable backend services
                - Lead code reviews and mentor junior developers
                - Collaborate with product and design teams
                """,
                location="San Francisco, CA",
                apply_link="https://example.com/apply/001",
                salary="$180,000 - $220,000"
            ),
            Job(
                job_id="mock_002",
                company="DataFlow Inc",
                title="Data Engineer",
                description="""
                DataFlow Inc is hiring a Data Engineer.
                
                Qualifications:
                - 3+ years in data engineering
                - Proficiency in Python and SQL
                - Experience with Apache Spark, Airflow
                - Knowledge of data warehousing concepts
                
                What you'll do:
                - Build and maintain ETL pipelines
                - Optimize data workflows
                - Work with ML teams on feature engineering
                """,
                location="Remote",
                apply_link="https://example.com/apply/002",
                salary="$150,000 - $180,000"
            )
        ]
    
    def scrape_jobs(self, *args, **kwargs) -> List[Job]:
        """Return mock jobs for testing."""
        logger.info("Using mock scraper - returning sample jobs")
        return self.sample_jobs
    
    def scrape_jobs_sync(self, *args, **kwargs) -> List[Job]:
        """Alias for scrape_jobs."""
        return self.scrape_jobs()


def load_companies(file_path: Path) -> List[str]:
    """
    Load target companies from JSON file.
    
    Expected format:
    {
        "companies": ["Company1", "Company2", ...]
    }
    
    or simply:
    ["Company1", "Company2", ...]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            companies = data
        elif isinstance(data, dict):
            companies = data.get("companies", [])
        else:
            logger.error(f"Invalid companies file format: {file_path}")
            return []
        
        logger.info(f"Loaded {len(companies)} target companies")
        return companies
        
    except FileNotFoundError:
        logger.error(f"Companies file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse companies file: {e}")
        return []
