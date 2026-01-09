# Job Application Co-Pilot

> A local automation tool for job application management that sources, filters, and tailors resumes while keeping humans in the loop.

## Features

- **Smart Discovery**: Automated job sourcing via Bright Data API
- **Intelligent Filtering**: Vector-based matching with configurable thresholds
- **Deduplication**: Tombstone pattern prevents reprocessing
- **AI-Powered Tailoring**: GPT-4o customizes your resume for each job
- **LaTeX Quality**: Professional PDF output via pdflatex
- **Cost Optimized**: Aggressive filtering minimizes API calls
- **Human-in-the-Loop**: Final application submission is always manual

## Project Structure

```
Auto-Application/
 src/
    __init__.py          # Package initialization
    config.py            # Configuration management
    database.py          # SQLite operations & deduplication
    vector_matcher.py    # Embedding & similarity matching
    scraper.py           # Bright Data API integration
    tailor.py            # OpenAI resume tailoring
    compiler.py          # LaTeX to PDF compilation
    main.py              # Main orchestrator & CLI
 data/
    companies.json       # Target company list
    resume.txt           # Your resume (plain text)
    generated/           # Output PDFs
 templates/
    resume_template.tex  # LaTeX template
 temp/                    # Temporary compilation files
 job_history.db           # SQLite database (auto-created)
 requirements.txt         # Python dependencies
 .env.example             # Environment variables template
 README.md                # This file
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- LaTeX distribution (MiKTeX or TeX Live) with pdflatex
- API keys for OpenAI and Bright Data

### 2. Installation

```bash
cd Auto-Application
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Setup Your Data

1. Edit data/companies.json - Add your target companies
2. Edit data/resume.txt - Paste your resume in plain text format

### 5. Run the Pipeline

```bash
python -m src.main           # Full pipeline
python -m src.main --dry-run # No API calls
python -m src.main --mock    # Use mock scraper
python -m src.main --ready   # List ready jobs
python -m src.main --stats   # Database statistics
```

## Architecture

### Phase 1: Discovery (The Net)
- Loads target companies from data/companies.json
- Searches for keywords: Software, Data, Engineer, Developer
- Filters to jobs posted in the last 24 hours

### Phase 2: Intelligence (The Gatekeeper)
- Deduplication via job_id OR MD5(description) check
- Vector matching with sentence-transformers
- Jobs below threshold (0.75) are rejected and tombstoned

### Phase 3: Factory (Creation)
- Tailoring via GPT-4o for LaTeX generation
- Compilation via pdflatex
- Storage with status=READY in SQLite

## CLI Commands

```bash
python -m src.main --dry-run           # Skip API calls
python -m src.main --mock              # Use mock scraper
python -m src.main --threshold 0.80    # Higher threshold
python -m src.main --verbose           # Debug logging
python -m src.main --stats             # Database statistics
python -m src.main --ready             # List ready jobs
```

## License

MIT License
