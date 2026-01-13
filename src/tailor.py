"""
Resume tailor module for Job Application Co-Pilot.
Uses OpenAI GPT-4o for LaTeX resume generation.
"""

import logging
from typing import Optional, Dict, Any
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class ResumeTailor:
    """
    Tailors resumes to job descriptions using OpenAI GPT-4o.
    Generates LaTeX code optimized for professional resumes.
    """
    
    SYSTEM_PROMPT = """You are an expert ATS‑optimized resume writer and LaTeX editor for data/ML/analytics roles. Your only task is to transform the user’s existing LaTeX resume into a new, fully compilable LaTeX file that is precisely tailored to the provided job description, while staying truthful to the user’s real experience.
​

Follow these rules strictly:

Output format

Respond with LaTeX code only, no explanations, comments, or markdown fencing.

The output must be a single, complete, compilable .tex document using the same document class, packages, and general layout as the user’s current resume, unless the user explicitly asks to change the template.
​

Do not invent external files or images; keep everything self‑contained in one .tex file.

Content constraints

Use only information that appears in the user’s current resume or that the user explicitly adds in the conversation.

Do not fabricate degrees, companies, dates, job titles, locations, publications, or certifications.

You may rephrase bullets, reorder content, and adjust emphasis to better match the job description, but the underlying facts must remain accurate.
​

If a required skill or tool is mentioned in the job description but not in the resume, you may:

Emphasize similar/related skills that are actually present.

Slightly reword existing bullets to highlight those relevant skills more clearly, without adding fake experience.

ATS‑friendly optimization

Ensure the resume is ATS‑friendly:

Use a clean structure: standard section headings such as Summary, Experience, Projects, Education, Skills, Certifications.
​

Avoid tables for core content, images, icons, text boxes, or multi‑column layouts that can confuse ATS parsers.

Use simple bullet lists and standard fonts; avoid fancy symbols and custom glyphs.

Incorporate relevant keywords and phrases from the job description naturally into:

The professional summary.

Skills section.

Experience and project bullets.

Keep wording concise and action‑oriented, focusing on impact, metrics, and technologies where possible.
​

Tailoring to the job description

Carefully analyze the job description to identify:

Required and preferred skills, tools, and technologies.

Key responsibilities and outcomes.

Seniority level and focus (e.g., data science vs. ML engineering vs. analytics).
​

Reorder sections and bullets to highlight the most relevant experience at the top of each section.

Emphasize:

Quantifiable achievements (metrics, performance improvements, efficiency gains) when available.

Matching tools and frameworks (Python, SQL, scikit‑learn, PyTorch, cloud, etc.) that the user actually has.

Remove or compress less relevant content if needed to keep the resume concise (typically 1 page for early‑career roles, 2 pages max if justified).

Bullet and language style

Each bullet should:

Start with a strong action verb.

Describe what was done, how it was done (tools/tech), and the impact or outcome where known.

Use clear, professional, and concise language.

Avoid first‑person pronouns.

Avoid generic buzzwords without evidence.

LaTeX‑specific instructions

Preserve the existing LaTeX style: commands, custom macros, and structure (e.g., custom \section, \cventry, etc.), unless they conflict with ATS readability.

Fix any obvious LaTeX issues (unbalanced braces, missing packages, compilation problems) that appear in the user’s source.
​

Ensure proper escaping of LaTeX special characters (%, _, &, #, $, {, }).

If the template includes custom commands for sections or entries, reuse them consistently instead of introducing new styles.

No extra commentary

Do not include any human‑readable explanations, notes, or comments in the output.

Do not wrap the LaTeX code in ``` fences.

The final answer must be only the LaTeX source code of the tailored, ATS‑optimized resume.

At the end of the transformation, the user should be able to compile the output directly as a .tex file to obtain an ATS‑friendly resume that is optimally tailored to the provided job description."""

    USER_PROMPT_TEMPLATE = """Here is my current resume:

---RESUME START---
{resume_text}
---RESUME END---

Here is the job description I'm applying for:

---JOB DESCRIPTION START---
Company: {company}
Title: {job_title}

{job_description}
---JOB DESCRIPTION END---

Please tailor my resume content to this job description. Return ONLY the LaTeX code for the resume body content (sections like Experience, Skills, Education, etc.). Do not include document setup or preamble."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 4000,
        temperature: float = 0.3
    ):
        """
        Initialize the resume tailor.
        
        Args:
            api_key: OpenAI API key.
            model: Model to use (default: gpt-4o).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (lower = more deterministic).
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(f"ResumeTailor initialized with model: {model}")
    
    def tailor_resume(
        self,
        resume_text: str,
        job_description: str,
        company: str,
        job_title: str
    ) -> Dict[str, Any]:
        """
        Generate tailored LaTeX resume content.
        
        Args:
            resume_text: Original resume text.
            job_description: Target job description.
            company: Company name.
            job_title: Job title.
            
        Returns:
            Dictionary with 'latex_content', 'success', and 'error' (if any).
        """
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            resume_text=resume_text,
            company=company,
            job_title=job_title,
            job_description=job_description
        )
        
        try:
            logger.info(f"Generating tailored resume for {company} - {job_title}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            latex_content = response.choices[0].message.content
            
            # Clean up any accidental markdown code blocks
            latex_content = self._clean_latex_response(latex_content)
            
            # Token usage for cost tracking
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            logger.info(f"Resume generated. Tokens used: {usage['total_tokens']}")
            
            return {
                "success": True,
                "latex_content": latex_content,
                "usage": usage,
                "error": None
            }
            
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            return {"success": False, "latex_content": None, "error": f"Connection error: {e}"}
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            return {"success": False, "latex_content": None, "error": f"Rate limit: {e}"}
            
        except openai.APIStatusError as e:
            logger.error(f"OpenAI API error: {e}")
            return {"success": False, "latex_content": None, "error": f"API error: {e}"}
            
        except Exception as e:
            logger.error(f"Unexpected error in resume tailoring: {e}")
            return {"success": False, "latex_content": None, "error": str(e)}
    
    def _clean_latex_response(self, content: str) -> str:
        """
        Clean up the LaTeX response from GPT.
        Removes any markdown code blocks that might have been added.
        """
        if not content:
            return ""
        
        # Remove markdown code blocks
        content = content.strip()
        
        if content.startswith("```latex"):
            content = content[8:]
        elif content.startswith("```tex"):
            content = content[6:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()
    
    def estimate_cost(self, resume_text: str, job_description: str) -> Dict[str, float]:
        """
        Estimate the cost of tailoring a resume.
        
        GPT-4o pricing (as of 2024):
        - Input: $5.00 / 1M tokens
        - Output: $15.00 / 1M tokens
        
        Args:
            resume_text: Resume text.
            job_description: Job description.
            
        Returns:
            Dictionary with estimated token counts and cost.
        """
        # Rough token estimation (1 token ≈ 4 characters)
        system_tokens = len(self.SYSTEM_PROMPT) // 4
        resume_tokens = len(resume_text) // 4
        jd_tokens = len(job_description) // 4
        
        estimated_input = system_tokens + resume_tokens + jd_tokens + 200  # Buffer for template
        estimated_output = min(self.max_tokens, resume_tokens * 1.2)  # Output similar to resume length
        
        # GPT-4o pricing per 1M tokens
        input_cost_per_m = 2.50
        output_cost_per_m = 10.00
        
        estimated_cost = (
            (estimated_input / 1_000_000) * input_cost_per_m +
            (estimated_output / 1_000_000) * output_cost_per_m
        )
        
        return {
            "estimated_input_tokens": int(estimated_input),
            "estimated_output_tokens": int(estimated_output),
            "estimated_cost_usd": round(estimated_cost, 4)
        }


class LaTeXTemplateManager:
    """Manages LaTeX resume templates."""
    
    DEFAULT_PREAMBLE = r"""
\documentclass[11pt,a4paper,sans]{moderncv}
\moderncvstyle{classic}
\moderncvcolor{blue}

\usepackage[utf8]{inputenc}
\usepackage[scale=0.85]{geometry}
\usepackage{hyperref}

% Personal Information
\name{FIRST}{LAST}
\email{email@example.com}
\phone[mobile]{+1-XXX-XXX-XXXX}
\social[linkedin]{linkedin.com/in/username}
\social[github]{github.com/username}

\begin{document}
\makecvtitle
"""

    DEFAULT_CLOSING = r"""
\end{document}
"""
    
    MINIMAL_PREAMBLE = r"""
\documentclass[11pt,letterpaper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=0.75in]{geometry}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{titlesec}

\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}[\titlerule]
\setlist[itemize]{noitemsep,topsep=0pt}

\pagestyle{empty}

\begin{document}
"""

    MINIMAL_CLOSING = r"""
\end{document}
"""
    
    def __init__(self, template_type: str = "minimal"):
        """
        Initialize template manager.
        
        Args:
            template_type: Type of template ('minimal' or 'moderncv').
        """
        self.template_type = template_type
        
        if template_type == "moderncv":
            self.preamble = self.DEFAULT_PREAMBLE
            self.closing = self.DEFAULT_CLOSING
        else:
            self.preamble = self.MINIMAL_PREAMBLE
            self.closing = self.MINIMAL_CLOSING
    
    def wrap_content(self, content: str) -> str:
        """
        Wrap content with preamble and closing.
        
        Args:
            content: LaTeX body content.
            
        Returns:
            Complete LaTeX document.
        """
        return f"{self.preamble}\n{content}\n{self.closing}"
    
    def load_custom_preamble(self, file_path: str) -> bool:
        """
        Load custom preamble from file.
        
        Args:
            file_path: Path to preamble file.
            
        Returns:
            True if loaded successfully.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split at \begin{document}
            if r'\begin{document}' in content:
                parts = content.split(r'\begin{document}')
                self.preamble = parts[0] + r'\begin{document}'
            else:
                self.preamble = content
            
            logger.info(f"Loaded custom preamble from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load custom preamble: {e}")
            return False
