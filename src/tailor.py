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
    
    SYSTEM_PROMPT = """You are an expert LaTeX resume writer and ATS optimization specialist.

Your task is to tailor the provided resume to match the job description while maintaining truthfulness.

CRITICAL RULES:
1. Return ONLY raw LaTeX code - no markdown, no explanations, no code blocks
2. ESCAPE ALL special LaTeX characters: %, &, $, #, _, {, }, ~, ^, \\
   - Use \\% for percent signs
   - Use \\& for ampersands  
   - Use \\$ for dollar signs
   - Use \\# for hash symbols
   - Use \\_ for underscores
3. Preserve the resume structure and formatting
4. Emphasize relevant skills and experiences that match the JD
5. Use strong action verbs and quantified achievements
6. Optimize bullet points for ATS keyword matching
7. Keep the resume to 1 page maximum
8. Do NOT fabricate experiences or skills not present in the original resume
9. Do NOT include any preamble, documentclass, or begin{document} - just the content

FORMAT GUIDELINES:
- Use \\textbf{} for section headers
- Use \\item for bullet points within itemize environments
- Maintain consistent spacing with \\vspace{}
- Use \\href{}{} for links (ensure URLs are escaped)

TAILORING STRATEGY:
1. Identify key skills and requirements from the JD
2. Reorder bullet points to prioritize relevant experience
3. Adjust wording to mirror JD language (naturally)
4. Highlight transferable skills that match requirements
5. Quantify achievements where possible (numbers, percentages, scale)"""

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
        input_cost_per_m = 5.00
        output_cost_per_m = 15.00
        
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
