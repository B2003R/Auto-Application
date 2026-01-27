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
    Tailors resumes to job descriptions using OpenAI GPT-4o-mini.
    Generates LaTeX code optimized for ATS and new grad applications.
    """
    
    SYSTEM_PROMPT = r"""You are an expert ATS-optimized resume writer specializing in creating LaTeX resumes for NEW GRADUATES and ENTRY-LEVEL candidates. Your task is to transform the user's complete CV into a tailored, ATS-friendly, single-page LaTeX resume optimized for the specific job description provided.

=== ATS OPTIMIZATION PRINCIPLES (CRITICAL) ===

1. SINGLE-COLUMN LAYOUT ONLY: ATS systems cannot parse multi-column layouts
2. NO GRAPHICS/ICONS/IMAGES: ATS cannot read FontAwesome icons or images
3. STANDARD SECTION HEADINGS: Use "Education", "Experience", "Skills", "Projects"
4. SIMPLE FORMATTING: No tables for content layout, use simple itemize lists
5. MACHINE-READABLE PDF: Use \pdfgentounicode=1 and \input{glyphtounicode}
6. KEYWORD OPTIMIZATION: Naturally incorporate keywords from job description
7. REVERSE CHRONOLOGICAL: Most recent experience/education first

=== NEW GRAD RESUME STRATEGY ===

For candidates with limited work experience:
- EDUCATION FIRST: Lead with education section (degree, GPA if >3.0, relevant coursework)
- EMPHASIZE PROJECTS: Technical projects demonstrate practical skills
- INTERNSHIPS COUNT: Even short internships show real-world experience
- SKILLS SECTION: Prominently display technical skills matching the job
- LEADERSHIP/ACTIVITIES: Include relevant extracurriculars if space allows

=== REQUIRED LATEX TEMPLATE (Based on Jake's Resume - Most ATS-Friendly) ===

Output this EXACT structure:

%-------------------------
% ATS-Optimized Resume in LaTeX
% Based on sb2nov/resume template (6.4k GitHub stars)
% Single-column, no graphics, machine-readable
%-------------------------

\documentclass[letterpaper,11pt]{article}

\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}
\input{glyphtounicode}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Adjust margins for single page
\addtolength{\oddsidemargin}{-0.5in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{-0.5in}
\addtolength{\textheight}{1.0in}

\urlstyle{same}
\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

% Section formatting
\titleformat{\section}{
  \vspace{-4pt}\scshape\raggedright\large
}{}{0em}{}[\color{black}\titlerule \vspace{-5pt}]

% Ensure PDF is machine readable/ATS parsable
\pdfgentounicode=1

%-------------------------
% Custom commands
\newcommand{\resumeItem}[1]{
  \item\small{#1 \vspace{-2pt}}
}

\newcommand{\resumeSubheading}[4]{
  \vspace{-2pt}\item
    \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & #2 \\
      \textit{\small#3} & \textit{\small #4} \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \small#1 & #2 \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeSubItem}[1]{\resumeItem{#1}\vspace{-4pt}}
\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}
\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-5pt}}

%-------------------------------------------
\begin{document}

%----------HEADING----------
\begin{center}
    \textbf{\Huge \scshape FULL NAME} \\ \vspace{1pt}
    \small Phone $|$ \href{mailto:email@email.com}{\underline{email@email.com}} $|$ 
    \href{https://linkedin.com/in/username}{\underline{linkedin.com/in/username}} $|$
    \href{https://github.com/username}{\underline{github.com/username}}
\end{center}

%-----------EDUCATION-----------
\section{Education}
  \resumeSubHeadingListStart
    \resumeSubheading
      {University Name}{City, State}
      {Degree (e.g., Bachelor of Science in Computer Science); GPA: X.XX}{Month Year -- Month Year}
  \resumeSubHeadingListEnd

%-----------TECHNICAL SKILLS-----------
\section{Technical Skills}
 \begin{itemize}[leftmargin=0.15in, label={}]
    \small{\item{
     \textbf{Languages}{: Python, Java, JavaScript, SQL, etc.} \\
     \textbf{Frameworks}{: React, Node.js, Django, etc.} \\
     \textbf{Developer Tools}{: Git, Docker, AWS, VS Code, etc.} \\
     \textbf{Libraries}{: pandas, NumPy, scikit-learn, etc.}
    }}
 \end{itemize}

%-----------EXPERIENCE-----------
\section{Experience}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Job Title}{Start Date -- End Date}
      {Company Name}{City, State}
      \resumeItemListStart
        \resumeItem{Action verb + what you did + result/impact}
        \resumeItem{Another achievement with metrics}
      \resumeItemListEnd
  \resumeSubHeadingListEnd

%-----------PROJECTS-----------
\section{Projects}
    \resumeSubHeadingListStart
      \resumeProjectHeading
          {\textbf{Project Name} $|$ \emph{Technologies Used}}{}
          \resumeItemListStart
            \resumeItem{Description of what you built and achieved}
            \resumeItem{Technical details and impact}
          \resumeItemListEnd
    \resumeSubHeadingListEnd

%-----------CERTIFICATIONS-----------
\section{Certifications}
 \begin{itemize}[leftmargin=0.15in, label={}]
    \small{\item{
     \textbf{Certification Name}{, Issuing Organization (Year)} \\
     \textbf{Another Certification}{, Issuing Organization (Year)}
    }}
 \end{itemize}

%-------------------------------------------
\end{document}

=== END OF TEMPLATE ===

=== STRICT CONTENT RULES ===

1. OUTPUT FORMAT:
   - Return ONLY pure LaTeX code, no markdown fencing (```), no explanations
   - Must be a COMPLETE, COMPILABLE .tex file
   - No external dependencies (no images, no custom cls files)

2. TRUTHFULNESS (CRITICAL):
   - Use ONLY information from the user's CV - NEVER fabricate
   - Do NOT invent companies, job titles, degrees, dates, or skills
   - If the job requires a skill not in the CV, highlight transferable/related skills
   - You may rephrase and reorganize, but never add false information

3. HEADER SECTION (IMPORTANT - Extract ALL contact info from CV):
   - Full name from CV
   - Phone number from CV
   - Email address from CV
   - LinkedIn URL/username from CV (MUST include if present in CV)
   - GitHub URL/username from CV (MUST include if present in CV)
   - Portfolio/Website from CV (include if present)
   - DO NOT leave placeholder text - use ACTUAL values from the CV

4. SECTION ORDER FOR NEW GRADS:
   - Header (name, contact info - NO icons)
   - Education (with GPA if >3.0, relevant coursework)
   - Technical Skills (tailored to job keywords)
   - Experience (internships, part-time, relevant work)
   - Projects (academic and personal projects - NO DATES for projects)
   - Certifications (ALWAYS include at the end if present in CV)

5. FULL PAGE UTILIZATION (CRITICAL):
   - The resume MUST fill the ENTIRE page - not half, not 70%, but FULL page
   - Include 4-5 bullet points per experience/internship role
   - Include 3-4 bullet points per project
   - Add MORE projects if needed to fill the page
   - Include relevant coursework under Education
   - ALWAYS include Certifications section at the END (if present in CV)
   - Expand bullet points with more technical details and metrics
   - If still not full, add more skills categories or elaborate descriptions

6. PROJECT SECTION RULES:
   - DO NOT include dates for projects
   - Format: \resumeProjectHeading{\textbf{Project Name} $|$ \emph{Technologies}}{}
   - Leave the second argument empty (no date)
   - Include 3-4 bullet points per project

7. BULLET POINT REQUIREMENTS:
   - EXPERIENCE: 4-5 detailed bullet points per role (not 2-3!)
   - PROJECTS: 3-4 bullet points per project with technical depth
   - Each bullet should be 1-2 lines long with specific details
   - Start with strong ACTION VERB (Developed, Implemented, Designed, Built, Led, Architected, Optimized)
   - Include WHAT you did + HOW (technologies/methods) + IMPACT (metrics, outcomes)
   - Add quantifiable metrics where possible (%, numbers, scale)
   - Example: "Developed a REST API using Python/FastAPI serving 10K+ requests/day, reducing response time by 40%"

8. KEYWORD OPTIMIZATION:
   - Extract key skills/technologies from job description
   - Mirror exact terminology (e.g., if job says "React.js", use "React.js" not "ReactJS")
   - Include both acronyms and full forms when space allows

9. LATEX ESCAPING:
   - Escape: % → \%, & → \&, $ → \$, # → \#, _ → \_
   - Use -- for date ranges (2023 -- Present)
   - Use $|$ for separators in header

The output must be ONLY the complete LaTeX source code that compiles to a professional, ATS-optimized, FULL single-page resume tailored to the specific job description."""

    USER_PROMPT_TEMPLATE = """=== MY COMPLETE CV/RESUME ===

{resume_text}

=== END OF CV ===

=== TARGET JOB DETAILS ===

Company: {company}
Position: {job_title}

Job Description:
{job_description}

=== END OF JOB DETAILS ===

=== YOUR TASK ===

Generate a complete, ATS-optimized LaTeX resume tailored for this specific position.

CRITICAL REQUIREMENTS:
1. EXTRACT ALL MY CONTACT INFO: Name, Phone, Email, LinkedIn URL, GitHub URL from my CV above
2. FILL THE ENTIRE PAGE: The resume must use the full page, not half or partial
3. MORE BULLET POINTS: 4-5 bullets per experience, 3-4 bullets per project
4. Include ALL my relevant projects and experiences to fill the page
5. Add relevant coursework, certifications, or activities if present in my CV
6. Use the EXACT LaTeX template structure (Jake's Resume/sb2nov template)
7. NO summary/objective section, NO icons or graphics
8. Return ONLY raw LaTeX code - no markdown, no explanations

Output the complete .tex file now:"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        temperature: float = 0.2
    ):
        """
        Initialize the resume tailor.
        
        Args:
            api_key: OpenAI API key.
            model: Model to use (default: gpt-4o-mini - fast and cost-effective).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (lower = more deterministic, better for LaTeX).
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
        
        GPT-4o-mini pricing (as of 2024):
        - Input: $0.15 / 1M tokens
        - Output: $0.60 / 1M tokens
        
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
        
        # GPT-4o-mini pricing per 1M tokens
        input_cost_per_m = 0.15
        output_cost_per_m = 0.60
        
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
    
    # Updated minimal preamble for 1-page resume
    MINIMAL_PREAMBLE = r"""
\documentclass[10pt,letterpaper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=0.5in]{geometry}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{xcolor}

\definecolor{headercolor}{RGB}{0, 51, 102}
\definecolor{linkcolor}{RGB}{0, 102, 204}

\hypersetup{
    colorlinks=true,
    linkcolor=linkcolor,
    urlcolor=linkcolor
}

\titleformat{\section}
    {\normalsize\bfseries\color{headercolor}}
    {}
    {0em}
    {}
    [\titlerule]

\titlespacing{\section}{0pt}{8pt}{4pt}

\setlist[itemize]{noitemsep, topsep=0pt, parsep=0pt, partopsep=0pt, leftmargin=*}

\setlength{\parskip}{0pt}
\setlength{\parindent}{0pt}

\pagestyle{empty}

\newcommand{\role}[4]{
    \textbf{#1} \hfill #2 \\
    \textit{#3} \hfill \textit{#4}
}

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
        
        Note: Since the new prompt generates COMPLETE documents,
        this method checks if the content already has a documentclass.
        If so, it returns the content as-is.
        
        Args:
            content: LaTeX body content or complete document.
            
        Returns:
            Complete LaTeX document.
        """
        # Check if content is already a complete document
        if r'\documentclass' in content:
            return content
        
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
