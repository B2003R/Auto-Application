"""
PDF compiler module for Job Application Co-Pilot.
Uses pdflatex subprocess for LaTeX to PDF conversion.
"""

import os
import re
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PDFCompiler:
    """
    Compiles LaTeX documents to PDF using pdflatex.
    Handles temporary files, error reporting, and output management.
    """
    
    def __init__(
        self,
        pdflatex_path: str = "pdflatex",
        temp_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the PDF compiler.
        
        Args:
            pdflatex_path: Path to pdflatex executable.
            temp_dir: Directory for temporary files.
            output_dir: Directory for generated PDFs.
        """
        self.pdflatex_path = pdflatex_path
        self.temp_dir = temp_dir or Path("temp")
        self.output_dir = output_dir or Path("data/generated")
        
        # Ensure directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify pdflatex is available
        self._verify_pdflatex()
    
    def _verify_pdflatex(self) -> bool:
        """Verify that pdflatex is installed and accessible."""
        try:
            result = subprocess.run(
                [self.pdflatex_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                logger.info(f"pdflatex found: {version_line}")
                return True
            else:
                logger.warning(f"pdflatex check failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error(
                "pdflatex not found. Please install a LaTeX distribution "
                "(e.g., MiKTeX, TeX Live) and ensure pdflatex is in PATH"
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("pdflatex version check timed out")
            return False
    
    def compile(
        self,
        latex_content: str,
        output_filename: Optional[str] = None,
        company: Optional[str] = None,
        role: Optional[str] = None,
        job_id: Optional[str] = None,
        clean_temp: bool = True
    ) -> Dict[str, Any]:
        """
        Compile LaTeX content to PDF.
        
        Args:
            latex_content: Complete LaTeX document content.
            output_filename: Custom output filename (without extension).
            company: Company name for auto-generated filename.
            role: Role/title for auto-generated filename.
            job_id: Job ID for auto-generated filename.
            clean_temp: Whether to clean up temporary files after compilation.
            
        Returns:
            Dictionary with 'success', 'pdf_path', 'log', and 'error'.
        """
        # Generate output filename if not provided
        if not output_filename:
            output_filename = self._generate_filename(company, role, job_id)
        
        # Sanitize filename
        output_filename = self._sanitize_filename(output_filename)
        
        # Create temporary .tex file
        temp_tex = self.temp_dir / f"{output_filename}.tex"
        temp_pdf = self.temp_dir / f"{output_filename}.pdf"
        final_pdf = self.output_dir / f"{output_filename}.pdf"
        
        try:
            # Write LaTeX content to temp file
            temp_tex.write_text(latex_content, encoding='utf-8')
            logger.info(f"Wrote LaTeX to {temp_tex}")
            
            # Run pdflatex (twice for references)
            success, log_output = self._run_pdflatex(temp_tex)
            
            if not success:
                # Try to extract error from log
                error_msg = self._extract_latex_error(log_output)
                return {
                    "success": False,
                    "pdf_path": None,
                    "log": log_output,
                    "error": error_msg or "LaTeX compilation failed"
                }
            
            # Check if PDF was created
            if not temp_pdf.exists():
                return {
                    "success": False,
                    "pdf_path": None,
                    "log": log_output,
                    "error": "PDF file was not created"
                }
            
            # Move PDF to output directory
            shutil.move(str(temp_pdf), str(final_pdf))
            logger.info(f"PDF compiled successfully: {final_pdf}")
            
            return {
                "success": True,
                "pdf_path": str(final_pdf),
                "log": log_output,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return {
                "success": False,
                "pdf_path": None,
                "log": "",
                "error": str(e)
            }
            
        finally:
            if clean_temp:
                self._cleanup_temp_files(output_filename)
    
    def _run_pdflatex(self, tex_file: Path) -> Tuple[bool, str]:
        """
        Run pdflatex on the given .tex file.
        
        Args:
            tex_file: Path to the .tex file.
            
        Returns:
            Tuple of (success, log_output).
        """
        cmd = [
            self.pdflatex_path,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory", str(self.temp_dir),
            str(tex_file)
        ]
        
        log_output = ""
        
        try:
            # Run twice for cross-references
            for run in range(2):
                logger.debug(f"pdflatex run {run + 1}/2")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(self.temp_dir)
                )
                
                log_output = result.stdout + "\n" + result.stderr
                
                if result.returncode != 0:
                    logger.warning(f"pdflatex returned code {result.returncode}")
                    return False, log_output
            
            return True, log_output
            
        except subprocess.TimeoutExpired:
            logger.error("pdflatex timed out")
            return False, "Compilation timed out after 60 seconds"
        except Exception as e:
            logger.error(f"pdflatex error: {e}")
            return False, str(e)
    
    def _extract_latex_error(self, log_output: str) -> Optional[str]:
        """
        Extract meaningful error message from LaTeX log.
        
        Args:
            log_output: Full pdflatex log output.
            
        Returns:
            Extracted error message or None.
        """
        # Common LaTeX error patterns
        error_patterns = [
            r'! LaTeX Error: (.+)',
            r'! (.+)',
            r'l\.\d+ (.+)',
            r'Missing .+ inserted',
            r'Undefined control sequence',
            r'File .+ not found',
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, log_output)
            if match:
                return match.group(0)
        
        return None
    
    def _generate_filename(
        self,
        company: Optional[str],
        role: Optional[str],
        job_id: Optional[str]
    ) -> str:
        """Generate filename from job details."""
        parts = []
        
        if company:
            parts.append(company)
        if role:
            parts.append(role)
        if job_id:
            parts.append(job_id)
        
        if not parts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"resume_{timestamp}"
        
        return "_".join(parts)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for file system compatibility."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Remove multiple underscores
        while '__' in filename:
            filename = filename.replace('__', '_')
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename.strip('_')
    
    def _cleanup_temp_files(self, base_name: str):
        """Clean up temporary LaTeX files."""
        extensions = ['.tex', '.aux', '.log', '.out', '.toc', '.nav', '.snm', '.fls', '.fdb_latexmk']
        
        for ext in extensions:
            temp_file = self.temp_dir / f"{base_name}{ext}"
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.debug(f"Failed to delete {temp_file}: {e}")
    
    def compile_from_file(self, tex_file: Path, output_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile an existing .tex file to PDF.
        
        Args:
            tex_file: Path to existing .tex file.
            output_filename: Custom output filename.
            
        Returns:
            Compilation result dictionary.
        """
        if not tex_file.exists():
            return {
                "success": False,
                "pdf_path": None,
                "log": "",
                "error": f"File not found: {tex_file}"
            }
        
        content = tex_file.read_text(encoding='utf-8')
        
        if not output_filename:
            output_filename = tex_file.stem
        
        return self.compile(content, output_filename)


class LaTeXValidator:
    """Validates LaTeX content before compilation."""
    
    REQUIRED_PACKAGES = []  # Can be customized
    
    COMMON_ISSUES = [
        (r'(?<!\\)[%]', 'Unescaped % character'),
        (r'(?<!\\)[&]', 'Unescaped & character'),
        (r'(?<!\\)[$](?!\$)', 'Unescaped $ character'),
        (r'(?<!\\)[#]', 'Unescaped # character'),
        (r'(?<!\\)[_](?!{)', 'Potentially unescaped _ character'),
    ]
    
    @classmethod
    def validate(cls, content: str) -> Dict[str, Any]:
        """
        Validate LaTeX content for common issues.
        
        Args:
            content: LaTeX content to validate.
            
        Returns:
            Dictionary with 'valid', 'warnings', and 'errors'.
        """
        warnings = []
        errors = []
        
        # Check for document structure
        if r'\documentclass' not in content:
            errors.append("Missing \\documentclass")
        
        if r'\begin{document}' not in content:
            errors.append("Missing \\begin{document}")
        
        if r'\end{document}' not in content:
            errors.append("Missing \\end{document}")
        
        # Check for common issues (these are warnings, not errors)
        for pattern, message in cls.COMMON_ISSUES:
            matches = re.findall(pattern, content)
            if matches:
                warnings.append(f"{message} (found {len(matches)} instance(s))")
        
        # Check for balanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
        
        # Check for balanced environments
        begin_envs = re.findall(r'\\begin\{(\w+)\}', content)
        end_envs = re.findall(r'\\end\{(\w+)\}', content)
        
        if sorted(begin_envs) != sorted(end_envs):
            errors.append("Unbalanced environments detected")
        
        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors
        }
    
    @classmethod
    def auto_fix_special_chars(cls, content: str) -> str:
        """
        Attempt to fix common special character issues.
        Use with caution - may affect intended LaTeX commands.
        
        Args:
            content: LaTeX content to fix.
            
        Returns:
            Fixed content.
        """
        # Only fix in text content, not in commands
        # This is a simplified approach - full parsing would be better
        
        # Fix unescaped & in obvious text contexts
        content = re.sub(r'(?<=[a-zA-Z]) & (?=[a-zA-Z])', r' \\& ', content)
        
        # Fix percentage numbers like "50%" -> "50\\%"
        content = re.sub(r'(\d+)%', r'\1\\%', content)
        
        return content
