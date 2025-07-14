#!/usr/bin/env python3
"""
DeepSeek Prompter Implementation
Uses deepseek-r1:14b via Ollama for creative prompt generation
"""

import os
import sys
import time
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_prompter import BasePrompter
from ..core import get_logger, get_config
from ..core.exceptions import PromptError

class DeepSeekPrompter(BasePrompter):
    """Generate prompts using deepseek-r1:14b via Ollama"""
    
    def __init__(self):
        """Initialize DeepSeek prompter"""
        super().__init__("DeepSeek")
        # Get Ollama path from config
        config = get_config()
        self.ollama_path = config.system.get('ollama_path', '/usr/local/bin/ollama') if config.system else '/usr/local/bin/ollama'
        self.model_name = "deepseek-r1:14b"
        self._server_started = False
        
    def initialize(self) -> bool:
        """Initialize Ollama and ensure model is available
        
        Returns:
            True if successful
        """
        try:
            # Start Ollama server
            self._start_ollama_server()
            
            # Ensure model is available
            self._ensure_model()
            
            return True
            
        except Exception as e:
            raise PromptError(self.name, e)
            
    def _start_ollama_server(self) -> None:
        """Start Ollama server if not running"""
        self.logger.info("Checking if Ollama server is running...")
        
        # Check if already running
        result = subprocess.run(
            [self.ollama_path, "list"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            self.logger.info("Starting Ollama server...")
            
            # Start server in background
            subprocess.Popen(
                [self.ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server to be ready
            self.logger.info("Waiting for Ollama server to be ready...")
            wait_seconds = 0
            
            while True:
                time.sleep(1)
                wait_seconds += 1
                
                result = subprocess.run(
                    [self.ollama_path, "list"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.logger.info(f"Ollama server ready after {wait_seconds} seconds")
                    self._server_started = True
                    break
                    
                if wait_seconds % 10 == 0:
                    self.logger.info(f"Still waiting... ({wait_seconds}s elapsed)")
                    
                if wait_seconds > 120:
                    raise PromptError(
                        self.name,
                        Exception("Ollama server failed to start after 2 minutes")
                    )
        else:
            self.logger.info("Ollama server is already running")
            
    def _ensure_model(self) -> None:
        """Ensure deepseek model is available"""
        self.logger.info(f"Checking if {self.model_name} is available...")
        
        result = subprocess.run(
            [self.ollama_path, "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if self.model_name not in result.stdout:
            self.logger.info(f"Pulling {self.model_name} model...")
            self.logger.info("This may take a while on first run...")
            
            subprocess.run(
                [self.ollama_path, "pull", self.model_name],
                check=True
            )
            
            self.logger.info(f"{self.model_name} model pulled successfully")
        else:
            self.logger.info(f"{self.model_name} model is already available")
            
    def generate_prompt(
        self,
        theme: Dict[str, Any],
        weather: Dict[str, Any], 
        context: Dict[str, Any],
        history: List[str],
        model_requirements: Dict[str, Any]
    ) -> str:
        """Generate prompt using deepseek
        
        Args:
            theme: Selected theme
            weather: Weather context
            context: Additional context
            history: Previous prompts
            model_requirements: Model-specific requirements
            
        Returns:
            Generated prompt
        """
        self.logger.info(f"Generating prompt for theme: {theme.get('name', 'Unknown')}")
        
        # Format theme instruction
        theme_instruction = self.format_theme_instruction(theme, weather)
        context_desc = self.get_context_description(context)
        
        # Get requirements
        max_words = model_requirements.get('max_words', 65)
        style = model_requirements.get('style', 'photorealistic')
        
        # Build deepseek instruction
        instruction = f"""
Generate a single, richly detailed image prompt for a desktop wallpaper.

{theme_instruction}

Context: {context_desc}

Requirements:
- The prompt MUST be under {max_words} words. Do NOT go over.
- The prompt MUST be the **only** thing in your output. Absolutely no extra text, no commentary, no quotes, no labels, no headers.
- Combine the theme elements creatively and unexpectedly.
- Describe a vivid scene with clear composition: foreground, midground, and background.
- Specify lighting and atmospheric conditions that enhance the theme.
- Include the color palette and artistic style mentioned above.
- Add rich texture and material details.
- Make it {style}, ultra-detailed, and gallery-worthy.

ONLY return the image prompt. No other text.

Image prompt:
""".strip()
        
        self.logger.debug("Sending instruction to deepseek...")
        
        try:
            # Run deepseek
            cmd = [self.ollama_path, "run", self.model_name, instruction]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60  # 1 minute timeout
            )
            
            raw_output = result.stdout
            self.logger.debug(f"Raw output: {raw_output[:200]}...")
            
            # Clean up output
            prompt = self.clean_prompt_output(raw_output)
            
            # Validate basic requirements
            if not prompt or len(prompt.strip()) < 10:
                raise PromptError(
                    self.name,
                    ValueError("Generated prompt is too short or empty")
                )
                
            if not self.validate_prompt(prompt, model_requirements):
                raise PromptError(
                    self.name,
                    ValueError("Generated prompt failed validation")
                )
            
            # Do NOT truncate - the word limit is just a guideline for conciseness
            # FLUX uses T5 which can handle the full prompt regardless of CLIP warnings
                
            self.logger.info(f"Generated prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Generated prompt: {prompt}")
            
            return prompt
            
        except subprocess.CalledProcessError as e:
            raise PromptError(self.name, e)
        except subprocess.TimeoutExpired:
            raise PromptError(self.name, Exception("Prompt generation timed out"))
        finally:
            # Stop model to free VRAM
            self._stop_model()
            
    def _stop_model(self) -> None:
        """Stop the model to free resources"""
        self.logger.info(f"Stopping {self.model_name} to free VRAM...")
        
        try:
            subprocess.run(
                [self.ollama_path, "stop", self.model_name],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("Model stopped successfully")
        except Exception as e:
            self.logger.warning(f"Failed to stop model: {e}")
            
    def validate_prompt(self, prompt: str, requirements: Dict[str, Any]) -> bool:
        """Validate prompt meets requirements
        
        Args:
            prompt: Generated prompt
            requirements: Model requirements
            
        Returns:
            True if valid (after truncation if needed)
        """
        # Check minimum length
        if len(prompt) < 20:
            self.logger.warning("Prompt is too short")
            return False
            
        # Check for common issues
        if prompt.lower().startswith(("here", "this", "i'll", "let me")):
            self.logger.warning("Prompt contains unwanted prefix")
            return False
            
        # Word count limits are just guidelines - no enforcement needed
        return True
        
        
    def cleanup(self) -> None:
        """Clean up resources"""
        # Stop model if loaded
        self._stop_model()
        
        # Note: We don't stop the Ollama server as it may be used by other processes