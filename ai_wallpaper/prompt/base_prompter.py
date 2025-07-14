#!/usr/bin/env python3
"""
Base Prompter Abstract Class
All prompt generation strategies inherit from this base
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..core import get_logger, get_config

class BasePrompter(ABC):
    """Abstract base class for prompt generation strategies"""
    
    def __init__(self, name: str = "BasePrompter"):
        """Initialize base prompter
        
        Args:
            name: Prompter name for logging
        """
        self.name = name
        self.logger = get_logger(model=f"Prompt-{name}")
        self.config = get_config()
        
    @abstractmethod
    def generate_prompt(
        self, 
        theme: Dict[str, Any],
        weather: Dict[str, Any],
        context: Dict[str, Any],
        history: List[str],
        model_requirements: Dict[str, Any]
    ) -> str:
        """Generate an image prompt
        
        Args:
            theme: Selected theme with elements, styles, colors
            weather: Current weather context
            context: Additional context (date, season, etc.)
            history: Previous prompts for uniqueness
            model_requirements: Model-specific requirements
            
        Returns:
            Generated prompt string
        """
        pass
        
    @abstractmethod
    def validate_prompt(self, prompt: str, requirements: Dict[str, Any]) -> bool:
        """Validate prompt meets requirements
        
        Args:
            prompt: Generated prompt
            requirements: Model requirements
            
        Returns:
            True if valid
        """
        pass
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize prompter (e.g., start services, load models)
        
        Returns:
            True if initialization successful
        """
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
        
    def format_theme_instruction(self, theme: Dict, weather: Dict) -> str:
        """Format theme into instruction text
        
        Args:
            theme: Theme dictionary
            weather: Weather context
            
        Returns:
            Formatted instruction
        """
        elements = ', '.join(theme.get('elements', []))
        styles = ', '.join(theme.get('styles', []))
        colors = ', '.join(theme.get('colors', []))
        
        instruction = f"""
THEME: {theme.get('name', 'Unknown')}
CORE ELEMENTS: {elements}
ARTISTIC STYLE: {styles}
COLOR PALETTE: {colors}
WEATHER INTEGRATION: Current {weather.get('condition', 'clear')} conditions with {weather.get('mood', 'neutral')} atmosphere

Create a desktop wallpaper that combines these elements in an unexpected, creative way.
The image should tell a visual story incorporating the weather naturally into the theme.
Focus on photorealistic quality with rich detail and perfect composition.
"""
        
        return instruction.strip()
        
    def get_context_description(self, context: Dict) -> str:
        """Get context description for prompt
        
        Args:
            context: Context dictionary
            
        Returns:
            Context description
        """
        return f"It's {context.get('day_of_week', 'today')} in {context.get('season', 'the current season')}"
        
    def clean_prompt_output(self, raw_output: str) -> str:
        """Clean up raw output from generation
        
        Args:
            raw_output: Raw generated text
            
        Returns:
            Cleaned prompt
        """
        prompt = raw_output.strip()
        
        # Remove thinking tags if present
        if "<think>" in prompt and "</think>" in prompt:
            end_tag = prompt.find("</think>")
            if end_tag != -1:
                prompt = prompt[end_tag + len("</think>"):].strip()
                
        # Remove code blocks
        if "```" in prompt:
            parts = prompt.split("```")
            if len(parts) >= 2:
                prompt = parts[1].strip()
                
        # Remove quotes
        if prompt.startswith('"') and prompt.endswith('"'):
            prompt = prompt[1:-1].strip()
            
        # Clean up formatting
        prompt = prompt.replace('\n', ' ').replace('  ', ' ').strip()
        prompt = prompt.replace('**', '')
        
        # Remove unwanted prefixes
        unwanted_prefixes = [
            "Here's", "Here is", "This is", "I'll create", "I'll generate",
            "Let me", "Sure", "Certainly", "Of course", "Image prompt:",
            "Image description:", "Description:", "**Image Description:**",
            "Image Description:"
        ]
        
        for prefix in unwanted_prefixes:
            if prompt.startswith(prefix):
                for punct in ['.', ':', '\n']:
                    idx = prompt.find(punct)
                    if idx > 0:
                        prompt = prompt[idx+1:].strip()
                        break
                        
        return prompt
        
    def load_prompt_history(self) -> List[str]:
        """Load previous prompts from history
        
        Returns:
            List of previous prompts
        """
        # Get history file path from system config or paths config
        if self.config.system and 'generation' in self.config.system:
            history_path = self.config.system['generation'].get('prompt_history_file', 
                                                               self.config.paths.get('prompt_history', 'prompt_history.txt'))
        else:
            history_path = self.config.paths.get('prompt_history', 'prompt_history.txt')
        history_file = Path(history_path)
        
        if not history_file.exists():
            self.logger.info("No prompt history found, starting fresh")
            return []
            
        try:
            with open(history_file, 'r') as f:
                history = [line.strip() for line in f.readlines() if line.strip()]
                
            self.logger.info(f"Loaded {len(history)} previous prompts")
            return history
            
        except Exception as e:
            self.logger.warning(f"Failed to load prompt history: {e}")
            return []
            
    def save_prompt(self, prompt: str, metadata: Optional[Dict] = None) -> None:
        """Save prompt to history
        
        Args:
            prompt: Generated prompt
            metadata: Optional metadata (seed, timestamp, etc.)
        """
        # Get history file path from system config or paths config
        if self.config.system and 'generation' in self.config.system:
            history_path = self.config.system['generation'].get('prompt_history_file', 
                                                               self.config.paths.get('prompt_history', 'prompt_history.txt'))
        else:
            history_path = self.config.paths.get('prompt_history', 'prompt_history.txt')
        history_file = Path(history_path)
        
        try:
            # Ensure directory exists
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file needs newline
            needs_newline = False
            if history_file.exists() and history_file.stat().st_size > 0:
                with open(history_file, 'rb') as f:
                    f.seek(-1, 2)
                    last_char = f.read(1)
                    needs_newline = last_char != b'\n'
                    
            # Append prompt
            with open(history_file, 'a') as f:
                if needs_newline:
                    f.write('\n')
                f.write(prompt + '\n')
                
            self.logger.info("Saved prompt to history")
            
            # Also save to last run file
            last_run_file = Path(self.config.paths.get('last_run_file', 'last_run.txt'))
            with open(last_run_file, 'w') as f:
                from datetime import datetime
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Prompt: {prompt}\n")
                if metadata:
                    for key, value in metadata.items():
                        f.write(f"{key.capitalize()}: {value}\n")
                f.write("Status: Success\n")
                
        except Exception as e:
            self.logger.error(f"Failed to save prompt: {e}")