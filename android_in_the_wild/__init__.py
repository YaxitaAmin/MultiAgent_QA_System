# android_in_the_wild/__init__.py
"""
Android In The Wild Dataset Integration for Multi-Agent QA System

This module provides comprehensive integration with the Android In The Wild dataset
to enhance training, evaluation, and robustness of multi-agent QA systems.

Components:
- dataset_handler: Manages video traces and metadata
- prompt_generator: Generates task prompts from user sessions
- integration_manager: Orchestrates evaluation and comparison
"""

from .dataset_handler import AndroidInTheWildHandler, VideoTrace
from .prompt_generator import TaskPromptGenerator
from .integration_manager import AndroidInTheWildIntegration, EvaluationResult

__version__ = "1.0.0"
__author__ = "Multi-Agent QA System"

__all__ = [
    'AndroidInTheWildHandler',
    'VideoTrace', 
    'TaskPromptGenerator',
    'AndroidInTheWildIntegration',
    'ComparisonResult'
]
