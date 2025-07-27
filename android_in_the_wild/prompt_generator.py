# android_in_the_wild/prompt_generator.py
import random
import json
from typing import Dict, List
from core.llm_interface import LLMRequest
from .dataset_handler import VideoTrace
from loguru import logger

class TaskPromptGenerator:
    """Generate task prompts from video traces"""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.templates = {
            "wifi_configuration": [
                "Test turning Wi-Fi on and off",
                "Toggle Wi-Fi in Settings",
                "Enable and disable Wi-Fi connection"
            ],
            "bluetooth_configuration": [
                "Test Bluetooth on/off toggle", 
                "Enable and disable Bluetooth",
                "Navigate to Bluetooth settings and toggle"
            ],
            "calculator_operation": [
                "Open Calculator app and perform basic calculation",
                "Use Calculator to add two numbers", 
                "Test basic arithmetic in Calculator"
            ],
            "storage_management": [
                "Navigate to Storage settings and check usage",
                "Check device storage details",
                "Open Storage settings and view usage"
            ],
            "alarm_management": [
                "Set a new alarm for 7:30 AM",
                "Create an alarm in Clock app",
                "Test alarm creation functionality"
            ]
        }
    
    def _generate_task_prompt_from_trace(self, trace: VideoTrace) -> str:
        """Generate realistic task prompts that match your QA system capabilities 👇"""

        actions = trace.user_actions
        metadata = trace.metadata
        task_type = metadata.get('task_type', 'unknown').lower()

        if task_type == 'wifi_configuration' or any('wifi' in str(action).lower() for action in actions):
            return "Test turning Wi-Fi on and off 📶"

        elif task_type == 'bluetooth_configuration' or any('bluetooth' in str(action).lower() for action in actions):
            return "Navigate to Bluetooth settings and verify state 🔵"

        elif task_type == 'alarm_management' or any('alarm' in str(action).lower() for action in actions):
            return "Open Clock app and manage alarms ⏰"

        elif task_type == 'calculator_operation' or any('calculator' in str(action).lower() for action in actions):
            return "Open Calculator app and perform basic calculation ➕➖"

        elif task_type == 'storage_management' or any('storage' in str(action).lower() for action in actions):
            return "Navigate to Storage settings and check usage 💾"

        else:
            return "Open Settings and navigate to system preferences ⚙️"