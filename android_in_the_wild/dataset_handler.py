# android_in_the_wild/dataset_handler.py
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger

@dataclass
class VideoTrace:
    video_path: str
    metadata: Dict[str, Any]
    ui_states: List[Dict[str, Any]]
    user_actions: List[Dict[str, Any]]
    duration: float = 10.0
    frame_count: int = 20

class AndroidInTheWildHandler:
    """Production handler for Android In The Wild dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create sample data if none exists
        if not list(self.dataset_path.glob("*.json")):
            self._create_sample_dataset()
    
    def _create_sample_dataset(self):
        """Create sample dataset for demonstration"""
        sample_data = [
            {
                "filename": "wifi_toggle.json",
                "metadata": {
                    "task_type": "wifi_configuration",
                    "duration": 15.2,
                    "screen_sequence": ["home", "settings", "wifi_settings"],
                    "user_actions": [
                        {"type": "touch", "coordinates": [200, 120], "target": "settings"},
                        {"type": "touch", "coordinates": [180, 240], "target": "wifi_option"},
                        {"type": "touch", "coordinates": [420, 180], "target": "wifi_toggle"}
                    ],
                    "success": True
                }
            },
            {
                "filename": "calculator_test.json", 
                "metadata": {
                    "task_type": "calculator_operation",
                    "duration": 22.8,
                    "screen_sequence": ["home", "calculator"],
                    "user_actions": [
                        {"type": "touch", "coordinates": [150, 320], "target": "calculator_icon"},
                        {"type": "touch", "coordinates": [120, 580], "target": "number_2"},
                        {"type": "touch", "coordinates": [240, 520], "target": "plus_button"},
                        {"type": "touch", "coordinates": [120, 520], "target": "number_3"},
                        {"type": "touch", "coordinates": [240, 640], "target": "equals_button"}
                    ],
                    "success": True
                }
            },
            {
                "filename": "bluetooth_settings.json",
                "metadata": {
                    "task_type": "bluetooth_configuration", 
                    "duration": 18.6,
                    "screen_sequence": ["home", "settings", "bluetooth_settings"],
                    "user_actions": [
                        {"type": "touch", "coordinates": [200, 120], "target": "settings"},
                        {"type": "touch", "coordinates": [180, 300], "target": "bluetooth_option"},
                        {"type": "touch", "coordinates": [420, 160], "target": "bluetooth_toggle"}
                    ],
                    "success": False
                }
            },
            {
                "filename": "storage_check.json",
                "metadata": {
                    "task_type": "storage_management",
                    "duration": 12.4, 
                    "screen_sequence": ["home", "settings", "storage_settings"],
                    "user_actions": [
                        {"type": "touch", "coordinates": [200, 120], "target": "settings"},
                        {"type": "touch", "coordinates": [180, 380], "target": "storage_option"}
                    ],
                    "success": True
                }
            },
            {
                "filename": "alarm_setup.json",
                "metadata": {
                    "task_type": "alarm_management",
                    "duration": 28.3,
                    "screen_sequence": ["home", "clock_app", "alarm_setup"],
                    "user_actions": [
                        {"type": "touch", "coordinates": [120, 680], "target": "clock_icon"},
                        {"type": "touch", "coordinates": [350, 100], "target": "add_alarm"},
                        {"type": "touch", "coordinates": [180, 280], "target": "hour_7"},
                        {"type": "touch", "coordinates": [280, 550], "target": "save_button"}
                    ],
                    "success": True
                }
            }
        ]
        
        for item in sample_data:
            file_path = self.dataset_path / item["filename"]
            with open(file_path, 'w') as f:
                json.dump(item["metadata"], f, indent=2)
        
        logger.info(f"Created {len(sample_data)} sample dataset files")
    
    def load_video_traces(self, num_videos: int = 5) -> List[VideoTrace]:
        """Load video traces from dataset"""
        traces = []
        json_files = list(self.dataset_path.glob("*.json"))[:num_videos]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                # Create video trace
                trace = VideoTrace(
                    video_path=str(json_file.with_suffix('.mp4')),
                    metadata=metadata,
                    ui_states=self._generate_ui_states(metadata),
                    user_actions=metadata.get('user_actions', []),
                    duration=metadata.get('duration', 10.0),
                    frame_count=int(metadata.get('duration', 10.0) * 2)
                )
                traces.append(trace)
                
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(traces)} video traces")
        return traces
    
    def _generate_ui_states(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate UI states from metadata"""
        screens = metadata.get('screen_sequence', ['unknown'])
        ui_states = []
        
        for screen in screens:
            ui_state = {
                "screen_type": screen,
                "elements": self._generate_elements_for_screen(screen),
                "timestamp": len(ui_states) * 0.5
            }
            ui_states.append(ui_state)
        
        return ui_states
    
    def _generate_elements_for_screen(self, screen_type: str) -> List[Dict[str, Any]]:
        """Generate UI elements for screen type"""
        if screen_type == "settings":
            return [
                {"id": "wifi_option", "text": "Wi-Fi", "clickable": True},
                {"id": "bluetooth_option", "text": "Bluetooth", "clickable": True},
                {"id": "storage_option", "text": "Storage", "clickable": True}
            ]
        elif "wifi" in screen_type:
            return [
                {"id": "wifi_toggle", "text": "Wi-Fi", "clickable": True, "checked": random.choice([True, False])}
            ]
        elif "calculator" in screen_type:
            return [
                {"id": "number_2", "text": "2", "clickable": True},
                {"id": "plus_button", "text": "+", "clickable": True},
                {"id": "equals_button", "text": "=", "clickable": True}
            ]
        else:
            return [{"id": "generic_element", "text": "Generic", "clickable": True}]
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        json_files = list(self.dataset_path.glob("*.json"))
        
        if not json_files:
            return {"error": "No dataset files found"}
        
        total_duration = 0
        task_types = {}
        success_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                total_duration += metadata.get('duration', 0)
                task_type = metadata.get('task_type', 'unknown')
                task_types[task_type] = task_types.get(task_type, 0) + 1
                
                if metadata.get('success', False):
                    success_count += 1
                    
            except Exception:
                continue
        
        return {
            "total_videos": len(json_files),
            "total_duration": total_duration,
            "average_duration": total_duration / len(json_files) if json_files else 0,
            "task_type_distribution": task_types,
            "success_rate": success_count / len(json_files) if json_files else 0
        }
