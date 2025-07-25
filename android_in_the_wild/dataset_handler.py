# android_in_the_wild/dataset_handler.py - COMPLETE WORKING VERSION
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class UserAction:
    """Represents a single user action from the dataset"""
    action_type: str
    timestamp: float
    coordinates: Optional[List[int]] = None
    element_id: Optional[str] = None
    text_input: Optional[str] = None

@dataclass
class VideoTrace:
    """Represents a complete video trace from the dataset"""
    video_id: str
    video_path: str
    duration: float
    user_actions: List[UserAction]
    metadata: Dict[str, Any]

class AndroidInTheWildHandler:
    """WORKING Handler for Android In The Wild dataset with sample data generation"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.traces: List[VideoTrace] = []
        
        print(f"[DATASET] Initializing handler for path: {dataset_path}")
        
        # Create directory if it doesn't exist
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
    def load_video_traces(self, num_videos: int = 5) -> List[VideoTrace]:
        """Load video traces from dataset or create sample data"""
        
        print(f"[DATASET] Loading {num_videos} video traces...")
        
        # Check if real dataset exists
        if self._has_real_dataset():
            print(f"[DATASET] Found real dataset, loading...")
            return self._load_real_dataset(num_videos)
        else:
            print(f"[DATASET] No real dataset found, creating sample data...")
            return self._create_sample_dataset(num_videos)
    
    def _has_real_dataset(self) -> bool:
        """Check if real dataset exists"""
        # Look for common dataset files
        dataset_files = list(self.dataset_path.glob("*.json"))
        video_files = list(self.dataset_path.glob("*.mp4"))
        
        return len(dataset_files) > 0 or len(video_files) > 0
    
    def _load_real_dataset(self, num_videos: int) -> List[VideoTrace]:
        """Load real dataset if available"""
        try:
            # Try to load JSON metadata files
            json_files = list(self.dataset_path.glob("*.json"))[:num_videos]
            
            traces = []
            for i, json_file in enumerate(json_files):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Parse the real dataset format
                    trace = self._parse_real_trace_data(data, f"video_{i+1}")
                    traces.append(trace)
                    
                except Exception as e:
                    print(f"[DATASET] Failed to load {json_file}: {e}")
                    continue
            
            print(f"[DATASET] Loaded {len(traces)} real traces")
            return traces
            
        except Exception as e:
            print(f"[DATASET] Error loading real dataset: {e}")
            return self._create_sample_dataset(num_videos)
    
    def _parse_real_trace_data(self, data: Dict[str, Any], video_id: str) -> VideoTrace:
        """Parse real dataset format into VideoTrace"""
        
        # Extract user actions from real dataset
        actions = []
        for action_data in data.get('actions', []):
            action = UserAction(
                action_type=action_data.get('type', 'touch'),
                timestamp=action_data.get('timestamp', 0),
                coordinates=action_data.get('coordinates'),
                element_id=action_data.get('element_id'),
                text_input=action_data.get('text')
            )
            actions.append(action)
        
        return VideoTrace(
            video_id=video_id,
            video_path=str(self.dataset_path / f"{video_id}.mp4"),
            duration=data.get('duration', 30.0),
            user_actions=actions,
            metadata=data.get('metadata', {})
        )
    
    def _create_sample_dataset(self, num_videos: int) -> List[VideoTrace]:
        """Create realistic sample dataset for testing"""
        
        sample_traces = []
        
        # Define sample task types
        task_templates = [
            {
                "task_type": "wifi_configuration",
                "description": "User toggling Wi-Fi settings",
                "actions": [
                    {"type": "touch", "coordinates": [200, 120], "element_id": "settings_icon"},
                    {"type": "touch", "coordinates": [180, 240], "element_id": "wifi_option"},
                    {"type": "touch", "coordinates": [420, 180], "element_id": "wifi_toggle"}
                ]
            },
            {
                "task_type": "bluetooth_configuration", 
                "description": "User managing Bluetooth settings",
                "actions": [
                    {"type": "touch", "coordinates": [200, 120], "element_id": "settings_icon"},
                    {"type": "touch", "coordinates": [180, 300], "element_id": "bluetooth_option"},
                    {"type": "touch", "coordinates": [420, 160], "element_id": "bluetooth_toggle"}
                ]
            },
            {
                "task_type": "calculator_operation",
                "description": "User performing calculation",
                "actions": [
                    {"type": "touch", "coordinates": [150, 320], "element_id": "calculator_icon"},
                    {"type": "touch", "coordinates": [150, 580], "element_id": "digit_5"},
                    {"type": "touch", "coordinates": [200, 530], "element_id": "plus_button"},
                    {"type": "touch", "coordinates": [100, 580], "element_id": "digit_3"}
                ]
            },
            {
                "task_type": "storage_management",
                "description": "User checking storage usage",
                "actions": [
                    {"type": "touch", "coordinates": [200, 120], "element_id": "settings_icon"},
                    {"type": "scroll", "coordinates": [200, 400]},
                    {"type": "touch", "coordinates": [180, 380], "element_id": "storage_option"}
                ]
            },
            {
                "task_type": "alarm_management",
                "description": "User setting alarm",
                "actions": [
                    {"type": "touch", "coordinates": [120, 680], "element_id": "clock_icon"},
                    {"type": "touch", "coordinates": [350, 100], "element_id": "add_alarm"},
                    {"type": "touch", "coordinates": [200, 400], "element_id": "time_picker"}
                ]
            }
        ]
        
        for i in range(num_videos):
            # Select random template
            template = random.choice(task_templates)
            
            # Create actions with some variation
            actions = []
            base_time = time.time()
            
            for j, action_template in enumerate(template["actions"]):
                # Add some randomness to coordinates and timing
                coords = action_template.get("coordinates", [200, 400])
                varied_coords = [
                    coords[0] + random.randint(-10, 10),
                    coords[1] + random.randint(-10, 10)
                ]
                
                action = UserAction(
                    action_type=action_template["type"],
                    timestamp=base_time + (j * 2.0) + random.uniform(0, 1),
                    coordinates=varied_coords,
                    element_id=action_template.get("element_id"),
                    text_input=action_template.get("text")
                )
                actions.append(action)
            
            # Create trace
            trace = VideoTrace(
                video_id=f"sample_video_{i+1}",
                video_path=str(self.dataset_path / f"sample_video_{i+1}.mp4"),
                duration=len(actions) * 2.5 + random.uniform(5, 15),
                user_actions=actions,
                metadata={
                    "task_type": template["task_type"],
                    "description": template["description"],
                    "synthetic": True,
                    "complexity": random.choice(["simple", "medium", "complex"]),
                    "device_type": random.choice(["phone", "tablet"]),
                    "app_version": f"1.{random.randint(0, 5)}.{random.randint(0, 9)}"
                }
            )
            
            sample_traces.append(trace)
        
        self.traces = sample_traces
        
        # Save sample dataset for future use
        self._save_sample_dataset(sample_traces)
        
        print(f"[DATASET] Created {len(sample_traces)} sample traces")
        return sample_traces
    
    def _save_sample_dataset(self, traces: List[VideoTrace]):
        """Save sample dataset to files"""
        try:
            for trace in traces:
                # Save trace metadata
                trace_file = self.dataset_path / f"{trace.video_id}.json"
                
                trace_data = {
                    "video_id": trace.video_id,
                    "duration": trace.duration,
                    "metadata": trace.metadata,
                    "actions": [
                        {
                            "type": action.action_type,
                            "timestamp": action.timestamp,
                            "coordinates": action.coordinates,
                            "element_id": action.element_id,
                            "text": action.text_input
                        }
                        for action in trace.user_actions
                    ]
                }
                
                with open(trace_file, 'w') as f:
                    json.dump(trace_data, f, indent=2)
            
            print(f"[DATASET] Saved {len(traces)} sample traces to {self.dataset_path}")
            
        except Exception as e:
            print(f"[DATASET] Failed to save sample dataset: {e}")
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        
        if not self.traces:
            return {
                "total_videos": 0,
                "total_duration": 0.0,
                "average_duration": 0.0,
                "task_type_distribution": {},
                "success_rate": 0.0
            }
        
        total_videos = len(self.traces)
        total_duration = sum(trace.duration for trace in self.traces)
        average_duration = total_duration / total_videos if total_videos > 0 else 0
        
        # Task type distribution
        task_types = {}
        for trace in self.traces:
            task_type = trace.metadata.get('task_type', 'unknown')
            task_types[task_type] = task_types.get(task_type, 0) + 1
        
        return {
            "total_videos": total_videos,
            "total_duration": total_duration,
            "average_duration": average_duration,
            "task_type_distribution": task_types,
            "success_rate": 1.0,  # Sample data assumes success
            "action_types": self._get_action_type_stats(),
            "complexity_distribution": self._get_complexity_stats()
        }
    
    def _get_action_type_stats(self) -> Dict[str, int]:
        """Get statistics on action types"""
        action_counts = {}
        
        for trace in self.traces:
            for action in trace.user_actions:
                action_type = action.action_type
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        return action_counts
    
    def _get_complexity_stats(self) -> Dict[str, int]:
        """Get statistics on task complexity"""
        complexity_counts = {}
        
        for trace in self.traces:
            complexity = trace.metadata.get('complexity', 'unknown')
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return complexity_counts
