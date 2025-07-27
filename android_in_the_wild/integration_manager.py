# android_in_the_wild/integration_manager.py
"""
Android In The Wild Integration Manager
Integrates with Android usage dataset for comprehensive QA testing
"""

import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class AndroidWildResult:
    """Result from Android In The Wild evaluation"""
    video_id: str
    generated_prompt: str
    success: bool
    accuracy_score: float
    robustness_score: float
    generalization_score: float
    execution_time: float
    ground_truth_trace: Dict[str, Any]
    agent_trace: Dict[str, Any]

class AndroidInTheWildIntegration:
    """Integration with Android In The Wild dataset"""
    
    def __init__(self, qa_manager, dataset_path: str = "./android_in_the_wild_dataset"):
        self.qa_manager = qa_manager
        self.dataset_path = Path(dataset_path)
        self.results = []
        
    def run_evaluation_sync(self, num_videos: int = 5) -> List[AndroidWildResult]:
        """Run evaluation synchronously (for Streamlit compatibility)"""
        try:
            # Create sample dataset if it doesn't exist
            if not self.dataset_path.exists():
                self._create_sample_dataset(num_videos)
            
            # Load video traces
            video_files = list(self.dataset_path.glob("video_*.json"))[:num_videos]
            
            results = []
            for video_file in video_files:
                try:
                    with open(video_file, 'r') as f:
                        video_data = json.load(f)
                    
                    # Generate prompt from video data
                    prompt = self._generate_prompt_from_video(video_data)
                    
                    # Run QA test
                    start_time = time.time()
                    
                    # Create test config
                    test_config = {
                        "goal": prompt,
                        "android_world_task": video_data.get("task_type", "settings_wifi"),
                        "max_steps": 15,
                        "timeout": 120
                    }
                    
                    # Run async test
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    test_result = loop.run_until_complete(
                        self.qa_manager.run_qa_test(test_config)
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Calculate scores
                    scores = self._calculate_scores(video_data, test_result)
                    
                    # Create result
                    result = AndroidWildResult(
                        video_id=video_data["video_id"],
                        generated_prompt=prompt,
                        success=test_result.final_result == "PASS",
                        accuracy_score=scores["accuracy"],
                        robustness_score=scores["robustness"],
                        generalization_score=scores["generalization"],
                        execution_time=execution_time,
                        ground_truth_trace=video_data,
                        agent_trace={
                            "total_steps": len(test_result.actions),
                            "successful_actions": sum(1 for a in test_result.actions if a.success),
                            "final_result": test_result.final_result
                        }
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing video {video_file}: {e}")
                    continue
            
            self.results = results
            return results
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return []
    
    def _create_sample_dataset(self, num_videos: int):
        """Create sample Android In The Wild dataset"""
        self.dataset_path.mkdir(exist_ok=True)
        
        sample_tasks = [
            {
                "task_type": "settings_wifi",
                "description": "Turn on Wi-Fi in settings",
                "expected_actions": ["open_settings", "tap_wifi", "toggle_wifi"]
            },
            {
                "task_type": "settings_wifi", 
                "description": "Navigate to display settings",
                "expected_actions": ["open_settings", "scroll_down", "tap_display"]
            },
            {
                "task_type": "calculator_basic",
                "description": "Calculate 15 + 25",
                "expected_actions": ["open_calculator", "tap_1", "tap_5", "tap_plus", "tap_2", "tap_5", "tap_equals"]
            },
            {
                "task_type": "clock_alarm",
                "description": "Set alarm for 7:00 AM",
                "expected_actions": ["open_clock", "tap_alarm", "tap_add", "set_time_7_00", "save_alarm"]
            },
            {
                "task_type": "contacts_add",
                "description": "Add new contact named John",
                "expected_actions": ["open_contacts", "tap_add", "type_name_john", "save_contact"]
            }
        ]
        
        for i in range(num_videos):
            task = sample_tasks[i % len(sample_tasks)]
            
            video_data = {
                "video_id": f"video_{i+1:03d}",
                "task_type": task["task_type"],
                "description": task["description"],
                "metadata": {
                    "task_type": task["task_type"],
                    "duration": 30 + (i * 5),  # Variable duration
                    "difficulty": "medium",
                    "device": "android_emulator"
                },
                "user_actions": task["expected_actions"],
                "expected_outcome": "task_completed",
                "timestamp": time.time()
            }
            
            video_file = self.dataset_path / f"video_{i+1:03d}.json"
            with open(video_file, 'w') as f:
                json.dump(video_data, f, indent=2)
        
        print(f"Created sample dataset with {num_videos} videos at {self.dataset_path}")
    
    def _generate_prompt_from_video(self, video_data: Dict[str, Any]) -> str:
        """Generate test prompt from video data"""
        description = video_data.get("description", "Perform Android UI task")
        task_type = video_data.get("task_type", "general")
        
        prompts = {
            "settings_wifi": f"Android QA Test: {description} - Verify Wi-Fi functionality",
            "calculator_basic": f"Android QA Test: {description} - Test calculator operations",
            "clock_alarm": f"Android QA Test: {description} - Verify alarm functionality",
            "contacts_add": f"Android QA Test: {description} - Test contact management",
            "general": f"Android QA Test: {description}"
        }
        
        return prompts.get(task_type, prompts["general"])
    
    def _calculate_scores(self, video_data: Dict[str, Any], test_result) -> Dict[str, float]:
        """Calculate performance scores"""
        expected_actions = len(video_data.get("user_actions", []))
        actual_actions = len(test_result.actions)
        successful_actions = sum(1 for a in test_result.actions if a.success)
        
        # Accuracy: How well did the agent perform the task
        accuracy = successful_actions / max(actual_actions, 1) if actual_actions > 0 else 0.0
        
        # Robustness: Consistency of performance
        robustness = min(1.0, successful_actions / max(expected_actions, 1))
        
        # Generalization: Ability to handle variations
        generalization = 0.8 if test_result.final_result == "PASS" else 0.3
        
        return {
            "accuracy": accuracy,
            "robustness": robustness,
            "generalization": generalization
        }
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not self.results:
            return {"error": "No results available"}
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        
        # Overall metrics
        avg_accuracy = sum(r.accuracy_score for r in self.results) / total_tests
        avg_robustness = sum(r.robustness_score for r in self.results) / total_tests
        avg_generalization = sum(r.generalization_score for r in self.results) / total_tests
        
        # Composite score
        composite_score = (avg_accuracy + avg_robustness + avg_generalization) / 3
        
        # Task type performance
        task_performance = {}
        for result in self.results:
            task_type = result.ground_truth_trace.get("task_type", "unknown")
            if task_type not in task_performance:
                task_performance[task_type] = {
                    "count": 0,
                    "successes": 0,
                    "total_accuracy": 0.0,
                    "total_robustness": 0.0,
                    "total_generalization": 0.0
                }
            
            perf = task_performance[task_type]
            perf["count"] += 1
            if result.success:
                perf["successes"] += 1
            perf["total_accuracy"] += result.accuracy_score
            perf["total_robustness"] += result.robustness_score
            perf["total_generalization"] += result.generalization_score
        
        # Calculate averages
        for task_type, perf in task_performance.items():
            count = perf["count"]
            perf["success_rate"] = perf["successes"] / count
            perf["avg_accuracy"] = perf["total_accuracy"] / count
            perf["avg_robustness"] = perf["total_robustness"] / count
            perf["avg_generalization"] = perf["total_generalization"] / count
        
        return {
            "evaluation_summary": {
                "total_videos": total_tests,
                "successful_tests": successful_tests,
                "overall_success_rate": successful_tests / total_tests,
                "average_accuracy": avg_accuracy,
                "average_robustness": avg_robustness,
                "average_generalization": avg_generalization,
                "composite_score": composite_score
            },
            "task_type_performance": task_performance,
            "recommendations": self._generate_recommendations(avg_accuracy, avg_robustness, avg_generalization)
        }
    
    def _generate_recommendations(self, accuracy: float, robustness: float, generalization: float) -> List[Dict[str, str]]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if accuracy < 0.7:
            recommendations.append({
                "type": "accuracy",
                "priority": "high",
                "message": "Action execution accuracy is below threshold. Consider improving UI element detection.",
                "specific_actions": [
                    "Review UI parsing logic",
                    "Improve element identification",
                    "Add more training data"
                ]
            })
        
        if robustness < 0.6:
            recommendations.append({
                "type": "robustness", 
                "priority": "medium",
                "message": "System robustness needs improvement for consistent performance.",
                "specific_actions": [
                    "Add error recovery mechanisms",
                    "Improve retry logic",
                    "Handle edge cases better"
                ]
            })
        
        if generalization < 0.5:
            recommendations.append({
                "type": "generalization",
                "priority": "medium", 
                "message": "Generalization capability is limited. Consider expanding training scenarios.",
                "specific_actions": [
                    "Test on more diverse tasks",
                    "Improve task understanding",
                    "Add contextual awareness"
                ]
            })
        
        return recommendations
