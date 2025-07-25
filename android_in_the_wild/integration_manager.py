# android_in_the_wild/integration_manager.py - COMPLETE WORKING VERSION
import asyncio
import time
import json
import random
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from env_manager import MultiAgentQAManager
from .dataset_handler import AndroidInTheWildHandler, VideoTrace

@dataclass
class EvaluationResult:
    """Results from evaluating a single video trace"""
    video_id: str
    generated_prompt: str
    ground_truth_trace: VideoTrace
    agent_trace: Dict[str, Any]
    success: bool
    execution_time: float
    accuracy_score: float
    robustness_score: float
    generalization_score: float

class AndroidInTheWildIntegration:
    """WORKING Integration manager for Android In The Wild dataset evaluation"""
    
    def __init__(self, qa_manager: MultiAgentQAManager, dataset_path: str):
        self.qa_manager = qa_manager
        self.dataset_path = Path(dataset_path)
        self.dataset_handler = AndroidInTheWildHandler(dataset_path)
        self.results: List[EvaluationResult] = []
        
        print(f"[INTEGRATION] Initialized with dataset path: {dataset_path}")
    
    def run_evaluation_sync(self, num_videos: int) -> List[EvaluationResult]:
        """WORKING: Synchronous evaluation for Streamlit compatibility"""
        results = []
        
        try:
            print(f"[INTEGRATION] Loading {num_videos} video traces...")
            
            # Load video traces with error handling
            traces = self.dataset_handler.load_video_traces(num_videos)
            
            if not traces:
                print("[INTEGRATION] ERROR: No traces loaded from dataset")
                return []
            
            print(f"[INTEGRATION] Successfully loaded {len(traces)} traces")
            
            for i, trace in enumerate(traces):
                print(f"[INTEGRATION] Processing trace {i+1}/{len(traces)}: {trace.video_id}")
                
                try:
                    # Generate task prompt properly
                    task_prompt = self._generate_task_prompt(trace)
                    print(f"[INTEGRATION] Generated prompt: {task_prompt}")
                    
                    # CRITICAL: Execute with proper timeout and steps
                    start_time = time.time()
                    
                    # Use your QA manager's sync method
                    qa_result = self.qa_manager.execute_qa_task_sync(
                        task_prompt, 
                        max_steps=15,  # Increased steps
                        timeout=60     # Proper timeout
                    )
                    
                    execution_time = time.time() - start_time
                    print(f"[INTEGRATION] QA execution took {execution_time:.2f}s")
                    
                    # Create evaluation result - FIXED WITH PROPER CONSTRUCTOR
                    eval_result = EvaluationResult(
                        video_id=trace.video_id,
                        generated_prompt=task_prompt,
                        ground_truth_trace=trace,
                        agent_trace=qa_result,
                        success=qa_result.get('success', False),
                        execution_time=execution_time,
                        accuracy_score=self._calculate_accuracy_score(trace, qa_result),
                        robustness_score=self._calculate_robustness_score(qa_result),
                        generalization_score=self._calculate_generalization_score(trace, qa_result)
                    )
                    
                    results.append(eval_result)
                    self.results.append(eval_result)  # Store for report generation
                    
                    print(f"[INTEGRATION] Result: {'SUCCESS' if eval_result.success else 'FAILED'}")
                    
                except Exception as e:
                    print(f"[INTEGRATION] ERROR processing trace {trace.video_id}: {e}")
                    # Add failed result
                    failed_result = self._create_failed_result(trace, str(e))
                    results.append(failed_result)
                    self.results.append(failed_result)
                    
        except Exception as e:
            print(f"[INTEGRATION] CRITICAL ERROR: {e}")
            return []
        
        return results
    
    def _generate_task_prompt(self, trace: VideoTrace) -> str:
        """Generate task prompt from video trace"""
        
        task_type = trace.metadata.get('task_type', 'unknown')
        
        # Map task types to realistic prompts
        prompt_mapping = {
            'wifi_configuration': "Test turning Wi-Fi on and off",
            'bluetooth_configuration': "Navigate to Bluetooth settings and verify state", 
            'calculator_operation': "Open Calculator app and perform basic calculation",
            'storage_management': "Navigate to Storage settings and check usage",
            'alarm_management': "Open Clock app and manage alarms"
        }
        
        return prompt_mapping.get(task_type, "Open Settings and navigate to system preferences")
    
    def _create_failed_result(self, trace: VideoTrace, error_message: str) -> EvaluationResult:
        """Create a failed evaluation result"""
        return EvaluationResult(
            video_id=trace.video_id,
            generated_prompt="Error generating prompt",
            ground_truth_trace=trace,
            agent_trace={"success": False, "error": error_message, "total_time": 0.0, "total_steps": 0},
            success=False,
            execution_time=0.0,
            accuracy_score=0.1,
            robustness_score=0.3,
            generalization_score=0.4
        )
    
    def _calculate_accuracy_score(self, ground_truth: VideoTrace, agent_result: Dict) -> float:
        """Calculate realistic accuracy score"""
        
        if not agent_result.get('success', False):
            return 0.1  # Minimum score for failed executions
        
        # Compare expected vs actual steps
        expected_steps = len(ground_truth.user_actions)
        actual_steps = agent_result.get('total_steps', 0)
        
        if expected_steps == 0:
            return 0.5  # Default for missing ground truth
        
        # Calculate step similarity (rough approximation)
        step_similarity = min(1.0, actual_steps / max(expected_steps, 1))
        
        # Factor in completion rate
        plan = agent_result.get('plan', {})
        completion_rate = plan.get('completed_subgoals', 0) / max(plan.get('subgoals_count', 1), 1)
        
        # Combined accuracy score
        accuracy = (step_similarity * 0.4) + (completion_rate * 0.6)
        return max(0.1, min(1.0, accuracy))

    def _calculate_robustness_score(self, agent_result: Dict) -> float:
        """Calculate robustness based on error handling"""
        
        if not agent_result.get('success', False):
            return 0.3  # Some credit for attempting execution
        
        # Check execution consistency
        execution_data = agent_result.get('execution', {})
        success_rate = execution_data.get('success_rate', 0.0)
        
        # Check verification consistency  
        verification_data = agent_result.get('verification', {})
        confidence = verification_data.get('average_confidence', 0.0)
        
        # Combined robustness score
        robustness = (success_rate * 0.6) + (confidence * 0.4)
        return max(0.3, min(1.0, robustness))

    def _calculate_generalization_score(self, ground_truth: VideoTrace, agent_result: Dict) -> float:
        """Calculate generalization score"""
        
        if not agent_result.get('success', False):
            return 0.4  # Base score for failed tasks
        
        # Factor in task complexity
        task_complexity = min(1.0, len(ground_truth.user_actions) / 10.0)
        
        # Factor in execution efficiency
        total_time = agent_result.get('total_time', 1.0)
        efficiency = max(0.1, min(1.0, 30.0 / total_time))  # Optimal around 30s
        
        # Combined generalization score
        generalization = (task_complexity * 0.5) + (efficiency * 0.5)
        return max(0.4, min(1.0, generalization))
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not self.results:
            return {"error": "No evaluation results available"}
        
        # Calculate aggregate metrics
        total_results = len(self.results)
        successful_results = [r for r in self.results if r.success]
        success_rate = len(successful_results) / total_results
        
        accuracy_scores = [r.accuracy_score for r in self.results]
        robustness_scores = [r.robustness_score for r in self.results]
        generalization_scores = [r.generalization_score for r in self.results]
        
        avg_accuracy = np.mean(accuracy_scores)
        avg_robustness = np.mean(robustness_scores)
        avg_generalization = np.mean(generalization_scores)
        
        composite_score = (avg_accuracy + avg_robustness + avg_generalization) / 3
        
        # Task type performance
        task_performance = {}
        for result in self.results:
            task_type = result.ground_truth_trace.metadata.get('task_type', 'unknown')
            if task_type not in task_performance:
                task_performance[task_type] = []
            task_performance[task_type].append({
                'accuracy': result.accuracy_score,
                'robustness': result.robustness_score, 
                'generalization': result.generalization_score,
                'success': result.success
            })
        
        # Aggregate by task type
        task_summary = {}
        for task_type, performances in task_performance.items():
            task_summary[task_type] = {
                'count': len(performances),
                'success_rate': sum(1 for p in performances if p['success']) / len(performances),
                'avg_accuracy': np.mean([p['accuracy'] for p in performances]),
                'avg_robustness': np.mean([p['robustness'] for p in performances]),
                'avg_generalization': np.mean([p['generalization'] for p in performances])
            }
        
        return {
            "evaluation_summary": {
                "total_videos_processed": total_results,
                "overall_success_rate": success_rate,
                "average_accuracy": avg_accuracy,
                "average_robustness": avg_robustness,
                "average_generalization": avg_generalization,
                "composite_score": composite_score
            },
            "task_type_performance": task_summary,
            "detailed_results": [
                {
                    "video_id": r.video_id,
                    "generated_prompt": r.generated_prompt,
                    "success": r.success,
                    "accuracy": r.accuracy_score,
                    "robustness": r.robustness_score,
                    "generalization": r.generalization_score,
                    "execution_time": r.execution_time
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations(success_rate, avg_accuracy, avg_robustness, avg_generalization)
        }
    
    def _generate_recommendations(self, success_rate: float, avg_accuracy: float, avg_robustness: float, avg_generalization: float) -> List[Dict[str, Any]]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if avg_accuracy < 0.7:
            recommendations.append({
                "type": "accuracy",
                "priority": "high",
                "message": f"Low accuracy score ({avg_accuracy:.2f}). Improve planner agent task decomposition and executor precision.",
                "specific_actions": [
                    "Review planner subgoal generation logic",
                    "Enhance executor UI element detection", 
                    "Improve action grounding accuracy"
                ]
            })
        
        if avg_robustness < 0.7:
            recommendations.append({
                "type": "robustness", 
                "priority": "high",
                "message": f"Low robustness score ({avg_robustness:.2f}). Strengthen error handling and recovery mechanisms.",
                "specific_actions": [
                    "Implement better error detection",
                    "Add more recovery strategies",
                    "Improve verification confidence calibration"
                ]
            })
        
        if avg_generalization < 0.7:
            recommendations.append({
                "type": "generalization",
                "priority": "medium", 
                "message": f"Low generalization score ({avg_generalization:.2f}). Enhance system adaptability to diverse UI patterns.",
                "specific_actions": [
                    "Train on more diverse UI layouts",
                    "Improve time estimation algorithms",
                    "Enhance UI element recognition robustness"
                ]
            })
        
        if success_rate < 0.6:
            recommendations.append({
                "type": "success_rate",
                "priority": "critical",
                "message": f"Low overall success rate ({success_rate:.1%}). Review fundamental system architecture.",
                "specific_actions": [
                    "Analyze failure patterns across all agents",
                    "Review agent coordination mechanisms", 
                    "Consider timeout and resource management"
                ]
            })
        
        return recommendations
