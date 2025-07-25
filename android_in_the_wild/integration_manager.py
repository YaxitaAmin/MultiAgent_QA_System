# android_in_the_wild/integration_manager.py
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
from .prompt_generator import TaskPromptGenerator

@dataclass
class ComparisonResult:
    video_id: str
    video_path: str
    generated_prompt: str
    agent_trace: Dict[str, Any]
    ground_truth_trace: VideoTrace
    accuracy_score: float
    robustness_score: float
    generalization_score: float
    execution_time: float
    success: bool

# android_in_the_wild/integration_manager.py - CRITICAL FIX
class AndroidInTheWildIntegration:
    def __init__(self, qa_manager, dataset_path: str):
        self.qa_manager = qa_manager
        self.dataset_path = Path(dataset_path)
        self.dataset_handler = AndroidInTheWildHandler(dataset_path)
        
    def run_evaluation_sync(self, num_videos: int) -> List[EvaluationResult]:
        """FIXED: Ensure proper task execution"""
        results = []
        
        try:
            # Load video traces with error handling
            traces = self.dataset_handler.load_video_traces(num_videos)
            
            if not traces:
                print("[INTEGRATION] ERROR: No traces loaded from dataset")
                return []
            
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
                    
                    # Create evaluation result
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
                    print(f"[INTEGRATION] Result: {'SUCCESS' if eval_result.success else 'FAILED'}")
                    
                except Exception as e:
                    print(f"[INTEGRATION] ERROR processing trace {trace.video_id}: {e}")
                    # Add failed result
                    results.append(self._create_failed_result(trace, str(e)))
                    
        except Exception as e:
            print(f"[INTEGRATION] CRITICAL ERROR: {e}")
            return []
        
        return results
    async def run_evaluation(self, num_videos: int = 5) -> List[ComparisonResult]:
        """Run complete evaluation on video traces"""
        logger.info(f"Starting evaluation with {num_videos} videos")
        
        # Load video traces
        video_traces = self.dataset_handler.load_video_traces(num_videos)
        
        if not video_traces:
            logger.error("No video traces loaded")
            return []
        
        results = []
        
        for i, trace in enumerate(video_traces):
            try:
                logger.info(f"Processing video {i+1}/{num_videos}")
                
                # Generate task prompt
                prompt = await self.prompt_generator.generate_task_prompt(trace)
                
                # Execute with QA system
                start_time = time.time()
                if hasattr(self.qa_manager, 'execute_qa_task_sync'):
                    agent_result = self.qa_manager.execute_qa_task_sync(
                        task_description=prompt,
                        max_steps=max(10, len(trace.user_actions) * 2),
                        timeout=300
                    )
                else:
                    agent_result = await self.qa_manager.execute_qa_task(
                        task_description=prompt,
                        max_steps=max(10, len(trace.user_actions) * 2),
                        timeout=300
                    )
                
                execution_time = time.time() - start_time
                
                # Calculate scores
                scores = self._calculate_scores(agent_result, trace)
                
                result = ComparisonResult(
                    video_id=f"video_{i+1}",
                    video_path=trace.video_path,
                    generated_prompt=prompt,
                    agent_trace=agent_result,
                    ground_truth_trace=trace,
                    accuracy_score=scores['accuracy'],
                    robustness_score=scores['robustness'], 
                    generalization_score=scores['generalization'],
                    execution_time=execution_time,
                    success=agent_result.get('success', False)
                )
                
                results.append(result)
                self.results.append(result)
                
                logger.info(f"Video {i+1} completed - Success: {result.success}")
                
            except Exception as e:
                logger.error(f"Failed to process video {i+1}: {e}")
                # Create error result
                error_result = ComparisonResult(
                    video_id=f"video_{i+1}_error",
                    video_path=trace.video_path if 'trace' in locals() else "unknown",
                    generated_prompt="Error generating prompt",
                    agent_trace={"success": False, "error": str(e)},
                    ground_truth_trace=trace if 'trace' in locals() else None,
                    accuracy_score=0.0,
                    robustness_score=0.0,
                    generalization_score=0.0,
                    execution_time=0.0,
                    success=False
                )
                results.append(error_result)
        
        logger.info(f"Evaluation completed. Processed {len(results)} videos")
        return results
    
    def _calculate_scores(self, agent_result: Dict[str, Any], ground_truth: VideoTrace) -> Dict[str, float]:
        """Calculate accuracy, robustness, and generalization scores"""
        
        # Accuracy: task completion and step similarity
        accuracy = 0.1  # Base score
        if agent_result.get('success', False):
            accuracy = 0.8
            
            # Bonus for step efficiency
            agent_steps = agent_result.get('total_steps', 0)
            expected_steps = len(ground_truth.user_actions)
            if expected_steps > 0 and agent_steps > 0:
                step_ratio = min(agent_steps, expected_steps) / max(agent_steps, expected_steps)
                accuracy += step_ratio * 0.2
        
        # Robustness: error handling and consistency
        robustness = 0.5  # Base score
        if 'error' not in agent_result:
            robustness += 0.2
        
        verification_data = agent_result.get('verification', {})
        confidence = verification_data.get('average_confidence', 0.0)
        robustness += confidence * 0.3
        
        # Generalization: adaptation to UI patterns
        generalization = 0.6  # Base score
        
        # Time efficiency
        agent_time = agent_result.get('total_time', 0)  
        expected_time = ground_truth.duration
        if expected_time > 0 and agent_time > 0:
            if agent_time < expected_time * 3:  # Within 3x expected
                time_ratio = expected_time / agent_time
                generalization += min(0.2, time_ratio * 0.2)
        
        # Confidence bonus
        generalization += confidence * 0.2
        
        return {
            'accuracy': min(1.0, max(0.0, accuracy)),
            'robustness': min(1.0, max(0.0, robustness)),
            'generalization': min(1.0, max(0.0, generalization))
        }
    
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
            task_type = result.ground_truth_trace.metadata.get('task_type', 'unknown') if result.ground_truth_trace else 'unknown'
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
        
        # Task-specific recommendations
        task_performance = {}
        for result in self.results:
            task_type = result.ground_truth_trace.metadata.get('task_type', 'unknown') if result.ground_truth_trace else 'unknown'
            if task_type not in task_performance:
                task_performance[task_type] = []
            task_performance[task_type].append(result.success)
        
        for task_type, successes in task_performance.items():
            task_success_rate = sum(successes) / len(successes)
            if task_success_rate < 0.5:
                recommendations.append({
                    "type": "task_specific",
                    "priority": "medium",
                    "message": f"Poor performance on {task_type} tasks ({task_success_rate:.1%} success rate).",
                    "specific_actions": [
                        f"Analyze {task_type} failure patterns",
                        f"Improve {task_type}-specific action sequences",
                        f"Add more {task_type} test scenarios"
                    ]
                })
        
        return recommendations
