# agents/supervisor_agent.py
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

from core.llm_interface import LLMRequest, CostEfficientLLMInterface
from core.logger import QALogger, TestEpisode
from .planner_agent import TestPlan
from .executor_agent import ExecutionResult
from .verifier_agent import VerificationResult

@dataclass
class SupervisorAnalysis:
    episode_id: str
    overall_success: bool
    execution_quality: float
    planning_quality: float
    verification_quality: float
    issues_identified: List[str]
    suggestions: List[str]
    improvement_areas: List[str]
    coverage_analysis: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    timestamp: float

class SupervisorAgent:
    """Agent responsible for analyzing full test episodes and providing improvements"""
    
    def __init__(self, 
                 llm_interface: CostEfficientLLMInterface,
                 logger: QALogger, 
                 config: Dict[str, Any]):
        self.llm = llm_interface
        self.logger = logger
        self.config = config
        self.analysis_history: List[SupervisorAnalysis] = []
        
    async def analyze_episode(self, 
                             episode: TestEpisode,
                             test_plan: TestPlan,
                             execution_results: List[ExecutionResult],
                             verification_results: List[VerificationResult],
                             screenshots: List[str] = None) -> SupervisorAnalysis:
        """Analyze complete test episode and provide feedback"""
        start_time = time.time()
        
        logger.info(f"Analyzing episode {episode.episode_id}")
        
        try:
            # Perform multi-faceted analysis
            execution_analysis = self._analyze_execution_quality(execution_results)
            planning_analysis = self._analyze_planning_quality(test_plan, execution_results)
            verification_analysis = self._analyze_verification_quality(verification_results)
            coverage_analysis = self._analyze_test_coverage(test_plan, execution_results)
            performance_analysis = self._analyze_performance(episode, execution_results, verification_results)
            
            # Generate comprehensive analysis using LLM
            llm_analysis = await self._generate_llm_analysis(
                episode, test_plan, execution_results, verification_results, screenshots
            )
            
            # Combine all analyses
            analysis = SupervisorAnalysis(
                episode_id=episode.episode_id,
                overall_success=episode.success,
                execution_quality=execution_analysis["quality_score"],
                planning_quality=planning_analysis["quality_score"],
                verification_quality=verification_analysis["quality_score"],
                issues_identified=self._combine_issues(execution_analysis, planning_analysis, verification_analysis, llm_analysis),
                suggestions=self._combine_suggestions(execution_analysis, planning_analysis, verification_analysis, llm_analysis),
                improvement_areas=llm_analysis.get("improvement_areas", []),
                coverage_analysis=coverage_analysis,
                performance_metrics=performance_analysis,
                timestamp=time.time()
            )
            
            self.analysis_history.append(analysis)
            
            execution_time = time.time() - start_time
            
            self.logger.log_agent_action(
                agent_type="supervisor",
                action_type="analyze_episode",
                input_data={
                    "episode_id": episode.episode_id,
                    "episode_success": episode.success,
                    "total_steps": len(execution_results)
                },
                output_data={
                    "overall_quality": (analysis.execution_quality + analysis.planning_quality + analysis.verification_quality) / 3,
                    "issues_count": len(analysis.issues_identified),
                    "suggestions_count": len(analysis.suggestions)
                },
                success=True,
                execution_time=execution_time
            )
            
            logger.info(f"Episode analysis complete. Overall quality: {analysis.execution_quality:.2f}")
            return analysis
            
        except Exception as e:
            execution_time = time.time() - start_time
            # agents/supervisor_agent.py (continued)
            error_message = str(e)
            
            # Create fallback analysis
            analysis = SupervisorAnalysis(
                episode_id=episode.episode_id,
                overall_success=episode.success,
                execution_quality=0.5,
                planning_quality=0.5,
                verification_quality=0.5,
                issues_identified=[f"Analysis error: {error_message}"],
                suggestions=["Retry analysis", "Check system logs"],
                improvement_areas=["Error handling"],
                coverage_analysis={"error": "analysis_failed"},
                performance_metrics={"error": "analysis_failed"},
                timestamp=time.time()
            )
            
            self.logger.log_agent_action(
                agent_type="supervisor",
                action_type="analyze_episode",
                input_data={"episode_id": episode.episode_id},
                output_data={},
                success=False,
                execution_time=execution_time,
                error_message=error_message
            )
            
            logger.error(f"Episode analysis failed: {error_message}")
            return analysis
    
    def _analyze_execution_quality(self, execution_results: List[ExecutionResult]) -> Dict[str, Any]:
        """Analyze execution quality based on success rates and timing"""
        if not execution_results:
            return {"quality_score": 0.0, "issues": [], "suggestions": []}
        
        success_rate = sum(1 for r in execution_results if r.success) / len(execution_results)
        avg_execution_time = sum(r.execution_time for r in execution_results) / len(execution_results)
        
        issues = []
        suggestions = []
        
        if success_rate < 0.8:
            issues.append(f"Low execution success rate: {success_rate:.2f}")
            suggestions.append("Improve action grounding and element detection")
        
        if avg_execution_time > 5.0:
            issues.append(f"Slow execution: {avg_execution_time:.2f}s average")
            suggestions.append("Optimize action execution and reduce delays")
        
        # Check for repeated failures
        failure_patterns = {}
        for result in execution_results:
            if not result.success and result.error_message:
                failure_patterns[result.error_message] = failure_patterns.get(result.error_message, 0) + 1
        
        for error, count in failure_patterns.items():
            if count > 1:
                issues.append(f"Repeated failure: {error} ({count} times)")
                suggestions.append(f"Add specific handling for: {error}")
        
        quality_score = min(1.0, success_rate + (0.2 if avg_execution_time < 3.0 else 0.0))
        
        return {
            "quality_score": quality_score,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _analyze_planning_quality(self, test_plan: TestPlan, execution_results: List[ExecutionResult]) -> Dict[str, Any]:
        """Analyze planning quality based on plan structure and execution alignment"""
        issues = []
        suggestions = []
        
        # Check plan completeness
        if len(test_plan.subgoals) < 3:
            issues.append("Plan too simple, may miss edge cases")
            suggestions.append("Add more detailed sub-steps and error handling")
        
        if len(test_plan.subgoals) > 10:
            issues.append("Plan too complex, may be over-engineered")
            suggestions.append("Simplify plan and combine related steps")
        
        # Check contingency planning
        if not test_plan.contingencies:
            issues.append("No contingency plans defined")
            suggestions.append("Add error handling and recovery strategies")
        
        # Analyze plan vs execution alignment
        executed_steps = len(execution_results)
        planned_steps = len(test_plan.subgoals)
        
        if executed_steps > planned_steps * 1.5:
            issues.append("Execution deviated significantly from plan")
            suggestions.append("Improve plan accuracy and adaptability")
        
        # Check dependencies
        dependency_issues = 0
        for subgoal in test_plan.subgoals:
            if subgoal.dependencies:
                for dep_id in subgoal.dependencies:
                    if not any(sg.id == dep_id for sg in test_plan.subgoals):
                        dependency_issues += 1
        
        if dependency_issues > 0:
            issues.append(f"Invalid dependencies in plan: {dependency_issues}")
            suggestions.append("Validate dependency chains in planning")
        
        quality_score = max(0.0, 1.0 - (len(issues) * 0.2))
        
        return {
            "quality_score": quality_score,
            "plan_complexity": len(test_plan.subgoals),
            "contingency_count": len(test_plan.contingencies),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _analyze_verification_quality(self, verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """Analyze verification quality based on accuracy and coverage"""
        if not verification_results:
            return {"quality_score": 0.0, "issues": ["No verifications performed"], "suggestions": ["Add verification steps"]}
        
        pass_rate = sum(1 for r in verification_results if r.passed) / len(verification_results)
        avg_confidence = sum(r.confidence for r in verification_results) / len(verification_results)
        
        issues = []
        suggestions = []
        
        if avg_confidence < 0.7:
            issues.append(f"Low verification confidence: {avg_confidence:.2f}")
            suggestions.append("Improve verification criteria and methods")
        
        # Check verification coverage
        verification_types = set(r.verification_type for r in verification_results)
        if len(verification_types) < 2:
            issues.append("Limited verification coverage")
            suggestions.append("Use multiple verification strategies")
        
        # Check for consistent failures
        if pass_rate < 0.5:
            issues.append(f"High verification failure rate: {1-pass_rate:.2f}")
            suggestions.append("Review test expectations and implementation")
        
        quality_score = (pass_rate + avg_confidence) / 2
        
        return {
            "quality_score": quality_score,
            "pass_rate": pass_rate,
            "avg_confidence": avg_confidence,
            "verification_types": list(verification_types),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _analyze_test_coverage(self, test_plan: TestPlan, execution_results: List[ExecutionResult]) -> Dict[str, Any]:
        """Analyze test coverage across different aspects"""
        coverage = {
            "ui_elements_tested": set(),
            "action_types_used": set(),
            "screens_visited": set(),
            "error_scenarios_handled": 0
        }
        
        for result in execution_results:
            action = result.action_performed
            if action.get("action_type"):
                coverage["action_types_used"].add(action["action_type"])
            
            if action.get("target_text"):
                coverage["ui_elements_tested"].add(action["target_text"])
            
            if not result.success:
                coverage["error_scenarios_handled"] += 1
        
        return {
            "elements_tested": len(coverage["ui_elements_tested"]),
            "action_types": list(coverage["action_types_used"]),
            "error_scenarios": coverage["error_scenarios_handled"],
            "coverage_score": min(1.0, len(coverage["action_types_used"]) / 4)  # Normalize to 4 action types
        }
    
    def _analyze_performance(self, episode: TestEpisode, execution_results: List[ExecutionResult], verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        total_time = (episode.end_time or episode.start_time) - episode.start_time
        
        return {
            "total_episode_time": total_time,
            "steps_per_minute": len(execution_results) / (total_time / 60) if total_time > 0 else 0,
            "avg_step_time": sum(r.execution_time for r in execution_results) / len(execution_results) if execution_results else 0,
            "verification_overhead": sum(r.timestamp for r in verification_results) / len(verification_results) if verification_results else 0
        }
    
    async def _generate_llm_analysis(self, episode: TestEpisode, test_plan: TestPlan, execution_results: List[ExecutionResult], verification_results: List[VerificationResult], screenshots: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis using LLM"""
        try:
            analysis_prompt = self._build_analysis_prompt(episode, test_plan, execution_results, verification_results)
            
            request = LLMRequest(
                prompt=analysis_prompt,
                model=self.config.get("model", "gemini-1.5-pro"),
                temperature=self.config.get("temperature", 0.2),
                max_tokens=1000,
                system_prompt=self._get_analysis_system_prompt()
            )
            
            response = await self.llm.generate(request)
            return self._parse_analysis_response(response.content)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {
                "improvement_areas": ["Error in LLM analysis"],
                "issues": [f"LLM analysis error: {str(e)}"],
                "suggestions": ["Use heuristic analysis only"]
            }
    
    def _build_analysis_prompt(self, episode: TestEpisode, test_plan: TestPlan, execution_results: List[ExecutionResult], verification_results: List[VerificationResult]) -> str:
        """Build prompt for comprehensive episode analysis"""
        # Summarize key metrics
        execution_summary = {
            "total_steps": len(execution_results),
            "successful_steps": sum(1 for r in execution_results if r.success),
            "average_execution_time": sum(r.execution_time for r in execution_results) / len(execution_results) if execution_results else 0
        }
        
        verification_summary = {
            "total_verifications": len(verification_results),
            "passed_verifications": sum(1 for r in verification_results if r.passed),
            "average_confidence": sum(r.confidence for r in verification_results) / len(verification_results) if verification_results else 0
        }
        
        prompt = f"""
Analyze this complete Android UI test episode:

Episode: {episode.episode_id}
Task: {episode.task_description}
Overall Success: {episode.success}
Duration: {(episode.end_time or episode.start_time) - episode.start_time:.2f}s

Test Plan:
- Goal: {test_plan.goal}
- Subgoals: {len(test_plan.subgoals)}
- Contingencies: {len(test_plan.contingencies)}

Execution Summary:
{json.dumps(execution_summary, indent=2)}

Verification Summary:
{json.dumps(verification_summary, indent=2)}

Recent Issues:
{self._get_recent_issues(execution_results, verification_results)}

Please provide a comprehensive analysis focusing on:
1. What worked well in this test execution
2. What areas need improvement
3. Specific suggestions for each agent (Planner, Executor, Verifier)
4. Overall system robustness assessment

Respond with JSON:
{{
    "improvement_areas": ["area1", "area2"],
    "agent_feedback": {{
        "planner": ["suggestion1", "suggestion2"],
        "executor": ["suggestion1", "suggestion2"],
        "verifier": ["suggestion1", "suggestion2"]
    }},
    "system_robustness": 0.0-1.0,
    "key_insights": ["insight1", "insight2"]
}}
"""
        return prompt
    
    def _get_recent_issues(self, execution_results: List[ExecutionResult], verification_results: List[VerificationResult]) -> List[str]:
        """Get recent issues from execution and verification"""
        issues = []
        
        # Last 3 execution issues
        for result in execution_results[-3:]:
            if not result.success and result.error_message:
                issues.append(f"Execution: {result.error_message}")
        
        # Last 3 verification issues
        for result in verification_results[-3:]:
            if not result.passed and result.issues:
                issues.extend([f"Verification: {issue}" for issue in result.issues[:2]])
        
        return issues[:5]  # Limit to 5 most recent
    
    def _get_analysis_system_prompt(self) -> str:
        """Get system prompt for episode analysis"""
        return """
You are an expert QA system analyst reviewing multi-agent Android UI test executions. 

Your analysis should be:
1. Constructive and actionable
2. Focused on system improvement
3. Balanced between praise and criticism
4. Technically specific where possible
5. Consider real-world QA constraints

Provide insights that would help improve the overall testing system's effectiveness, reliability, and coverage.
"""
    
    def _parse_analysis_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        try:
            response_content = response_content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3]
            elif response_content.startswith("```"):
                response_content = response_content[3:-3]
            
            return json.loads(response_content)
            
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return {
                "improvement_areas": ["Response parsing"],
                "agent_feedback": {"planner": [], "executor": [], "verifier": []},
                "system_robustness": 0.5,
                "key_insights": ["Analysis parsing failed"]
            }
    
    def _combine_issues(self, *analyses) -> List[str]:
        """Combine issues from multiple analyses"""
        all_issues = []
        for analysis in analyses:
            if isinstance(analysis, dict) and "issues" in analysis:
                all_issues.extend(analysis["issues"])
        return list(set(all_issues))  # Remove duplicates
    
    def _combine_suggestions(self, *analyses) -> List[str]:
        """Combine suggestions from multiple analyses"""
        all_suggestions = []
        for analysis in analyses:
            if isinstance(analysis, dict) and "suggestions" in analysis:
                all_suggestions.extend(analysis["suggestions"])
        return list(set(all_suggestions))  # Remove duplicates
    
    def get_improvement_report(self) -> Dict[str, Any]:
        """Generate comprehensive improvement report from all analyses"""
        if not self.analysis_history:
            return {"message": "No analyses available"}
        
        # Aggregate metrics
        avg_execution_quality = sum(a.execution_quality for a in self.analysis_history) / len(self.analysis_history)
        avg_planning_quality = sum(a.planning_quality for a in self.analysis_history) / len(self.analysis_history)
        avg_verification_quality = sum(a.verification_quality for a in self.analysis_history) / len(self.analysis_history)
        
        # Common issues and suggestions
        all_issues = []
        all_suggestions = []
        for analysis in self.analysis_history:
            all_issues.extend(analysis.issues_identified)
            all_suggestions.extend(analysis.suggestions)
        
        issue_frequency = {}
        for issue in all_issues:
            issue_frequency[issue] = issue_frequency.get(issue, 0) + 1
        
        top_issues = sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "summary": {
                "total_episodes_analyzed": len(self.analysis_history),
                "avg_execution_quality": avg_execution_quality,
                "avg_planning_quality": avg_planning_quality,
                "avg_verification_quality": avg_verification_quality,
                "overall_system_quality": (avg_execution_quality + avg_planning_quality + avg_verification_quality) / 3
            },
            "top_issues": [{"issue": issue, "frequency": freq} for issue, freq in top_issues],
            "improvement_trends": self._analyze_improvement_trends(),
            "recommendations": self._generate_system_recommendations()
        }
    
    def _analyze_improvement_trends(self) -> Dict[str, Any]:
        """Analyze improvement trends over time"""
        if len(self.analysis_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_analyses = self.analysis_history[-3:]
        older_analyses = self.analysis_history[:-3] if len(self.analysis_history) > 3 else []
        
        if not older_analyses:
            return {"trend": "insufficient_data"}
        
        recent_avg = sum(a.execution_quality + a.planning_quality + a.verification_quality for a in recent_analyses) / (len(recent_analyses) * 3)
        older_avg = sum(a.execution_quality + a.planning_quality + a.verification_quality for a in older_analyses) / (len(older_analyses) * 3)
        
        improvement = recent_avg - older_avg
        
        return {
            "trend": "improving" if improvement > 0.1 else "declining" if improvement < -0.1 else "stable",
            "improvement_rate": improvement,
            "recent_quality": recent_avg,
            "historical_quality": older_avg
        }
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        if not self.analysis_history:
            return ["Collect more test data for analysis"]
        
        avg_execution = sum(a.execution_quality for a in self.analysis_history) / len(self.analysis_history)
        avg_planning = sum(a.planning_quality for a in self.analysis_history) / len(self.analysis_history)
        avg_verification = sum(a.verification_quality for a in self.analysis_history) / len(self.analysis_history)
        
        if avg_execution < 0.7:
            recommendations.append("Focus on improving action grounding and UI element detection")
        
        if avg_planning < 0.7:
            recommendations.append("Enhance planning algorithms and contingency handling")
        
        if avg_verification < 0.7:
            recommendations.append("Strengthen verification methods and confidence scoring")
        
        # Check for common failure patterns
        common_issues = {}
        for analysis in self.analysis_history:
            for issue in analysis.issues_identified:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        most_common = max(common_issues.items(), key=lambda x: x[1]) if common_issues else None
        if most_common and most_common[1] >= len(self.analysis_history) * 0.5:
            recommendations.append(f"Priority fix needed for: {most_common[0]}")
        
        return recommendations

