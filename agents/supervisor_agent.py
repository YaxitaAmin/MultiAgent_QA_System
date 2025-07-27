"""
Supervisor Agent - Integrates with Agent-S for episode analysis
Reviews entire test episodes and proposes improvements using Gemini 2.5 or mock LLM
"""

import time
import json
import base64
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from agents.base_agents import BaseQAAgent, MessageType  # ✅ CORRECTED IMPORT
from core.logger import QALogger, QATestResult
from core.ui_utils import UIParser
from config.default_config import config

@dataclass
class SupervisorAnalysis:
    """Supervisor analysis of test episode"""
    test_id: str
    overall_assessment: str
    performance_score: float  # 0-1
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    bug_detection_accuracy: float
    agent_recovery_ability: float
    recommended_prompt_improvements: List[str]
    test_coverage_gaps: List[str]
    analysis_timestamp: float

class SupervisorAgent(BaseQAAgent):
    """
    Agent-S compatible Supervisor Agent
    Analyzes complete test episodes and provides feedback for system improvement
    """
    
    def __init__(self):
        super().__init__("SupervisorAgent")
        self.ui_parser = UIParser()
        self.analysis_history = []
        self.visual_traces = []
        
        self.logger.info("SupervisorAgent initialized with Agent-S integration")
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process supervisor analysis task"""
        start_time = time.time()
        
        try:
            test_result = task_data.get("test_result")
            visual_trace = task_data.get("visual_trace", [])
            
            if not test_result:
                raise ValueError("No test result provided for analysis")
            
            # Perform comprehensive analysis
            analysis = await self.analyze_test_episode(test_result, visual_trace)
            
            duration = time.time() - start_time
            
            self.log_action(
                "analyze_episode",
                {"test_id": test_result.test_id, "actions_count": len(test_result.actions)},
                {
                    "performance_score": analysis.performance_score,
                    "suggestions_count": len(analysis.improvement_suggestions),
                    "coverage_gaps": len(analysis.test_coverage_gaps)
                },
                True,
                duration
            )
            
            # Send analysis to other agents
            await self.send_message(
                "all_agents",
                MessageType.TASK_RESPONSE,
                {
                    "supervisor_analysis": analysis.__dict__,
                    "test_id": test_result.test_id
                }
            )
            
            return {
                "success": True,
                "analysis": analysis,
                "performance_score": analysis.performance_score
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process supervisor analysis: {e}")
            
            self.log_action(
                "analyze_episode",
                task_data,
                {},
                False,
                time.time() - start_time,
                str(e)
            )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_test_episode(self, test_result: QATestResult, 
                                 visual_trace: List[Dict[str, Any]]) -> SupervisorAnalysis:
        """Analyze complete test episode using Agent-S and LLM capabilities"""
        
        try:
            self.logger.info(f"Analyzing test episode: {test_result.test_id}")
            
            # Store visual trace
            self.visual_traces.append({
                "test_id": test_result.test_id,
                "trace": visual_trace,
                "timestamp": time.time()
            })
            
            # Perform different types of analysis
            performance_analysis = await self._analyze_performance(test_result)
            agent_effectiveness = await self._analyze_agent_effectiveness(test_result)
            bug_detection_analysis = await self._analyze_bug_detection(test_result)
            
            # Use Agent-S/LLM for deeper analysis
            llm_analysis = await self._perform_llm_analysis(test_result, visual_trace)
            
            # Combine all analyses
            combined_analysis = self._combine_analyses(
                test_result, performance_analysis, agent_effectiveness, 
                bug_detection_analysis, llm_analysis
            )
            
            # Store in history
            self.analysis_history.append(combined_analysis)
            
            return combined_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze test episode: {e}")
            
            # Return minimal analysis
            return SupervisorAnalysis(
                test_id=test_result.test_id,
                overall_assessment="Analysis failed",
                performance_score=0.0,
                strengths=[],
                weaknesses=[f"Analysis error: {e}"],
                improvement_suggestions=["Review analysis system"],
                bug_detection_accuracy=0.0,
                agent_recovery_ability=0.0,
                recommended_prompt_improvements=[],
                test_coverage_gaps=[],
                analysis_timestamp=time.time()
            )
    
    async def _analyze_performance(self, test_result: QATestResult) -> Dict[str, Any]:
        """Analyze overall test performance"""
        total_actions = len(test_result.actions)
        successful_actions = sum(1 for action in test_result.actions if action.success)
        
        success_rate = successful_actions / total_actions if total_actions > 0 else 0
        avg_duration = sum(action.duration for action in test_result.actions) / total_actions if total_actions > 0 else 0
        
        # Performance scoring
        performance_score = 0.0
        if test_result.final_result == "PASS":
            performance_score += 0.5
        
        performance_score += (success_rate * 0.3)  # Action success rate
        
        if avg_duration < 3.0:  # Fast execution
            performance_score += 0.1
        elif avg_duration > 10.0:  # Slow execution
            performance_score -= 0.1
        
        if test_result.bug_detected:
            performance_score += 0.1  # Bonus for bug detection
        
        performance_score = max(0.0, min(1.0, performance_score))
        
        return {
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "performance_score": performance_score,
            "total_actions": total_actions,
            "test_passed": test_result.final_result == "PASS"
        }
    
    async def _analyze_agent_effectiveness(self, test_result: QATestResult) -> Dict[str, Any]:
        """Analyze individual agent effectiveness"""
        agent_stats = {}
        
        for action in test_result.actions:
            agent_name = action.agent_name
            if agent_name not in agent_stats:
                agent_stats[agent_name] = {
                    "total_actions": 0,
                    "successful_actions": 0,
                    "total_duration": 0.0,
                    "error_count": 0
                }
            
            stats = agent_stats[agent_name]
            stats["total_actions"] += 1
            stats["total_duration"] += action.duration
            
            if action.success:
                stats["successful_actions"] += 1
            else:
                stats["error_count"] += 1
        
        # Calculate effectiveness metrics
        for agent_name, stats in agent_stats.items():
            if stats["total_actions"] > 0:
                stats["success_rate"] = stats["successful_actions"] / stats["total_actions"]
                stats["avg_duration"] = stats["total_duration"] / stats["total_actions"]
                stats["effectiveness_score"] = stats["success_rate"] * (1.0 - min(stats["avg_duration"] / 10.0, 0.5))
            else:
                stats["success_rate"] = 0.0
                stats["avg_duration"] = 0.0
                stats["effectiveness_score"] = 0.0
        
        # Overall recovery ability
        recovery_actions = [a for a in test_result.actions if "recover" in a.action_type.lower() or "replan" in a.action_type.lower()]
        recovery_ability = len(recovery_actions) / max(1, len([a for a in test_result.actions if not a.success]))
        recovery_ability = min(1.0, recovery_ability)
        
        return {
            "agent_stats": agent_stats,
            "recovery_ability": recovery_ability,
            "coordination_quality": self._assess_coordination_quality(test_result.actions)
        }
    
    def _assess_coordination_quality(self, actions: List) -> float:
        """Assess how well agents coordinated"""
        if len(actions) < 2:
            return 1.0
        
        # Simple coordination metric: how well agents passed information
        coordination_score = 0.8  # Base score
        
        # Check for agent transitions
        agent_transitions = 0
        for i in range(1, len(actions)):
            if actions[i].agent_name != actions[i-1].agent_name:
                agent_transitions += 1
        
        # Good coordination should have reasonable transitions
        expected_transitions = len(actions) * 0.3  # Expect 30% transitions
        transition_ratio = agent_transitions / expected_transitions if expected_transitions > 0 else 1.0
        
        if 0.5 <= transition_ratio <= 1.5:
            coordination_score += 0.1
        else:
            coordination_score -= 0.1
        
        return max(0.0, min(1.0, coordination_score))
    
    async def _analyze_bug_detection(self, test_result: QATestResult) -> Dict[str, Any]:
        """Analyze bug detection effectiveness"""
        verification_actions = [a for a in test_result.actions if a.agent_name == "VerifierAgent"]
        
        if not verification_actions:
            return {
                "bug_detection_accuracy": 0.0,
                "false_positives": 0,
                "false_negatives": 0,
                "detection_confidence": 0.0,
                "verification_count": 0  # ✅ REQUIRED FIELD
            }
        
        # Estimate detection accuracy (simplified)
        bug_detection_accuracy = 0.7  # Base accuracy
        
        if test_result.bug_detected and test_result.final_result == "FAIL":
            bug_detection_accuracy += 0.2  # Correctly identified failure
        elif not test_result.bug_detected and test_result.final_result == "PASS":
            bug_detection_accuracy += 0.1  # Correctly identified success
        elif test_result.bug_detected and test_result.final_result == "PASS":
            bug_detection_accuracy -= 0.2  # False positive
        else:
            bug_detection_accuracy -= 0.1  # False negative
        
        bug_detection_accuracy = max(0.0, min(1.0, bug_detection_accuracy))
        
        return {
            "bug_detection_accuracy": bug_detection_accuracy,
            "verification_count": len(verification_actions),  # ✅ REQUIRED FIELD
            "detection_confidence": sum(getattr(a, 'confidence', 0.5) for a in verification_actions) / len(verification_actions)
        }
    
    async def _perform_llm_analysis(self, test_result: QATestResult, 
                                  visual_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform deep analysis using Agent-S/LLM capabilities"""
        try:
            # ✅ SIMPLIFIED FIX: For mock mode, skip complex async operations
            if config.USE_MOCK_LLM:
                # Return a comprehensive mock analysis to avoid async issues
                return {
                    "llm_assessment": "Mock analysis completed successfully - System executed basic test workflow",
                    "strengths": [
                        "System initialized all agents correctly",
                        "Mock environment responded appropriately",
                        "Basic agent coordination functional",
                        "Test execution pipeline operational"
                    ],
                    "weaknesses": [
                        "Limited real-world validation in mock mode",
                        "Simplified execution paths",
                        "No actual UI interaction validation",
                        "Mock responses may not reflect real scenarios"
                    ],
                    "suggestions": [
                        "Test with real Android environment when available",
                        "Validate UI interactions with actual device",
                        "Implement more sophisticated mock scenarios",
                        "Add integration tests with real hardware"
                    ],
                    "prompt_improvements": [
                        "Consider more specific task descriptions",
                        "Add error handling instructions",
                        "Include device-specific considerations",
                        "Enhance test case specificity"
                    ],
                    "coverage_gaps": [
                        "Real device testing scenarios",
                        "Network connectivity variations",
                        "Different Android versions",
                        "Hardware-specific behaviors"
                    ],
                    "coordination_score": 0.7,
                    "interaction_quality": 0.6,
                    "error_handling": 0.5
                }
            
            # For real LLM mode (when implemented)
            analysis_prompt = self._create_analysis_prompt(test_result, visual_trace)
            
            if self.agent_s:
                llm_response = await self._use_agent_s_for_analysis(analysis_prompt, visual_trace)
            else:
                llm_response = await self.llm_interface.generate_response(analysis_prompt)
            
            parsed_analysis = self._parse_llm_analysis_response(llm_response.content)
            return parsed_analysis
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {
                "llm_assessment": f"Analysis failed: {e}",
                "strengths": [],
                "weaknesses": [f"LLM analysis error: {e}"],
                "suggestions": ["Review LLM analysis system"],
                "prompt_improvements": [],
                "coverage_gaps": []
            }
    
    async def _use_agent_s_for_analysis(self, prompt: str, visual_trace: List[Dict[str, Any]]) -> Any:
        """Use Agent-S for analysis with visual traces"""
        try:
            # Create observation with visual data if available
            import io
            from PIL import Image
            
            # Use last screenshot if available
            if visual_trace and "screenshot" in visual_trace[-1]:
                screenshot_data = visual_trace[-1]["screenshot"]
                if isinstance(screenshot_data, str):
                    # Base64 encoded
                    screenshot_bytes = base64.b64decode(screenshot_data)
                else:
                    # Already bytes
                    screenshot_bytes = screenshot_data
            else:
                # Create mock screenshot
                blank_img = Image.new('RGB', (1080, 1920), color='white')
                buffered = io.BytesIO()
                blank_img.save(buffered, format="PNG")
                screenshot_bytes = buffered.getvalue()
            
            obs = {"screenshot": screenshot_bytes}
            
            # Use Agent-S for analysis
            info, action = self.agent_s.predict(instruction=prompt, observation=obs)
            
            # Create mock response object
            class MockResponse:
                def __init__(self, content, confidence=0.8):
                    self.content = content
                    self.confidence = confidence
            
            analysis_content = str(info) if info else "Analysis completed"
            
            return MockResponse(analysis_content, 0.8)
            
        except Exception as e:
            self.logger.error(f"Agent-S analysis failed: {e}")
            # ✅ FIX: Await the fallback LLM interface
            return await self.llm_interface.generate_response(prompt)
    
    def _create_analysis_prompt(self, test_result: QATestResult, 
                               visual_trace: List[Dict[str, Any]]) -> str:
        """Create comprehensive analysis prompt"""
        return f"""
Analyze this mobile QA test episode and provide comprehensive feedback:

TEST SUMMARY:
- Test ID: {test_result.test_id}
- Task: {test_result.task_name}
- Result: {test_result.final_result}
- Duration: {test_result.end_time - test_result.start_time:.2f}s
- Actions Count: {len(test_result.actions)}
- Bug Detected: {test_result.bug_detected}

AGENT ACTIONS:
{self._format_actions_for_analysis(test_result.actions)}

VISUAL TRACE:
- Screenshots captured: {len(visual_trace)}
- UI transitions observed: {len([t for t in visual_trace if t.get('ui_changed', False)])}

ANALYSIS REQUIREMENTS:
1. Overall test effectiveness assessment
2. Agent coordination and communication quality  
3. Bug detection accuracy and false positive/negative analysis
4. UI interaction quality and precision
5. Recovery and error handling effectiveness
6. Test coverage gaps identification
7. Prompt optimization suggestions
8. System improvement recommendations

Respond in JSON format:
{{
    "overall_assessment": "detailed assessment",
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"], 
    "suggestions": ["improvement suggestions"],
    "prompt_improvements": ["prompt optimization ideas"],
    "coverage_gaps": ["missing test scenarios"],
    "agent_coordination_score": 0.0-1.0,
    "ui_interaction_quality": 0.0-1.0,
    "error_handling_score": 0.0-1.0
}}
"""
    
    def _format_actions_for_analysis(self, actions: List) -> str:
        """Format actions for analysis prompt"""
        formatted_actions = []
        
        for i, action in enumerate(actions[:10]):  # Limit to first 10 actions
            formatted_actions.append(
                f"{i+1}. {action.agent_name}: {action.action_type} "
                f"({'SUCCESS' if action.success else 'FAILED'}) "
                f"[{action.duration:.1f}s]"
            )
        
        if len(actions) > 10:
            formatted_actions.append(f"... and {len(actions) - 10} more actions")
        
        return "\n".join(formatted_actions)
    
    def _parse_llm_analysis_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        try:
            # Try to parse as JSON
            response_json = json.loads(response_content)
            
            return {
                "llm_assessment": response_json.get("overall_assessment", "Analysis completed"),
                "strengths": response_json.get("strengths", []),
                "weaknesses": response_json.get("weaknesses", []),
                "suggestions": response_json.get("suggestions", []),
                "prompt_improvements": response_json.get("prompt_improvements", []),
                "coverage_gaps": response_json.get("coverage_gaps", []),
                "coordination_score": response_json.get("agent_coordination_score", 0.5),
                "interaction_quality": response_json.get("ui_interaction_quality", 0.5),
                "error_handling": response_json.get("error_handling_score", 0.5)
            }
            
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "llm_assessment": response_content[:200],
                "strengths": ["Analysis completed"],
                "weaknesses": ["Could not parse detailed analysis"],
                "suggestions": ["Review LLM response format"],
                "prompt_improvements": [],
                "coverage_gaps": [],
                "coordination_score": 0.5,
                "interaction_quality": 0.5,
                "error_handling": 0.5
            }
    
    def _combine_analyses(self, test_result: QATestResult, performance_analysis: Dict[str, Any],
                         agent_effectiveness: Dict[str, Any], bug_detection_analysis: Dict[str, Any],
                         llm_analysis: Dict[str, Any]) -> SupervisorAnalysis:
        """Combine all analyses into comprehensive assessment"""
        
        # Calculate overall performance score
        performance_score = (
            performance_analysis["performance_score"] * 0.4 +
            agent_effectiveness["recovery_ability"] * 0.2 +
            bug_detection_analysis["bug_detection_accuracy"] * 0.2 +
            llm_analysis.get("coordination_score", 0.5) * 0.1 +
            llm_analysis.get("interaction_quality", 0.5) * 0.1
        )
        
        # Combine strengths
        strengths = []
        if performance_analysis["success_rate"] > 0.8:
            strengths.append("High action success rate")
        if performance_analysis["avg_duration"] < 3.0:
            strengths.append("Fast execution")
        if bug_detection_analysis["bug_detection_accuracy"] > 0.7:
            strengths.append("Good bug detection")
        if agent_effectiveness["recovery_ability"] > 0.5:
            strengths.append("Effective error recovery")
        
        strengths.extend(llm_analysis.get("strengths", []))
        
        # Combine weaknesses
        weaknesses = []
        if performance_analysis["success_rate"] < 0.6:
            weaknesses.append("Low action success rate")
        if performance_analysis["avg_duration"] > 8.0:
            weaknesses.append("Slow execution")
        if bug_detection_analysis["bug_detection_accuracy"] < 0.5:
            weaknesses.append("Poor bug detection")
        if agent_effectiveness["recovery_ability"] < 0.3:
            weaknesses.append("Ineffective error recovery")
        
        weaknesses.extend(llm_analysis.get("weaknesses", []))
        
        # Generate improvement suggestions
        suggestions = []
        if performance_analysis["avg_duration"] > 5.0:
            suggestions.append("Optimize action timing and reduce delays")
        if agent_effectiveness["coordination_quality"] < 0.7:
            suggestions.append("Improve inter-agent communication")
        if bug_detection_analysis["verification_count"] < 3:
            suggestions.append("Increase verification frequency")
        
        suggestions.extend(llm_analysis.get("suggestions", []))
        
        return SupervisorAnalysis(
            test_id=test_result.test_id,
            overall_assessment=llm_analysis.get("llm_assessment", "Analysis completed"),
            performance_score=performance_score,
            strengths=list(set(strengths)),  # Remove duplicates
            weaknesses=list(set(weaknesses)),
            improvement_suggestions=list(set(suggestions)),
            bug_detection_accuracy=bug_detection_analysis["bug_detection_accuracy"],
            agent_recovery_ability=agent_effectiveness["recovery_ability"],
            recommended_prompt_improvements=llm_analysis.get("prompt_improvements", []),
            test_coverage_gaps=llm_analysis.get("coverage_gaps", []),
            analysis_timestamp=time.time()
        )
    
    def generate_evaluation_report(self, test_results: List[QATestResult]) -> str:
        """Generate comprehensive evaluation report"""
        if not self.analysis_history:
            return "No supervisor analyses available for report generation."
        
        # Calculate aggregate metrics
        total_tests = len(self.analysis_history)
        avg_performance = sum(a.performance_score for a in self.analysis_history) / total_tests
        avg_bug_detection = sum(a.bug_detection_accuracy for a in self.analysis_history) / total_tests
        avg_recovery = sum(a.agent_recovery_ability for a in self.analysis_history) / total_tests
        
        # Collect common issues and suggestions
        all_weaknesses = []
        all_suggestions = []
        all_coverage_gaps = []
        
        for analysis in self.analysis_history:
            all_weaknesses.extend(analysis.weaknesses)
            all_suggestions.extend(analysis.improvement_suggestions)
            all_coverage_gaps.extend(analysis.test_coverage_gaps)
        
        # Count frequency of issues
        from collections import Counter
        weakness_counts = Counter(all_weaknesses)
        suggestion_counts = Counter(all_suggestions)
        coverage_counts = Counter(all_coverage_gaps)
        
        report = f"""
# Multi-Agent QA System Evaluation Report

## Executive Summary
- **Total Tests Analyzed**: {total_tests}
- **Average Performance Score**: {avg_performance:.2f}/1.0
- **Bug Detection Accuracy**: {avg_bug_detection:.1%}
- **Agent Recovery Ability**: {avg_recovery:.1%}

## Performance Analysis

### Overall System Performance
The multi-agent QA system achieved an average performance score of {avg_performance:.2f} across {total_tests} test episodes. 

### Bug Detection Effectiveness
The system demonstrated {avg_bug_detection:.1%} accuracy in bug detection, with verification agents successfully identifying issues in mobile UI interactions.

### Agent Recovery and Coordination
Agents showed {avg_recovery:.1%} effectiveness in error recovery, indicating {'good' if avg_recovery > 0.7 else 'moderate' if avg_recovery > 0.4 else 'poor'} coordination and adaptive planning capabilities.

## Most Common Issues
"""
        
        # Add top issues
        for i, (weakness, count) in enumerate(weakness_counts.most_common(5)):
            report += f"{i+1}. **{weakness}** (occurred {count} times)\n"
        
        report += "\n## Recommended Improvements\n"
        
        # Add top suggestions
        for i, (suggestion, count) in enumerate(suggestion_counts.most_common(5)):
            report += f"{i+1}. **{suggestion}** (suggested {count} times)\n"
        
        report += "\n## Test Coverage Gaps\n"
        
        # Add coverage gaps
        for i, (gap, count) in enumerate(coverage_counts.most_common(3)):
            report += f"{i+1}. **{gap}** (identified {count} times)\n"
        
        report += f"""

## Agent-S Integration Assessment
The system successfully integrates with Agent-S framework, utilizing:
- Modular messaging architecture for inter-agent communication
- LLM-powered decision making for intelligent UI interaction
- Grounded action execution using android_world environment
- Visual reasoning capabilities for UI state analysis

## Android World Integration Assessment  
The android_world integration provides:
- Reproducible Android UI testing environment
- Support for {len(config.ANDROID_WORLD_TASKS)} different task types
- Realistic mobile app interaction simulation
- Comprehensive UI hierarchy access for verification

## Recommendations for System Enhancement

### Immediate Actions
1. Address the most frequent issues identified above
2. Implement suggested improvements with high occurrence rates
3. Expand test coverage for identified gaps

### Long-term Improvements
1. Enhance agent coordination protocols
2. Improve LLM prompt engineering based on analysis feedback
3. Expand android_world task coverage
4. Integrate additional verification heuristics

---
*Report generated by SupervisorAgent at {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
