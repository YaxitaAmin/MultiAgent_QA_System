"""
Verifier Agent - Integrates with Agent-S for result verification
Determines whether the app behaves as expected after each step
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from agents.base_agents import BaseQAAgent, MessageType  # ✅ CORRECTED IMPORT
from agents.planner_agent import PlanStep
from agents.executor_agent import ExecutionResult
from core.android_env_wrapper import AndroidObservation
from core.ui_utils import UIParser, UIElement
from core.logger import QALogger
from config.default_config import config

class VerificationStatus(Enum):
    """Verification result status"""
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"

@dataclass
class VerificationResult:
    """Result of verification check"""
    step_id: int
    status: VerificationStatus
    confidence: float
    expected_state: str
    actual_state: str
    issues_detected: List[str]
    verification_time: float
    reasoning: Optional[str] = None
    suggested_recovery: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        result_dict = asdict(self)
        result_dict['status'] = self.status.value  # Convert enum to string
        return result_dict

class VerifierAgent(BaseQAAgent):
    """
    Agent-S compatible Verifier Agent
    Verifies QA test step results using LLM reasoning and heuristics
    """
    
    def __init__(self):
        super().__init__("VerifierAgent")
        self.ui_parser = UIParser()
        self.verification_history = []
        self.verification_rules = config.VERIFICATION_RULES
        
        self.logger.info("VerifierAgent initialized with Agent-S integration")
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process verification task - ✅ RETURNS DICT, NOT VerificationResult OBJECT"""
        start_time = time.time()
        
        try:
            plan_step = task_data.get("plan_step")
            execution_result = task_data.get("execution_result")
            current_observation = task_data.get("current_observation")
            
            if not plan_step:
                raise ValueError("Missing required verification data: plan_step")
            
            # Perform verification
            verification_result = await self.verify_step_result(
                plan_step, execution_result, current_observation
            )
            
            duration = time.time() - start_time
            
            self.log_action(
                "verify_step",
                {"step_id": plan_step.step_id, "expected": getattr(plan_step, 'success_criteria', 'Unknown')},
                {
                    "status": verification_result.status.value,
                    "confidence": verification_result.confidence,
                    "issues_count": len(verification_result.issues_detected)
                },
                verification_result.status == VerificationStatus.PASS,
                duration,
                None if verification_result.status != VerificationStatus.ERROR else verification_result.reasoning
            )
            
            # Send verification result to other agents
            await self.send_message(
                "all_agents",
                MessageType.VERIFICATION,
                {
                    "verification_result": verification_result.to_dict(),
                    "step_id": plan_step.step_id,
                    "requires_replanning": verification_result.status == VerificationStatus.FAIL
                }
            )
            
            # ✅ FIX: Return dictionary instead of VerificationResult object
            return {
                "success": True,
                "verification_result": {
                    "step_id": verification_result.step_id,
                    "status": verification_result.status.value,
                    "confidence": verification_result.confidence,
                    "expected_state": verification_result.expected_state,
                    "actual_state": verification_result.actual_state,
                    "issues_detected": verification_result.issues_detected,
                    "verification_time": verification_result.verification_time,
                    "reasoning": verification_result.reasoning,
                    "suggested_recovery": verification_result.suggested_recovery
                },
                "requires_replanning": verification_result.status == VerificationStatus.FAIL
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process verification task: {e}")
            
            self.log_action(
                "verify_step",
                task_data,
                {},
                False,
                time.time() - start_time,
                str(e)
            )
            
            return {
                "success": False,
                "verification_result": {
                    "step_id": -1,
                    "status": "ERROR",
                    "confidence": 0.0,
                    "expected_state": "Unknown",
                    "actual_state": f"Error: {str(e)}",
                    "issues_detected": [str(e)],
                    "verification_time": time.time() - start_time,
                    "reasoning": f"Verification failed: {str(e)}",
                    "suggested_recovery": "Review verification system"
                },
                "requires_replanning": False,
                "error": str(e)
            }
    
    async def verify_step_result(self, plan_step: PlanStep, execution_result: Any, 
                               current_observation: Optional[AndroidObservation]) -> VerificationResult:
        """Verify the result of a plan step execution"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Verifying step {plan_step.step_id}: {getattr(plan_step, 'success_criteria', 'Unknown')}")
            
            # Extract current state information
            current_state = self._extract_current_state(current_observation)
            
            # Perform different types of verification
            heuristic_result = await self._verify_with_heuristics(plan_step, execution_result, current_state)
            llm_result = await self._verify_with_llm(plan_step, execution_result, current_state)
            ui_result = await self._verify_ui_state(plan_step, current_observation)
            
            # Combine verification results
            combined_result = self._combine_verification_results(
                plan_step, heuristic_result, llm_result, ui_result
            )
            
            combined_result.verification_time = time.time() - start_time
            
            # Store in history
            self.verification_history.append(combined_result)
            
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Verification failed for step {plan_step.step_id}: {e}")
            
            return VerificationResult(
                step_id=plan_step.step_id,
                status=VerificationStatus.ERROR,
                confidence=0.0,
                expected_state=getattr(plan_step, 'success_criteria', 'Unknown'),
                actual_state="Error during verification",
                issues_detected=[str(e)],
                verification_time=time.time() - start_time,
                reasoning=f"Verification error: {e}"
            )
    
    def _extract_current_state(self, observation: Optional[AndroidObservation]) -> Dict[str, Any]:
        """Extract current state information from observation"""
        if not observation:
            return {"status": "no_observation"}
        
        try:
            ui_hierarchy = getattr(observation, 'ui_hierarchy', '')
            elements = self.ui_parser.parse_ui_hierarchy(ui_hierarchy)
            screen_analysis = self.ui_parser.analyze_screen_state(elements)
            
            return {
                "activity": getattr(observation, 'current_activity', 'unknown'),
                "screen_title": screen_analysis.get("screen_title", "Unknown"),
                "clickable_elements": screen_analysis.get("clickable_elements", 0),
                "input_elements": screen_analysis.get("input_elements", 0),
                "key_texts": screen_analysis.get("key_texts", []),
                "buttons": screen_analysis.get("buttons", []),
                "timestamp": getattr(observation, 'timestamp', time.time()),
                "ui_stable": True  # Assume stable for now
            }
        except Exception as e:
            self.logger.error(f"Failed to extract current state: {e}")
            return {
                "status": "extraction_error",
                "error": str(e),
                "activity": "unknown",
                "timestamp": time.time()
            }
    
    async def _verify_with_heuristics(self, plan_step: PlanStep, execution_result: Any, 
                                    current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify using heuristic rules"""
        issues = []
        confidence = 0.7  # Base confidence for heuristics
        status = VerificationStatus.PASS
        
        try:
            # Check execution success
            exec_success = getattr(execution_result, 'success', 
                                 execution_result.get('success', True) if isinstance(execution_result, dict) else True)
            
            if not exec_success:
                issues.append("Step execution failed")
                status = VerificationStatus.FAIL
                confidence = 0.9
            
            # Check UI stability
            if self.verification_rules.get("ui_stability_check", True):
                if not current_state.get("ui_stable", True):
                    issues.append("UI appears unstable")
                    confidence *= 0.8
            
            # Check for state transitions
            if self.verification_rules.get("screen_transition_validation", True):
                ui_changes = getattr(execution_result, 'ui_changes_detected', 
                                   execution_result.get('ui_changes_detected', False) if isinstance(execution_result, dict) else False)
                
                if ui_changes:
                    # UI changed - this might be expected or unexpected
                    if getattr(plan_step, 'action_type', 'unknown') in ["touch", "type"]:
                        # UI change expected for interactive actions
                        confidence *= 1.1
                    else:
                        # Unexpected UI change
                        issues.append("Unexpected UI change detected")
                        confidence *= 0.9
            
            # Check timeout
            exec_time = getattr(execution_result, 'execution_time', 
                              execution_result.get('execution_time', 0) if isinstance(execution_result, dict) else 0)
            
            if exec_time > self.verification_rules.get("state_transition_timeout", 5):
                issues.append("Action took too long to complete")
                confidence *= 0.8
            
            # Action-specific heuristics
            action_type = getattr(plan_step, 'action_type', 'unknown')
            if action_type == "touch":
                # Verify touch had some effect
                if not ui_changes and "toggle" not in getattr(plan_step, 'description', '').lower():
                    issues.append("Touch action had no visible effect")
                    status = VerificationStatus.INCONCLUSIVE
                    confidence *= 0.7
            
            elif action_type == "type":
                # Check if text input was successful (heuristic)
                if current_state.get("input_elements", 0) == 0:
                    issues.append("No text input elements found after type action")
                    confidence *= 0.8
            
        except Exception as e:
            self.logger.error(f"Heuristic verification failed: {e}")
            issues.append(f"Heuristic verification error: {e}")
            confidence = 0.3
            status = VerificationStatus.ERROR
        
        return {
            "status": status,
            "confidence": max(0.0, min(1.0, confidence)),  # Clamp between 0 and 1
            "issues": issues,
            "method": "heuristics"
        }
    
    async def _verify_with_llm(self, plan_step: PlanStep, execution_result: Any, 
                             current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify using LLM reasoning"""
        try:
            # ✅ SIMPLIFIED FIX: For mock mode, skip complex LLM operations
            if config.USE_MOCK_LLM:
                # Return a simple mock verification
                return {
                    "status": VerificationStatus.PASS,
                    "confidence": 0.6,
                    "issues": [],
                    "reasoning": "Mock LLM verification - basic heuristics applied",
                    "method": "llm_mock"
                }
            
            # Create verification prompt
            verification_prompt = self._create_verification_prompt(plan_step, execution_result, current_state)
            
            # Use Agent-S or LLM interface for verification
            if self.agent_s:
                llm_response = await self._use_agent_s_for_verification(verification_prompt, current_state)
            else:
                llm_response = await self.llm_interface.generate_response(verification_prompt)
            
            # Parse LLM response
            parsed_result = self._parse_llm_verification_response(llm_response.content)
            parsed_result["confidence"] = getattr(llm_response, 'confidence', 0.7)
            parsed_result["method"] = "llm"
            
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"LLM verification failed: {e}")
            return {
                "status": VerificationStatus.INCONCLUSIVE,
                "confidence": 0.3,
                "issues": [f"LLM verification error: {e}"],
                "method": "llm_error"
            }
    
    async def _use_agent_s_for_verification(self, prompt: str, current_state: Dict[str, Any]) -> Any:
        """Use Agent-S for verification with current screen"""
        try:
            # Create mock observation for Agent-S
            import io
            from PIL import Image
            
            # Create simple verification image (could be enhanced with actual screenshot)
            blank_img = Image.new('RGB', (1080, 1920), color='lightgray')
            buffered = io.BytesIO()
            blank_img.save(buffered, format="PNG")
            screenshot_bytes = buffered.getvalue()
            
            obs = {"screenshot": screenshot_bytes}
            
            # Use Agent-S for verification
            info, action = self.agent_s.predict(instruction=prompt, observation=obs)
            
            # Create mock response object
            class MockResponse:
                def __init__(self, content, confidence=0.8):
                    self.content = content
                    self.confidence = confidence
            
            # Extract verification info from Agent-S response
            verification_content = str(info) if info else "Verification completed"
            
            return MockResponse(verification_content, 0.8)
            
        except Exception as e:
            self.logger.error(f"Agent-S verification failed: {e}")
            # Fallback to direct LLM
            return await self.llm_interface.generate_response(prompt)
    
    def _create_verification_prompt(self, plan_step: PlanStep, execution_result: Any, 
                                  current_state: Dict[str, Any]) -> str:
        """Create verification prompt for LLM"""
        return f"""
Verify the result of a QA test step on Android:

STEP INFORMATION:
- Step ID: {plan_step.step_id}
- Action Type: {getattr(plan_step, 'action_type', 'unknown')}
- Target: {getattr(plan_step, 'target_element', 'N/A') or 'N/A'}
- Description: {getattr(plan_step, 'description', 'No description')}
- Success Criteria: {getattr(plan_step, 'success_criteria', 'No criteria specified')}

EXECUTION RESULT:
- Execution Success: {getattr(execution_result, 'success', execution_result.get('success', 'unknown') if isinstance(execution_result, dict) else 'unknown')}
- Execution Time: {getattr(execution_result, 'execution_time', execution_result.get('execution_time', 0) if isinstance(execution_result, dict) else 0):.2f}s
- UI Changes Detected: {getattr(execution_result, 'ui_changes_detected', execution_result.get('ui_changes_detected', False) if isinstance(execution_result, dict) else False)}
- Error Message: {getattr(execution_result, 'error_message', execution_result.get('error_message', 'None') if isinstance(execution_result, dict) else 'None') or 'None'}

CURRENT STATE:
- Activity: {current_state.get('activity', 'Unknown')}
- Screen Title: {current_state.get('screen_title', 'Unknown')}
- Clickable Elements: {current_state.get('clickable_elements', 0)}
- Key UI Texts: {', '.join(current_state.get('key_texts', [])[:5])}

TASK: Analyze whether the success criteria was met. Consider:
1. Did the action execute as expected?
2. Is the UI in the expected state?
3. Are there any apparent bugs or issues?
4. Does the current state match the success criteria?

Respond in JSON format:
{{
    "status": "PASS|FAIL|INCONCLUSIVE",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation",
    "issues": ["list", "of", "issues"],
    "suggestions": "recovery suggestions if failed"
}}
"""
    
    def _parse_llm_verification_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM verification response"""
        try:
            # Try to parse as JSON
            response_json = json.loads(response_content)
            
            status_str = response_json.get("status", "INCONCLUSIVE").upper()
            status = VerificationStatus(status_str) if status_str in [s.value for s in VerificationStatus] else VerificationStatus.INCONCLUSIVE
            
            return {
                "status": status,
                "confidence": response_json.get("confidence", 0.5),
                "issues": response_json.get("issues", []),
                "reasoning": response_json.get("reasoning", ""),
                "suggestions": response_json.get("suggestions", "")
            }
            
        except json.JSONDecodeError:
            # Fallback parsing for non-JSON responses
            response_lower = response_content.lower()
            
            if "pass" in response_lower and "fail" not in response_lower:
                status = VerificationStatus.PASS
                confidence = 0.7
            elif "fail" in response_lower:
                status = VerificationStatus.FAIL
                confidence = 0.8
            else:
                status = VerificationStatus.INCONCLUSIVE
                confidence = 0.5
            
            return {
                "status": status,
                "confidence": confidence,
                "issues": ["Could not parse LLM response"],
                "reasoning": response_content[:200],
                "suggestions": "Review manually"
            }
    
    async def _verify_ui_state(self, plan_step: PlanStep, observation: Optional[AndroidObservation]) -> Dict[str, Any]:
        """Verify UI state directly"""
        if not observation:
            return {
                "status": VerificationStatus.INCONCLUSIVE,
                "confidence": 0.1,
                "issues": ["No observation available"],
                "method": "ui_direct"
            }
        
        try:
            ui_hierarchy = getattr(observation, 'ui_hierarchy', '')
            elements = self.ui_parser.parse_ui_hierarchy(ui_hierarchy)
            issues = []
            confidence = 0.6
            status = VerificationStatus.PASS
            
            # Element visibility checks
            if self.verification_rules.get("element_visibility_check", True):
                clickable_elements = [e for e in elements if getattr(e, 'clickable', False) and getattr(e, 'visible', True)]
                if len(clickable_elements) == 0:
                    issues.append("No clickable elements visible")
                    confidence *= 0.8
            
            # Text content verification
            if self.verification_rules.get("text_content_verification", True):
                # Check for error messages
                error_keywords = ["error", "failed", "cannot", "unable", "invalid"]
                for element in elements:
                    element_text = getattr(element, 'text', '')
                    if element_text:
                        element_text_lower = element_text.lower()
                        if any(keyword in element_text_lower for keyword in error_keywords):
                            issues.append(f"Potential error message detected: {element_text}")
                            status = VerificationStatus.FAIL
                            confidence = 0.9
            
            # Activity-specific checks
            android_world_action = getattr(plan_step, 'android_world_action', None)
            if android_world_action:
                verification_target = android_world_action.get("verification_target", "")
                if verification_target:
                    # Try to find expected elements
                    found_elements = self.ui_parser.find_elements_by_text(elements, verification_target)
                    if not found_elements:
                        issues.append(f"Expected element not found: {verification_target}")
                        status = VerificationStatus.FAIL
                        confidence = 0.8
            
            return {
                "status": status,
                "confidence": confidence,
                "issues": issues,
                "method": "ui_direct"
            }
        except Exception as e:
            self.logger.error(f"UI state verification failed: {e}")
            return {
                "status": VerificationStatus.ERROR,
                "confidence": 0.1,
                "issues": [f"UI verification error: {e}"],
                "method": "ui_direct_error"
            }
    
    def _combine_verification_results(self, plan_step: PlanStep, heuristic_result: Dict[str, Any], 
                                    llm_result: Dict[str, Any], ui_result: Dict[str, Any]) -> VerificationResult:
        """Combine multiple verification results"""
        all_results = [heuristic_result, llm_result, ui_result]
        
        # Combine issues
        all_issues = []
        for result in all_results:
            all_issues.extend(result.get("issues", []))
        
        # Determine overall status (most restrictive wins)
        statuses = [result.get("status", VerificationStatus.INCONCLUSIVE) for result in all_results]
        
        if VerificationStatus.FAIL in statuses:
            final_status = VerificationStatus.FAIL
        elif VerificationStatus.ERROR in statuses:
            final_status = VerificationStatus.ERROR
        elif VerificationStatus.PASS in statuses and VerificationStatus.INCONCLUSIVE not in statuses:
            final_status = VerificationStatus.PASS
        else:
            final_status = VerificationStatus.INCONCLUSIVE
        
        # Average confidence with weights
        weights = {"heuristics": 0.3, "llm": 0.5, "ui_direct": 0.2, "llm_mock": 0.4}
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for result in all_results:
            method = result.get("method", "unknown")
            weight = weights.get(method, 0.1)
            confidence = result.get("confidence", 0.5)
            weighted_confidence += confidence * weight
            total_weight += weight
        
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        # Create reasoning
        reasoning_parts = []
        for result in all_results:
            method = result.get("method", "unknown")
            reasoning = result.get("reasoning", "")
            if reasoning:
                reasoning_parts.append(f"{method}: {reasoning}")
        
        combined_reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Combined verification analysis"
        
        # Generate recovery suggestions
        recovery_suggestion = None
        if final_status == VerificationStatus.FAIL:
            recovery_suggestion = self._generate_recovery_suggestion(plan_step, all_issues)
        
        return VerificationResult(
            step_id=plan_step.step_id,
            status=final_status,
            confidence=final_confidence,
            expected_state=getattr(plan_step, 'success_criteria', 'Unknown'),
            actual_state=self._summarize_actual_state(all_results),
            issues_detected=list(set(all_issues)),  # Remove duplicates
            verification_time=0.0,  # Set by caller
            reasoning=combined_reasoning,
            suggested_recovery=recovery_suggestion
        )
    
    def _summarize_actual_state(self, verification_results: List[Dict[str, Any]]) -> str:
        """Summarize actual state from verification results"""
        state_parts = []
        
        for result in verification_results:
            method = result.get("method", "unknown")
            status = result.get("status", VerificationStatus.INCONCLUSIVE)
            confidence = result.get("confidence", 0.0)
            
            status_str = status.value if hasattr(status, 'value') else str(status)
            state_parts.append(f"{method}: {status_str} (conf: {confidence:.2f})")
        
        return "; ".join(state_parts)
    
    def _generate_recovery_suggestion(self, plan_step: PlanStep, issues: List[str]) -> str:
        """Generate recovery suggestion for failed verification"""
        suggestions = []
        
        # Common recovery patterns
        if any("timeout" in issue.lower() for issue in issues):
            suggestions.append("Increase timeout or add wait step")
        
        if any("element not found" in issue.lower() for issue in issues):
            suggestions.append("Update element selector or check UI hierarchy")
        
        if any("no effect" in issue.lower() for issue in issues):
            suggestions.append("Verify element is clickable and try alternative interaction")
        
        if any("error" in issue.lower() for issue in issues):
            suggestions.append("Check for error dialogs and handle appropriately")
        
        # Action-specific suggestions
        action_type = getattr(plan_step, 'action_type', 'unknown')
        if action_type == "touch":
            suggestions.append("Try different coordinates or element selector")
        elif action_type == "type":
            suggestions.append("Ensure text field is focused before typing")
        elif action_type == "swipe":
            suggestions.append("Adjust swipe coordinates or duration")
        
        return "; ".join(suggestions) if suggestions else "Manual review required"
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """Get summary of all verifications"""
        if not self.verification_history:
            return {"total": 0, "summary": "No verifications performed"}
        
        total = len(self.verification_history)
        passed = sum(1 for v in self.verification_history if v.status == VerificationStatus.PASS)
        failed = sum(1 for v in self.verification_history if v.status == VerificationStatus.FAIL)
        inconclusive = sum(1 for v in self.verification_history if v.status == VerificationStatus.INCONCLUSIVE)
        errors = sum(1 for v in self.verification_history if v.status == VerificationStatus.ERROR)
        
        avg_confidence = sum(v.confidence for v in self.verification_history) / total
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "inconclusive": inconclusive,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0,
            "average_confidence": avg_confidence,
            "summary": f"{passed}/{total} passed ({passed/total*100:.1f}%)"
        }
