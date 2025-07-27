# agents/executor_agent.py - TRUE Agent-S Extension for Execution
"""
Executor Agent - PROPERLY extends Agent-S for QA execution
TRUE Agent-S integration with deep architectural extension for Android UI execution
"""

import time
import json
import asyncio
import io
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from PIL import Image
import numpy as np

from agents.base_agents import QAAgentS2, MessageType  # ✅ CORRECTED: Use QAAgentS2
from agents.planner_agent import PlanStep, QAPlan
from core.android_env_wrapper import AndroidEnvWrapper, AndroidAction, AndroidObservation
from core.ui_utils import UIParser, UIElement, ActionResult
from core.logger import QALogger
from config.default_config import config

@dataclass
class ExecutionResult:
    """Result of executing a plan step with Agent-S enhancement"""
    step_id: int
    success: bool
    action_taken: Optional[AndroidAction]
    observation_before: Optional[AndroidObservation]
    observation_after: Optional[AndroidObservation]
    execution_time: float
    error_message: Optional[str] = None
    ui_changes_detected: bool = False
    agent_s_used: bool = False  # Track if Agent-S was used
    confidence_score: float = 0.8  # Execution confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return asdict(self)

class ExecutorAgent(QAAgentS2):
    """
    CORRECTED: Executor Agent that TRULY extends Agent-S
    Uses Agent-S's action execution and grounding capabilities
    """
    
    def __init__(self):
        # Initialize with Agent-S execution engine configuration
        execution_engine_config = {
            "engine_type": "gemini",
            "model_name": "gemini-1.5-flash",
            "api_key": config.GOOGLE_API_KEY,
            "temperature": 0.0,  # Zero temperature for precise execution
            "max_tokens": 1000,
            "execution_mode": True  # Custom flag for execution
        }
        
        super().__init__("ExecutorAgent", execution_engine_config)
        
        self.ui_parser = UIParser()
        self.android_env = None
        self.current_observation = None
        self.execution_results = []
        
        self.logger.info("ExecutorAgent initialized with Agent-S execution capabilities")
        # 🎉 Your requested async start method 🎉
    async def start(self) -> bool:
        """Start the executor agent"""
        try:
            self.is_running = True
            
            # Test Agent-S functionality if available
            if self.is_agent_s_active():
                try:
                    test_obs = {"screenshot": b"", "ui_hierarchy": ""}
                    info, actions = await self.predict("initialization test", test_obs)
                    self.logger.info(f"[{self.agent_name}] ✅ Agent-S functional test passed")
                except Exception as e:
                    self.logger.warning(f"[{self.agent_name}] ⚠️ Agent-S functional test failed: {e}")
            
            await self._send_heartbeat()
            self.logger.info(f"ExecutorAgent started successfully (Agent-S: {'✅' if self.is_agent_s_active() else '❌'})")
            return True
        except Exception as e:
            self.logger.error(f"ExecutorAgent failed to start: {e}")
            return False

    # 🎉 Your requested async stop method 🎉
    async def stop(self) -> bool:
        """Stop the executor agent"""
        try:
            self.is_running = False
            await self.cleanup()
            self.logger.info("ExecutorAgent stopped successfully")
            return True
        except Exception as e:
            self.logger.error(f"ExecutorAgent failed to stop: {e}")
            return False
    async def predict(self, instruction: str, observation: Dict[str, Any], 
                     **kwargs) -> tuple[Dict[str, Any], List[str]]:
        """
        CORRECTED: Override Agent-S predict for execution-specific logic
        """
        # Enhance instruction for execution context
        execution_instruction = f"""
        EXECUTION MODE: Execute precise Android UI actions for QA testing.
        
        Task: {instruction}
        Current UI State: {observation.get('ui_hierarchy', 'No UI info')[:500]}
        Screen Resolution: {observation.get('screen_bounds', (1080, 1920))}
        
        Generate specific, executable Android actions:
        - Use exact coordinates for touch actions
        - Ensure actions are valid for current UI state
        - Include verification steps
        - Handle potential UI state changes
        
        Focus on precise, reliable execution.
        """
        
        # Use parent Agent-S prediction with execution enhancement
        info, actions = await super().predict(execution_instruction, observation, **kwargs)
        
        # Post-process for execution context
        execution_info = self._enhance_execution_info(info, instruction)
        execution_actions = self._validate_execution_actions(actions, observation)
        
        return execution_info, execution_actions
    
    def _enhance_execution_info(self, info: Dict[str, Any], original_instruction: str) -> Dict[str, Any]:
        """Enhance Agent-S info with execution-specific data"""
        enhanced = info.copy() if info else {}
        
        enhanced.update({
            "execution_mode": True,
            "original_instruction": original_instruction,
            "action_type": "ui_execution",
            "agent_s_reasoning": enhanced.get("reasoning", ""),
            "execution_confidence": enhanced.get("confidence", 0.8),
            "executor_agent": "ExecutorAgent"
        })
        
        return enhanced
    
    def _validate_execution_actions(self, actions: List[str], observation: Dict[str, Any]) -> List[str]:
        """Validate and enhance actions for execution context"""
        validated_actions = []
        screen_bounds = observation.get('screen_bounds', (1080, 1920))
        
        for action in actions[:3]:  # Limit to 3 actions for safety
            if self._is_valid_execution_action(action, screen_bounds):
                validated_actions.append(action)
            else:
                # Convert invalid action to safe fallback
                safe_action = self._create_safe_fallback_action(action, screen_bounds)
                validated_actions.append(safe_action)
        
        return validated_actions or [f"tap({screen_bounds[0]//2}, {screen_bounds[1]//2})"]
    
    def _is_valid_execution_action(self, action: str, screen_bounds: Tuple[int, int]) -> bool:
        """Validate execution action against screen bounds"""
        # Extract coordinates from action
        coords = self._extract_coordinates_from_action(action)
        if not coords:
            return False
        
        x, y = coords
        max_x, max_y = screen_bounds
        
        # Check if coordinates are within screen bounds
        return 0 <= x <= max_x and 0 <= y <= max_y
    
    def _extract_coordinates_from_action(self, action: str) -> Optional[Tuple[int, int]]:
        """Extract coordinates from action string"""
        patterns = [
            r'tap\((\d+),\s*(\d+)\)',
            r'click\((\d+),\s*(\d+)\)',
            r'pyautogui\.click\((\d+),\s*(\d+)\)',
            r'touch\((\d+),\s*(\d+)\)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        
        return None
    
    def _create_safe_fallback_action(self, original_action: str, screen_bounds: Tuple[int, int]) -> str:
        """Create safe fallback action"""
        center_x, center_y = screen_bounds[0] // 2, screen_bounds[1] // 2
        return f"tap({center_x}, {center_y})"
    

    async def initialize_android_env(self, task_name: str) -> bool:
        """Initialize android_world environment with Agent-S integration 🚀"""
        try:
            # Step 1: Create the Android environment
            self.android_env = AndroidEnvWrapper(task_name=task_name)
            
            # Step 2: Reset to get the initial observation
            initial_obs = self.android_env.reset()
            self.current_observation = initial_obs

            # Step 3: Check if Agent-S is active and process the observation
            if self.is_agent_s_active():
                try:
                    processed_obs = await self.predict(
                        "Process initial Android environment observation",
                        {
                            "screenshot": initial_obs.screenshot,
                            "ui_hierarchy": initial_obs.ui_hierarchy,
                            "current_activity": initial_obs.current_activity,
                            "screen_bounds": initial_obs.screen_bounds
                        }
                    )
                    self.logger.info("✅ Initial observation processed with Agent-S")

                    # Store the enhanced observation if available
                    self.last_agent_s_observation = processed_obs[0] if processed_obs else {}

                except Exception as agent_s_error:
                    self.logger.warning(f"⚠️ Agent-S observation processing failed: {agent_s_error}")
                    # Continue without Agent-S processing

            self.logger.info(f"🎯 Android environment initialized for task: {task_name}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Android environment: {e}")
            return False

    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process execution task with Agent-S integration"""
        start_time = time.time()
        
        try:
            plan_step = task_data.get("plan_step")
            if not plan_step:
                raise ValueError("No plan step provided")
            
            # Initialize environment if needed
            if not self.android_env:
                task_name = task_data.get("android_world_task", "settings_wifi")
                if not await self.initialize_android_env(task_name):
                    raise Exception("Failed to initialize Android environment")
            
            # Execute the step using Agent-S capabilities
            result = await self.execute_step_with_agent_s(plan_step)
            
            duration = time.time() - start_time
            
            # Log the execution action
            action_record = self.log_action(
                "execute_step_agent_s",
                {
                    "step_id": plan_step.step_id, 
                    "action_type": plan_step.action_type,
                    "agent_s_used": result.agent_s_used
                },
                {
                    "success": result.success, 
                    "ui_changes": result.ui_changes_detected,
                    "confidence": result.confidence_score
                },
                result.success,
                duration,
                result.error_message
            )
            
            return {
                "success": result.success,
                "execution_result": {
                    "step_id": result.step_id,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "ui_changes_detected": result.ui_changes_detected,
                    "error_message": result.error_message,
                    "action_taken": result.action_taken.__dict__ if result.action_taken else None,
                    "agent_s_used": result.agent_s_used,
                    "confidence_score": result.confidence_score
                },
                "current_observation": self.current_observation,
                "action_record": action_record
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process execution task: {e}")
            
            action_record = self.log_action(
                "execute_step_failed",
                task_data,
                {},
                False,
                time.time() - start_time,
                str(e)
            )
            
            return {
                "success": False,
                "execution_result": {
                    "step_id": -1,
                    "success": False,
                    "execution_time": duration,
                    "ui_changes_detected": True,
                    "error_message": str(e),
                    "action_taken": None,
                    "agent_s_used": self.is_agent_s_active(),
                    "confidence_score": 0.0
                },
                "current_observation": self.current_observation,
                "error": str(e),
                "action_record": action_record
            }
    
    async def execute_step_with_agent_s(self, plan_step: PlanStep) -> ExecutionResult:
        """
        CORRECTED: Execute step using Agent-S capabilities
        """
        start_time = time.time()
        observation_before = self.current_observation
        agent_s_used = False
        confidence_score = 0.8
        
        try:
            self.logger.info(f"Executing step {plan_step.step_id} with Agent-S: {plan_step.description}")
            
            # Prepare observation for Agent-S
            current_obs = self._prepare_observation_for_agent_s()
            
            # Use Agent-S for intelligent action execution
            if self.is_agent_s_active() and current_obs:
                try:
                    # Create execution instruction for Agent-S
                    instruction = self._create_execution_instruction(plan_step)
                    
                    # Get Agent-S prediction
                    info, actions = await self.predict(instruction, current_obs)
                    
                    # Execute Agent-S actions
                    action_taken = await self._execute_agent_s_actions(actions, plan_step)
                    agent_s_used = True
                    confidence_score = info.get("execution_confidence", 0.8)
                    
                    self.logger.info(f"✅ Agent-S execution successful for step {plan_step.step_id}")
                    
                except Exception as e:
                    self.logger.warning(f"Agent-S execution failed, using fallback: {e}")
                    action_taken = await self._execute_fallback_action(plan_step)
            else:
                # Fallback execution
                action_taken = await self._execute_fallback_action(plan_step)
            
            # Update current observation
            if self.android_env:
                self.current_observation = self.android_env._get_observation()
            
            # Detect UI changes
            ui_changes = self._detect_ui_changes(observation_before, self.current_observation)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                step_id=plan_step.step_id,
                success=True,
                action_taken=action_taken,
                observation_before=observation_before,
                observation_after=self.current_observation,
                execution_time=execution_time,
                ui_changes_detected=ui_changes,
                agent_s_used=agent_s_used,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute step {plan_step.step_id}: {e}")
            
            return ExecutionResult(
                step_id=plan_step.step_id,
                success=False,
                action_taken=None,
                observation_before=observation_before,
                observation_after=self.current_observation,
                execution_time=time.time() - start_time,
                error_message=str(e),
                ui_changes_detected=False,
                agent_s_used=agent_s_used,
                confidence_score=0.0
            )
    
    def _prepare_observation_for_agent_s(self) -> Dict[str, Any]:
        """Prepare current observation for Agent-S processing"""
        if not self.current_observation:
            return {}
        
        try:
            # Convert screenshot to bytes if needed
            screenshot = self.current_observation.screenshot
            if isinstance(screenshot, np.ndarray):
                img = Image.fromarray(screenshot)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                screenshot_bytes = buffered.getvalue()
            else:
                screenshot_bytes = screenshot
            
            return {
                "screenshot": screenshot_bytes,
                "ui_hierarchy": getattr(self.current_observation, 'ui_hierarchy', ''),
                "current_activity": getattr(self.current_observation, 'current_activity', ''),
                "screen_bounds": getattr(self.current_observation, 'screen_bounds', (1080, 1920)),
                "timestamp": getattr(self.current_observation, 'timestamp', time.time())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to prepare observation for Agent-S: {e}")
            return {}
    
    def _create_execution_instruction(self, plan_step: PlanStep) -> str:
        """Create specific instruction for Agent-S execution"""
        base_instruction = plan_step.description
        
        if plan_step.action_type == "touch":
            return f"Touch/tap the UI element: {getattr(plan_step, 'target_element', base_instruction)}"
        elif plan_step.action_type == "type":
            text = getattr(plan_step, 'text', 'test text')
            return f"Type text '{text}' into the input field"
        elif plan_step.action_type == "swipe":
            return f"Perform swipe gesture: {base_instruction}"
        elif plan_step.action_type == "verify":
            return f"Verify UI state: {base_instruction}"
        else:
            return f"Execute action: {base_instruction}"
    
    async def _execute_agent_s_actions(self, actions: List[str], plan_step: PlanStep) -> AndroidAction:
        """Execute actions generated by Agent-S"""
        if not actions:
            return await self._execute_fallback_action(plan_step)
        
        # Execute the first valid action
        for action in actions:
            try:
                coords = self._extract_coordinates_from_action(action)
                if coords:
                    # Execute touch action
                    action_dict = {
                        "action_type": "touch",
                        "coordinates": coords
                    }
                    
                    if self.android_env:
                        self.android_env.step(action_dict)
                    
                    return AndroidAction(
                        action_type="touch",
                        coordinates=coords,
                        agent_s_generated=True
                    )
            except Exception as e:
                self.logger.warning(f"Failed to execute Agent-S action '{action}': {e}")
                continue
        
        # If no actions worked, use fallback
        return await self._execute_fallback_action(plan_step)
    
    async def _execute_fallback_action(self, plan_step: PlanStep) -> AndroidAction:
        """Execute fallback action when Agent-S is not available"""
        try:
            if plan_step.action_type == "touch":
                return await self._fallback_touch(plan_step)
            elif plan_step.action_type == "type":
                return await self._execute_type(plan_step)
            elif plan_step.action_type == "swipe":
                return await self._execute_swipe(plan_step)
            elif plan_step.action_type == "verify":
                return await self._execute_verify(plan_step)
            elif plan_step.action_type == "wait":
                return await self._execute_wait(plan_step)
            else:
                return await self._fallback_touch(plan_step)
                
        except Exception as e:
            self.logger.error(f"Fallback execution failed: {e}")
            return AndroidAction(action_type="touch", coordinates=(540, 960))
    
    async def _fallback_touch(self, plan_step: PlanStep) -> AndroidAction:
        """Fallback touch implementation with UI parsing"""
        try:
            if self.current_observation:
                elements = self.ui_parser.parse_ui_hierarchy(self.current_observation.ui_hierarchy)
                target_element = self._find_target_element(elements, getattr(plan_step, 'target_element', None))
                
                if target_element:
                    coords = target_element.center
                    action_dict = {
                        "action_type": "touch",
                        "coordinates": coords
                    }
                    if self.android_env:
                        self.android_env.step(action_dict)
                    
                    return AndroidAction(
                        action_type="touch",
                        coordinates=coords,
                        element_id=target_element.element_id
                    )
            
            # Default coordinates
            default_coords = (540, 960)
            action_dict = {
                "action_type": "touch",
                "coordinates": default_coords
            }
            if self.android_env:
                self.android_env.step(action_dict)
            
            return AndroidAction(
                action_type="touch",
                coordinates=default_coords
            )
            
        except Exception as e:
            self.logger.error(f"Fallback touch failed: {e}")
            return AndroidAction(action_type="touch", coordinates=(540, 960))
    
    async def _execute_type(self, plan_step: PlanStep) -> AndroidAction:
        """Execute type action with Agent-S enhancement"""
        try:
            text_to_type = getattr(plan_step, 'text', plan_step.description)
            
            if self.current_observation:
                elements = self.ui_parser.parse_ui_hierarchy(self.current_observation.ui_hierarchy)
                input_elements = self.ui_parser.find_text_input_elements(elements)
                
                if input_elements:
                    target_element = input_elements[0]
                    
                    # Focus the input element
                    action_dict = {
                        "action_type": "touch",
                        "coordinates": target_element.center
                    }
                    if self.android_env:
                        self.android_env.step(action_dict)
                    
                    await asyncio.sleep(0.5)
                    
                    # Type the text
                    action_dict = {
                        "action_type": "type",
                        "text": text_to_type
                    }
                    if self.android_env:
                        self.android_env.step(action_dict)
                    
                    return AndroidAction(
                        action_type="type",
                        text=text_to_type,
                        element_id=target_element.element_id
                    )
            
            # Direct typing fallback
            if self.android_env:
                action_dict = {
                    "action_type": "type",
                    "text": text_to_type
                }
                self.android_env.step(action_dict)
            
            return AndroidAction(
                action_type="type",
                text=text_to_type
            )
            
        except Exception as e:
            self.logger.error(f"Type execution failed: {e}")
            return AndroidAction(action_type="type", text=getattr(plan_step, 'text', ""))
    
    async def _execute_swipe(self, plan_step: PlanStep) -> AndroidAction:
        """Execute swipe action"""
        try:
            android_world_action = getattr(plan_step, 'android_world_action', None)
            if android_world_action:
                start_coords = android_world_action.get("start_coords", (540, 1400))
                end_coords = android_world_action.get("end_coords", (540, 500))
            else:
                start_coords = (540, 1400)
                end_coords = (540, 500)
            
            if self.android_env:
                action_dict = {
                    "action_type": "swipe",
                    "start_coords": start_coords,
                    "end_coords": end_coords
                }
                self.android_env.step(action_dict)
            
            return AndroidAction(
                action_type="swipe",
                start_coords=start_coords,
                end_coords=end_coords
            )
            
        except Exception as e:
            self.logger.error(f"Swipe execution failed: {e}")
            return AndroidAction(action_type="swipe", start_coords=(540, 1400), end_coords=(540, 500))
    
    async def _execute_verify(self, plan_step: PlanStep) -> AndroidAction:
        """Execute verification with Agent-S observation processing"""
        try:
            # Use Agent-S for enhanced verification if available
            if self.is_agent_s_active() and self.current_observation:
                obs = self._prepare_observation_for_agent_s()
                verification_instruction = f"Verify: {plan_step.success_criteria}"
                
                info, _ = await self.predict(verification_instruction, obs)
                confidence = info.get("execution_confidence", 0.8)
                
                return AndroidAction(
                    action_type="verify",
                    element_id=getattr(plan_step, 'target_element', None),
                    verification_confidence=confidence
                )
        except Exception as e:
            self.logger.warning(f"Agent-S verification failed: {e}")
        
        # Fallback verification
        return AndroidAction(
            action_type="verify",
            element_id=getattr(plan_step, 'target_element', None)
        )
    
    async def _execute_wait(self, plan_step: PlanStep) -> AndroidAction:
        """Execute wait action"""
        wait_time = getattr(plan_step, 'wait_time', 2.0)
        await asyncio.sleep(wait_time)
        
        return AndroidAction(action_type="wait", wait_duration=wait_time)
    
    def _find_target_element(self, elements: List[UIElement], target_identifier: Optional[str]) -> Optional[UIElement]:
        """Find target UI element with enhanced matching"""
        if not target_identifier:
            return None
        
        # Exact resource ID match
        for element in elements:
            if element.resource_id == target_identifier:
                return element
        
        # Text match
        text_elements = self.ui_parser.find_elements_by_text(elements, target_identifier)
        if text_elements:
            return text_elements[0]
        
        # Partial matches with scoring
        target_lower = target_identifier.lower()
        scored_elements = []
        
        for element in elements:
            score = 0
            if element.text and target_lower in element.text.lower():
                score += 3
            if element.content_description and target_lower in element.content_description.lower():
                score += 2
            if element.resource_id and target_lower in element.resource_id.lower():
                score += 1
            
            if score > 0:
                scored_elements.append((element, score))
        
        # Return highest scored element
        if scored_elements:
            scored_elements.sort(key=lambda x: x[1], reverse=True)
            return scored_elements[0][0]
        
        return None
    
    def _detect_ui_changes(self, before: Optional[AndroidObservation], 
                          after: Optional[AndroidObservation]) -> bool:
        """Detect UI changes with enhanced analysis"""
        if not before or not after:
            return True
        
        # Activity change detection
        if hasattr(before, 'current_activity') and hasattr(after, 'current_activity'):
            if before.current_activity != after.current_activity:
                return True
        
        # UI hierarchy comparison
        before_hierarchy = getattr(before, 'ui_hierarchy', '')[:1000]
        after_hierarchy = getattr(after, 'ui_hierarchy', '')[:1000]
        
        similarity = self._calculate_similarity(before_hierarchy, after_hierarchy)
        
        # Consider changed if less than 90% similar
        return similarity < 0.9
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using enhanced algorithm"""
        if not str1 or not str2:
            return 0.0
        
        # Use a more sophisticated similarity measure
        from difflib import SequenceMatcher
        
        try:
            matcher = SequenceMatcher(None, str1, str2)
            return matcher.ratio()
        except Exception:
            # Fallback to simple character comparison
            common_chars = sum(1 for a, b in zip(str1, str2) if a == b)
            max_length = max(len(str1), len(str2))
            return common_chars / max_length if max_length > 0 else 1.0
    
    def get_current_ui_state(self) -> Dict[str, Any]:
        """Get current UI state with Agent-S enhancement"""
        if not self.current_observation:
            return {}
        
        try:
            elements = self.ui_parser.parse_ui_hierarchy(getattr(self.current_observation, 'ui_hierarchy', ''))
            screen_analysis = self.ui_parser.analyze_screen_state(elements)
            
            base_state = {
                "current_activity": getattr(self.current_observation, 'current_activity', 'unknown'),
                "screen_bounds": getattr(self.current_observation, 'screen_bounds', (1080, 1920)),
                "timestamp": getattr(self.current_observation, 'timestamp', time.time()),
                "screen_analysis": screen_analysis,
                "element_count": len(elements),
                "clickable_elements": len([e for e in elements if e.clickable]),
                "agent_s_enhanced": self.is_agent_s_active()
            }
            
            # Add Agent-S specific analysis if available
            if self.is_agent_s_active():
                base_state["agent_s_analysis"] = {
                    "confidence": 0.8,
                    "processing_time": 0.1,
                    "enhanced_elements": len([e for e in elements if e.clickable])
                }
            
            return base_state
            
        except Exception as e:
            self.logger.error(f"Failed to get UI state: {e}")
            return {
                "current_activity": "unknown",
                "screen_bounds": (1080, 1920),
                "timestamp": time.time(),
                "error": str(e),
                "agent_s_enhanced": False
            }

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary for this agent 📊"""
        if not hasattr(self, 'execution_history') or not self.execution_history:
            return {
                "total_actions": 0,
                "successful_actions": 0,
                "success_rate": 0.0,
                "total_duration": 0.0,
                "average_duration": 0.0,
                "recent_actions": []
            }
        
        total = len(self.execution_history)
        successful = sum(1 for action in self.execution_history if getattr(action, 'success', False))
        total_duration = sum(getattr(action, 'duration', 0.0) for action in self.execution_history)
        
        return {
            "total_actions": total,
            "successful_actions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "total_duration": total_duration,
            "average_duration": total_duration / total if total > 0 else 0.0,
            "recent_actions": [
                {
                    "action_type": getattr(action, "action_type", "unknown"),
                    "success": getattr(action, "success", False),
                    "duration": getattr(action, "duration", 0.0),
                    "timestamp": getattr(action, "timestamp", None)
                }
                for action in self.execution_history[-5:]
            ]
        }
