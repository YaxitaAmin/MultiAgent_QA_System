"""
Executor Agent - Integrates with Agent-S and android_world
Executes subgoals in the Android UI environment with grounded mobile gestures
"""

import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from agents.base_agents import BaseQAAgent, MessageType  # ✅ CORRECTED IMPORT
from agents.planner_agent import PlanStep, QAPlan
from core.android_env_wrapper import AndroidEnvWrapper, AndroidAction, AndroidObservation
from core.ui_utils import UIParser, UIElement, ActionResult
from core.logger import QALogger
from config.default_config import config

@dataclass
class ExecutionResult:
    """Result of executing a plan step"""
    step_id: int
    success: bool
    action_taken: Optional[AndroidAction]
    observation_before: Optional[AndroidObservation]
    observation_after: Optional[AndroidObservation]
    execution_time: float
    error_message: Optional[str] = None
    ui_changes_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return asdict(self)

class ExecutorAgent(BaseQAAgent):
    """
    Agent-S compatible Executor Agent
    Executes QA test steps using android_world environment
    """
    
    def __init__(self):
        super().__init__("ExecutorAgent")
        self.ui_parser = UIParser()
        self.android_env = None
        self.current_observation = None
        self.execution_results = []
        
        self.logger.info("ExecutorAgent initialized with android_world integration")
    
    async def initialize_android_env(self, task_name: str) -> bool:
        """Initialize android_world environment"""
        try:
            self.android_env = AndroidEnvWrapper(task_name=task_name)
            self.current_observation = self.android_env.reset()
            
            self.logger.info(f"Android environment initialized for task: {task_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Android environment: {e}")
            return False
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process execution task - CORRECTED to include action tracking ✅"""
        start_time = time.time()
        
        try:
            plan_step = task_data.get("plan_step")
            if not plan_step:
                raise ValueError("No plan step provided")
            
            # Initialize environment if needed 📱
            if not self.android_env:
                task_name = task_data.get("android_world_task", "settings_wifi")
                if not await self.initialize_android_env(task_name):
                    raise Exception("Failed to initialize Android environment")
            
            # Execute the step 🏃‍♂️💨
            result = await self.execute_step(plan_step)
            
            duration = time.time() - start_time
            
            # ✅ CORRECTED: Create and store the action record
            action_record = self.log_action(
                "execute_step",
                {"step_id": plan_step.step_id, "action_type": plan_step.action_type},
                {"success": result.success, "ui_changes": result.ui_changes_detected},
                result.success,
                duration,
                result.error_message
            )
            
            # ✅ CORRECTED: Return the action record so it can be collected
            return {
                "success": result.success,
                "execution_result": {
                    "step_id": result.step_id,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "ui_changes_detected": result.ui_changes_detected,
                    "error_message": result.error_message,
                    "action_taken": result.action_taken.__dict__ if result.action_taken else None
                },
                "current_observation": self.current_observation,
                "action_record": action_record  # ✅ Include the action record
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process execution task: {e}")
            
            # ✅ CORRECTED: Create action record for failed execution
            action_record = self.log_action(
                "execute_step",
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
                    "execution_time": time.time() - start_time,
                    "ui_changes_detected": False,
                    "error_message": str(e),
                    "action_taken": None
                },
                "current_observation": self.current_observation,
                "error": str(e),
                "action_record": action_record  # ✅ Include the failed action record
            }
    
    async def execute_step(self, plan_step: PlanStep) -> ExecutionResult:
        """Execute a single plan step using Agent-S and android_world"""
        start_time = time.time()
        observation_before = self.current_observation
        
        try:
            self.logger.info(f"Executing step {plan_step.step_id}: {plan_step.description}")
            
            # Execute action based on step type
            action_taken = None
            if plan_step.action_type == "touch":
                action_taken = await self._execute_touch(plan_step)
            elif plan_step.action_type == "type":
                action_taken = await self._execute_type(plan_step)
            elif plan_step.action_type == "swipe":
                action_taken = await self._execute_swipe(plan_step)
            elif plan_step.action_type == "verify":
                action_taken = await self._execute_verify(plan_step)
            elif plan_step.action_type == "wait":
                action_taken = await self._execute_wait(plan_step)
            else:
                # Default to touch action
                action_taken = await self._execute_touch(plan_step)
            
            # Get new observation
            if self.android_env:
                self.current_observation = self.android_env._get_observation()
            
            # Check for UI changes
            ui_changes = self._detect_ui_changes(observation_before, self.current_observation)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                step_id=plan_step.step_id,
                success=True,
                action_taken=action_taken,
                observation_before=observation_before,
                observation_after=self.current_observation,
                execution_time=execution_time,
                ui_changes_detected=ui_changes
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
                ui_changes_detected=False
            )
    
    async def _execute_touch(self, plan_step: PlanStep) -> AndroidAction:
        """Execute touch action using Agent-S grounding"""
        try:
            # Use Agent-S for intelligent touch target identification
            if self.agent_s and self.current_observation and not config.USE_MOCK_LLM:
                return await self._agent_s_touch(plan_step)
            else:
                return await self._fallback_touch(plan_step)
                
        except Exception as e:
            self.logger.error(f"Touch execution failed: {e}")
            return await self._fallback_touch(plan_step)
    
    async def _agent_s_touch(self, plan_step: PlanStep) -> AndroidAction:
        """Use Agent-S for intelligent touch execution"""
        try:
            # Prepare observation for Agent-S
            import io
            from PIL import Image
            import numpy as np
            
            screenshot = self.current_observation.screenshot
            img = Image.fromarray(screenshot)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            screenshot_bytes = buffered.getvalue()
            
            obs = {"screenshot": screenshot_bytes}
            
            # Create instruction for Agent-S
            instruction = f"Touch the UI element: {getattr(plan_step, 'target_element', '') or plan_step.description}"
            
            # Get Agent-S prediction
            info, action = self.agent_s.predict(instruction=instruction, observation=obs)
            
            # Execute the action code from Agent-S
            if action and len(action) > 0:
                action_code = action[0]
                # Parse pyautogui action to get coordinates
                coords = self._parse_pyautogui_click(action_code)
                
                if coords:
                    # Execute touch via android_world
                    android_action = AndroidAction(
                        action_type="touch",
                        coordinates=coords
                    )
                    
                    action_dict = {
                        "action_type": "touch",
                        "coordinates": coords
                    }
                    
                    self.android_env.step(action_dict)
                    return android_action
            
            # Fallback if Agent-S didn't provide usable action
            return await self._fallback_touch(plan_step)
            
        except Exception as e:
            self.logger.error(f"Agent-S touch failed: {e}")
            return await self._fallback_touch(plan_step)
    
    def _parse_pyautogui_click(self, action_code: str) -> Optional[Tuple[int, int]]:
        """Parse pyautogui click coordinates from Agent-S action code"""
        try:
            # Look for pyautogui.click(x, y) pattern
            import re
            
            # Pattern to match pyautogui.click(x, y)
            pattern = r'pyautogui\.click\((\d+),\s*(\d+)\)'
            match = re.search(pattern, action_code)
            
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return (x, y)
            
            # Pattern to match click with named arguments
            pattern = r'pyautogui\.click\(x=(\d+),\s*y=(\d+)\)'
            match = re.search(pattern, action_code)
            
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return (x, y)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to parse pyautogui coordinates: {e}")
            return None
    
    async def _fallback_touch(self, plan_step: PlanStep) -> AndroidAction:
        """Fallback touch implementation"""
        try:
            # Parse UI to find target element
            if self.current_observation:
                elements = self.ui_parser.parse_ui_hierarchy(self.current_observation.ui_hierarchy)
                
                # Find target element
                target_element = self._find_target_element(elements, getattr(plan_step, 'target_element', None))
                
                if target_element:
                    coords = target_element.center
                    action_dict = {
                        "action_type": "touch",
                        "coordinates": coords
                    }
                    self.android_env.step(action_dict)
                    
                    return AndroidAction(
                        action_type="touch",
                        coordinates=coords,
                        element_id=target_element.element_id
                    )
                else:
                    # Use default coordinates if element not found
                    default_coords = (540, 960)  # Center of typical Android screen
                    action_dict = {
                        "action_type": "touch", 
                        "coordinates": default_coords
                    }
                    self.android_env.step(action_dict)
                    
                    return AndroidAction(
                        action_type="touch",
                        coordinates=default_coords
                    )
            else:
                # No observation available, use default coords
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
            # Final fallback - create a basic action without executing
            return AndroidAction(action_type="touch", coordinates=(540, 960))
    
    async def _execute_type(self, plan_step: PlanStep) -> AndroidAction:
        """Execute type action"""
        try:
            text_to_type = getattr(plan_step, 'text', plan_step.description)
            
            # Find text input element
            if self.current_observation:
                elements = self.ui_parser.parse_ui_hierarchy(self.current_observation.ui_hierarchy)
                input_elements = self.ui_parser.find_text_input_elements(elements)
                
                if input_elements:
                    target_element = input_elements[0]  # Use first input element
                    
                    # First tap to focus
                    action_dict = {
                        "action_type": "touch",
                        "coordinates": target_element.center
                    }
                    self.android_env.step(action_dict)
                    
                    # Wait for focus
                    await asyncio.sleep(0.5)
                    
                    # Then type text
                    action_dict = {
                        "action_type": "type",
                        "text": text_to_type
                    }
                    self.android_env.step(action_dict)
                    
                    return AndroidAction(
                        action_type="type",
                        text=text_to_type,
                        element_id=target_element.element_id
                    )
            
            # Fallback: just try to type
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
            # Get swipe coordinates from android_world_action or use defaults
            android_world_action = getattr(plan_step, 'android_world_action', None)
            if android_world_action:
                start_coords = android_world_action.get("start_coords", (540, 1400))
                end_coords = android_world_action.get("end_coords", (540, 500))
            else:
                # Default vertical swipe (scroll up)
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
        """Execute verification - no actual action needed"""
        # Verification is passive - just record the attempt
        return AndroidAction(
            action_type="verify",
            element_id=getattr(plan_step, 'target_element', None)
        )
    
    async def _execute_wait(self, plan_step: PlanStep) -> AndroidAction:
        """Execute wait action"""
        wait_time = getattr(plan_step, 'wait_time', 2.0)
        await asyncio.sleep(wait_time)
        
        return AndroidAction(action_type="wait")
    
    def _find_target_element(self, elements: List[UIElement], target_identifier: Optional[str]) -> Optional[UIElement]:
        """Find target UI element"""
        if not target_identifier:
            return None
        
        # Try exact resource ID match
        for element in elements:
            if element.resource_id == target_identifier:
                return element
        
        # Try text match
        text_elements = self.ui_parser.find_elements_by_text(elements, target_identifier)
        if text_elements:
            return text_elements[0]
        
        # Try partial matches
        target_lower = target_identifier.lower()
        for element in elements:
            if element.text and target_lower in element.text.lower():
                return element
            if element.content_description and target_lower in element.content_description.lower():
                return element
            if element.resource_id and target_lower in element.resource_id.lower():
                return element
        
        return None
    
    def _detect_ui_changes(self, before: Optional[AndroidObservation], 
                          after: Optional[AndroidObservation]) -> bool:
        """Detect if UI has changed between observations"""
        if not before or not after:
            return True
        
        # Compare activity names
        if hasattr(before, 'current_activity') and hasattr(after, 'current_activity'):
            if before.current_activity != after.current_activity:
                return True
        
        # Compare UI hierarchy (simplified)
        before_hierarchy = getattr(before, 'ui_hierarchy', '')[:1000]  # Compare first 1000 chars
        after_hierarchy = getattr(after, 'ui_hierarchy', '')[:1000]
        
        similarity = self._calculate_similarity(before_hierarchy, after_hierarchy)
        return similarity < 0.9  # Consider changed if less than 90% similar
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity"""
        if not str1 or not str2:
            return 0.0
        
        # Simple character-based similarity
        common_chars = sum(1 for a, b in zip(str1, str2) if a == b)
        max_length = max(len(str1), len(str2))
        
        return common_chars / max_length if max_length > 0 else 1.0
    
    def get_current_ui_state(self) -> Dict[str, Any]:
        """Get current UI state information"""
        if not self.current_observation:
            return {}
        
        try:
            elements = self.ui_parser.parse_ui_hierarchy(getattr(self.current_observation, 'ui_hierarchy', ''))
            screen_analysis = self.ui_parser.analyze_screen_state(elements)
            
            return {
                "current_activity": getattr(self.current_observation, 'current_activity', 'unknown'),
                "screen_bounds": getattr(self.current_observation, 'screen_bounds', (1080, 1920)),
                "timestamp": getattr(self.current_observation, 'timestamp', time.time()),
                "screen_analysis": screen_analysis,
                "element_count": len(elements),
                "clickable_elements": len([e for e in elements if e.clickable])
            }
        except Exception as e:
            self.logger.error(f"Failed to get UI state: {e}")
            return {
                "current_activity": "unknown",
                "screen_bounds": (1080, 1920),
                "timestamp": time.time(),
                "error": str(e)
            }
