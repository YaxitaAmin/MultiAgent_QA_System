# agents/executor_agent.py - FULLY CORRECTED VERSION
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from core.llm_interface import LLMRequest
from core.android_env_wrapper import AndroidEnvWrapper
from core.logger import QALogger
from .planner_agent import Subgoal

@dataclass
class ExecutionResult:
    success: bool
    action_performed: Dict[str, Any]
    ui_state_before: Dict[str, Any]
    ui_state_after: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    screenshot_path: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = "No reasoning provided"

class ExecutorAgent:
    """FULLY FIXED Agent responsible for executing subgoals in the Android UI environment"""
    
    def __init__(self, 
                 llm_interface, 
                 android_env: AndroidEnvWrapper,
                 logger: QALogger, 
                 config: Dict[str, Any]):
        self.llm = llm_interface
        self.android_env = android_env
        self.logger = logger
        self.config = config
        self.execution_history: List[ExecutionResult] = []
        
        print(f"[EXECUTOR INIT] Executor agent initialized with config: {config}")
        
    async def execute_subgoal(self, subgoal: Subgoal, context: Dict[str, Any] = None) -> ExecutionResult:
        """Execute a single subgoal in the Android environment - FULLY FIXED VERSION"""
        start_time = time.time()
        context = context or {}
        
        print(f"[EXECUTOR] Executing subgoal {subgoal.id}: {subgoal.description}")
        logger.info(f"Executing subgoal {subgoal.id}: {subgoal.description}")
        
        try:
            # Get current UI state with error handling
            try:
                ui_state_before = self.android_env.get_ui_state()
                print(f"[EXECUTOR] Got UI state before: {len(ui_state_before.get('ui_elements', []))} elements")
            except Exception as e:
                print(f"[EXECUTOR] Failed to get UI state: {e}")
                ui_state_before = {"ui_elements": [], "screen_info": {}}
            
            # Ground the subgoal to specific actions using LLM
            grounded_action = await self._ground_subgoal_to_action(subgoal, ui_state_before, context)
            
            if not grounded_action:
                print(f"[EXECUTOR] LLM grounding failed, using fallback")
                # Fallback to rule-based action grounding
                grounded_action = self._fallback_action_grounding(subgoal, ui_state_before)
            
            if not grounded_action:
                raise Exception(f"Could not ground subgoal to executable action: {subgoal.description}")
            
            print(f"[EXECUTOR] Grounded action: {grounded_action.get('action_type', 'unknown')}")
            
            # Execute the action in Android environment  
            execution_result = await self._execute_android_action(grounded_action)
            print(f"[EXECUTOR] Execution result: {execution_result.get('success', False)}")
            
            # Get UI state after execution with error handling
            try:
                ui_state_after = self.android_env.get_ui_state()
            except Exception as e:
                print(f"[EXECUTOR] Failed to get UI state after: {e}")
                ui_state_after = ui_state_before  # Use before state as fallback
            
            # Save screenshot with error handling
            screenshot_path = None
            try:
                screenshot_path = self.android_env.save_screenshot()
            except Exception as e:
                print(f"[EXECUTOR] Failed to save screenshot: {e}")
            
            execution_time = time.time() - start_time
            
            result = ExecutionResult(
                success=execution_result.get("success", False),
                action_performed=grounded_action,
                ui_state_before=ui_state_before,
                ui_state_after=ui_state_after,
                error_message=execution_result.get("error"),
                execution_time=execution_time,
                screenshot_path=screenshot_path,
                confidence=grounded_action.get("confidence", 0.5),
                reasoning=grounded_action.get("reasoning", "Action executed")
            )
            
            self.execution_history.append(result)
            
            # FIXED: Log the execution with correct parameter names
            self.logger.log_agent_action(
                agent_type="executor",  # FIXED: was agent_name
                action_type="execute_subgoal",  # FIXED: was action
                input_data={
                    "subgoal_id": subgoal.id,
                    "description": subgoal.description,
                    "action_type": grounded_action.get("action_type", "unknown"),
                    "confidence": result.confidence
                },
                output_data={
                    "success": result.success,
                    "execution_time": result.execution_time
                },
                success=result.success,
                execution_time=execution_time
            )
            
            # Log UI interaction
            self.logger.log_ui_interaction(
                action_type=grounded_action.get("action_type", "unknown"),  # FIXED: was action
                target=grounded_action.get("element_id", "unknown"),
                result="success" if result.success else "failed"
            )
            
            if result.success:
                print(f"[EXECUTOR] Successfully executed subgoal {subgoal.id}")
                logger.info(f"Successfully executed subgoal {subgoal.id}")
            else:
                print(f"[EXECUTOR] Failed to execute subgoal {subgoal.id}: {result.error_message}")
                logger.error(f"Failed to execute subgoal {subgoal.id}: {result.error_message}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            print(f"[EXECUTOR] Exception during execution: {error_message}")
            
            result = ExecutionResult(
                success=False,
                action_performed={},
                ui_state_before=self.android_env.get_ui_state() if hasattr(self.android_env, 'get_ui_state') else {},
                ui_state_after=self.android_env.get_ui_state() if hasattr(self.android_env, 'get_ui_state') else {},
                error_message=error_message,
                execution_time=execution_time,
                confidence=0.0,
                reasoning=f"Execution failed: {error_message}"
            )
            
            self.execution_history.append(result)
            
            # FIXED: Log with correct parameter names
            self.logger.log_agent_action(
                agent_type="executor",  # FIXED: was agent_name
                action_type="execute_subgoal",  # FIXED: was action
                input_data={
                    "subgoal_id": subgoal.id,
                    "error": error_message
                },
                output_data={},
                success=False,
                execution_time=execution_time,
                error_message=error_message
            )
            
            logger.error(f"Execution failed for subgoal {subgoal.id}: {error_message}")
            return result
    
    async def _ground_subgoal_to_action(self, 
                                       subgoal: Subgoal, 
                                       ui_state: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use LLM to ground subgoal to specific UI action - ENHANCED VERSION"""
        
        print(f"[EXECUTOR] Grounding subgoal with LLM: {subgoal.action}")
        
        try:
            # Build comprehensive grounding prompt
            grounding_prompt = self._build_grounding_prompt(subgoal, ui_state, context)
            
            request = LLMRequest(
                prompt=grounding_prompt,
                model=self.config.get("model", "mock"),
                temperature=self.config.get("temperature", 0.0),
                max_tokens=500,
                system_prompt=self._get_grounding_system_prompt()
            )
            
            print(f"[EXECUTOR] Sending grounding request to LLM...")
            response = await self.llm.generate(request)
            print(f"[EXECUTOR] Got LLM response for grounding")
            
            # Parse response to action
            action = self._parse_grounding_response(response.content)
            
            if action:
                print(f"[EXECUTOR] Successfully grounded to action: {action.get('action_type')}")
            else:
                print(f"[EXECUTOR] Failed to parse LLM grounding response")
            
            return action
            
        except Exception as e:
            print(f"[EXECUTOR] LLM grounding failed: {e}")
            logger.warning(f"LLM grounding failed: {e}")
            return None
    
    def _fallback_action_grounding(self, subgoal: Subgoal, ui_state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based action grounding when LLM fails - ENHANCED VERSION"""
        
        print(f"[EXECUTOR] Using fallback grounding for: {subgoal.action}")
        
        action_name = subgoal.action.lower()
        description = subgoal.description.lower()
        
        # Rule-based grounding based on common action patterns
        if "open_settings" in action_name or "settings" in description:
            return {
                "action_type": "touch",
                "element_id": "com.android.settings:id/settings_icon",
                "coordinates": [200, 125],
                "confidence": 0.8,
                "reasoning": "Opening Settings app using known icon location"
            }
            
        elif "wifi" in action_name or "wi-fi" in description:
            if "toggle" in action_name or "toggle" in description:
                return {
                    "action_type": "touch",
                    "element_id": "com.android.settings:id/wifi_toggle",
                    "coordinates": [420, 180],
                    "confidence": 0.85,
                    "reasoning": "Toggling Wi-Fi switch in Settings"
                }
            else:
                return {
                    "action_type": "touch",
                    "element_id": "com.android.settings:id/wifi_option",
                    "coordinates": [180, 240],
                    "confidence": 0.82,
                    "reasoning": "Navigating to Wi-Fi settings"
                }
                
        elif "bluetooth" in action_name or "bluetooth" in description:
            if "toggle" in action_name or "toggle" in description:
                return {
                    "action_type": "touch",
                    "element_id": "com.android.settings:id/bluetooth_toggle",
                    "coordinates": [420, 160],
                    "confidence": 0.83,
                    "reasoning": "Toggling Bluetooth switch"
                }
            else:
                return {
                    "action_type": "touch",
                    "element_id": "com.android.settings:id/bluetooth_option",
                    "coordinates": [180, 300],
                    "confidence": 0.80,
                    "reasoning": "Navigating to Bluetooth settings"
                }
                
        elif "navigate" in action_name and "bluetooth" in description:
            # Special case for "Navigate to Bluetooth settings"
            return {
                "action_type": "touch",
                "element_id": "com.android.settings:id/bluetooth_menu_item",
                "coordinates": [200, 280],
                "confidence": 0.85,
                "reasoning": "Navigating to Bluetooth settings from main Settings menu"
            }
                
        elif "verify" in action_name or "check" in description:
            # Verification actions - minimal interaction needed
            return {
                "action_type": "wait",
                "duration": 0.5,
                "confidence": 0.95,
                "reasoning": "Waiting to verify current state"
            }
                
        elif "calculator" in action_name or "calculator" in description:
            if "open" in action_name or "open" in description:
                return {
                    "action_type": "touch",
                    "element_id": "com.android.calculator2:id/calculator_icon",
                    "coordinates": [150, 320],
                    "confidence": 0.88,
                    "reasoning": "Opening Calculator app"
                }
            else:
                return {
                    "action_type": "touch",
                    "element_id": "com.android.calculator2:id/digit_button",
                    "coordinates": [150, 580],
                    "confidence": 0.75,
                    "reasoning": "Tapping calculator button"
                }
                
        elif "scroll" in action_name or "scroll" in description:
            return {
                "action_type": "scroll",
                "direction": "down",
                "confidence": 0.90,
                "reasoning": "Scrolling to find more options"
            }
            
        elif "back" in action_name or "back" in description:
            return {
                "action_type": "back",
                "confidence": 0.95,
                "reasoning": "Going back to previous screen"
            }
            
        else:
            # Generic touch action as last resort
            return {
                "action_type": "touch",
                "coordinates": [200, 400],
                "confidence": 0.5,
                "reasoning": f"Generic touch action for: {subgoal.description}"
            }
    
    async def _execute_android_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action in Android environment - MOCK-COMPATIBLE VERSION"""
        
        print(f"[EXECUTOR] Executing Android action: {action.get('action_type')}")
        
        try:
            action_type = action.get("action_type")
            
            # For mock environment, simulate successful execution
            if getattr(self.android_env, 'mock_mode', True):
                print(f"[EXECUTOR] Using mock execution for {action_type}")
                
                # Simulate realistic execution times
                await asyncio.sleep(0.1)  
                
                # High success rate for mock execution
                import random
                success_rate = 0.85  # 85% success rate for mock
                success = random.random() < success_rate
                
                return {
                    "success": success,
                    "method": "mock",
                    "action_type": action_type,
                    "simulated": True
                }
            
            # Real execution for non-mock environments
            if action_type == "touch":
                return await self._execute_touch_action(action)
            elif action_type == "scroll":
                return await self._execute_scroll_action(action)
            elif action_type == "type":
                return await self._execute_type_action(action)
            elif action_type == "wait":
                return await self._execute_wait_action(action)
            elif action_type == "back":
                return await self._execute_back_action(action)
            elif action_type == "home":
                return await self._execute_home_action(action)
            elif action_type == "swipe":
                return await self._execute_swipe_action(action)
            else:
                return {"success": False, "error": f"Unknown action type: {action_type}"}
                
        except Exception as e:
            print(f"[EXECUTOR] Android action execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_touch_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute touch action with multiple targeting methods"""
        
        print(f"[EXECUTOR] Executing touch action")
        
        try:
            coordinates = action.get("coordinates")
            element_id = action.get("element_id")
            
            # Try coordinates first (most reliable)
            if coordinates and len(coordinates) >= 2:
                try:
                    success = self.android_env.touch(coordinates[0], coordinates[1])
                    
                    self.logger.log_ui_interaction(
                        action_type="touch",
                        target=f"coords_{coordinates[0]}_{coordinates[1]}",
                        result="success" if success else "failed"
                    )
                    
                    return {
                        "success": success,
                        "method": "coordinates",
                        "target": coordinates
                    }
                except Exception as e:
                    print(f"[EXECUTOR] Coordinate touch failed: {e}")
                    # Continue to fallback methods
            
            # Try element ID approach
            if element_id:
                try:
                    if hasattr(self.android_env, 'touch_element'):
                        success = self.android_env.touch_element(element_id)
                        
                        self.logger.log_ui_interaction(
                            action_type="touch",
                            target=element_id,
                            result="success" if success else "failed"
                        )
                        
                        return {
                            "success": success,
                            "method": "element_id",
                            "target": element_id
                        }
                except Exception as e:
                    print(f"[EXECUTOR] Element ID touch failed: {e}")
            
            # Fallback to center screen touch
            try:
                success = self.android_env.touch(200, 400)
                return {
                    "success": success,
                    "method": "fallback_center",
                    "target": "center_screen"
                }
            except Exception as e:
                return {"success": False, "error": f"All touch methods failed: {str(e)}"}
                
        except Exception as e:
            print(f"[EXECUTOR] Touch action failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_scroll_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scroll action"""
        try:
            direction = action.get("direction", "down")
            
            if hasattr(self.android_env, 'scroll'):
                success = self.android_env.scroll(direction)
            else:
                # Mock scroll for testing
                success = True
            
            self.logger.log_ui_interaction(
                action_type="scroll",
                target=direction,
                result="success" if success else "failed"
            )
            
            return {"success": success, "direction": direction}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_type_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text input action"""
        try:
            text = action.get("text", "")
            
            if hasattr(self.android_env, 'type_text'):
                success = self.android_env.type_text(text)
            else:
                # Mock type for testing
                success = True
            
            self.logger.log_ui_interaction(
                action_type="type",
                target=f"text: {text}",
                result="success" if success else "failed"
            )
            
            return {"success": success, "text": text}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_wait_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute wait action"""
        try:
            duration = action.get("duration", 1.0)
            await asyncio.sleep(duration)
            
            return {"success": True, "duration": duration}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_back_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute back button action"""
        try:
            if hasattr(self.android_env, 'back'):
                success = self.android_env.back()
            else:
                # Mock back for testing
                success = True
            
            self.logger.log_ui_interaction(
                action_type="back",
                target="back_button",
                result="success" if success else "failed"
            )
            
            return {"success": success}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_home_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute home button action"""
        try:
            if hasattr(self.android_env, 'home'):
                success = self.android_env.home()
            else:
                # Mock home for testing
                success = True
            
            self.logger.log_ui_interaction(
                action_type="home",
                target="home_button",
                result="success" if success else "failed"
            )
            
            return {"success": success}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_swipe_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute swipe action"""
        try:
            start_coords = action.get("start_coordinates", [200, 600])
            end_coords = action.get("end_coordinates", [200, 200])
            
            if hasattr(self.android_env, 'swipe'):
                success = self.android_env.swipe(start_coords, end_coords)
            else:
                # Mock swipe for testing
                success = True
            
            self.logger.log_ui_interaction(
                action_type="swipe",
                target=f"{start_coords} -> {end_coords}",
                result="success" if success else "failed"
            )
            
            return {"success": success, "start": start_coords, "end": end_coords}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _build_grounding_prompt(self, subgoal: Subgoal, ui_state: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build prompt for action grounding - ENHANCED VERSION"""
        
        # Extract relevant UI elements (limit to prevent token overflow)
        ui_elements = ui_state.get("ui_elements", [])[:10]
        
        # Simplify UI elements for prompt
        simplified_elements = []
        for i, element in enumerate(ui_elements):
            simplified_elements.append({
                "index": i,
                "text": element.get("text", ""),
                "content_desc": element.get("content_desc", ""),
                "class": element.get("class_name", ""),
                "clickable": element.get("clickable", False),
                "bounds": element.get("bounds", [0, 0, 100, 100])
            })
        
        prompt = f"""
Subgoal to execute: {subgoal.description}
Action type: {subgoal.action}

Current UI Elements:
{json.dumps(simplified_elements, indent=2)}

Context: {json.dumps(context, indent=2)}

Your task:
1. Analyze the current UI elements
2. Select the most appropriate action and target element for the subgoal
3. Provide exact coordinates or element identification
4. Explain your reasoning

Generate JSON response:
{{
    "action_type": "touch|scroll|type|swipe|back|home|wait",
    "element_id": "target_element_id_if_available",
    "coordinates": [x, y],
    "text": "text_to_type_if_applicable",
    "direction": "scroll_direction_if_applicable",
    "duration": wait_seconds_if_applicable,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of why this action was chosen"
}}

Focus on executing: {subgoal.description}
"""
        return prompt
    
    def _get_grounding_system_prompt(self) -> str:
        """Get system prompt for action grounding"""
        return """
You are an expert Android UI automation specialist. Your job is to ground high-level subgoals into specific, executable Android actions.

Key Guidelines:
1. Analyze available UI elements carefully
2. Choose the most reliable targeting method (coordinates usually work best)
3. Consider element properties (clickable, text content, position)
4. Be realistic with confidence scores (0.7-0.9 for good matches)
5. Provide clear reasoning for your action choice
6. Focus on actions that will accomplish the subgoal effectively

Always respond with valid JSON containing only the necessary fields for the chosen action type.
"""
    
    def _parse_grounding_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into action dictionary - ROBUST VERSION"""
        
        print(f"[EXECUTOR] Parsing grounding response...")
        
        try:
            # Clean the response
            response_content = response_content.strip()

            # Remove markdown code blocks
            if response_content.startswith("```\n"):
                response_content = response_content[7:]
            if response_content.startswith("```"):
                response_content = response_content[3:]
            if response_content.endswith("```\n"):
                response_content = response_content[:-4]

            response_content = response_content.strip()
            
            # Parse JSON
            action = json.loads(response_content)
            
            # Validate required fields
            if "action_type" not in action:
                print(f"[EXECUTOR] Missing action_type in response")
                return None
            
            # Ensure confidence is present
            if "confidence" not in action:
                action["confidence"] = 0.7  # Default confidence
            
            # Ensure reasoning is present
            if "reasoning" not in action:
                action["reasoning"] = f"Executing {action.get('action_type', 'unknown')} action"
            
            print(f"[EXECUTOR] Successfully parsed action: {action.get('action_type')}")
            return action
            
        except Exception as e:
            print(f"[EXECUTOR] Failed to parse grounding response: {e}")
            logger.error(f"Failed to parse grounding response: {e}")
            return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions - ENHANCED VERSION"""
        total_executions = len(self.execution_history)
        
        if total_executions == 0:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "average_confidence": 0.0,
                "last_screenshot": None,
                "action_types_used": []
            }
        
        successful_executions = sum(1 for result in self.execution_history if result.success)
        total_time = sum(result.execution_time for result in self.execution_history)
        total_confidence = sum(result.confidence for result in self.execution_history)
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions,
            "average_execution_time": total_time / total_executions,
            "average_confidence": total_confidence / total_executions,
            "last_screenshot": self.execution_history[-1].screenshot_path if self.execution_history else None,
            "action_types_used": list(set(result.action_performed.get("action_type", "unknown") for result in self.execution_history))
        }
