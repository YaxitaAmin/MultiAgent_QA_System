# core/llm_interface.py - ENHANCED VERSION WITH BETTER ERROR HANDLING
import os
import json
import hashlib
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from loguru import logger

@dataclass
class LLMRequest:
    prompt: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1000
    system_prompt: Optional[str] = None

@dataclass
class LLMResponse:
    content: str
    usage_tokens: int = 0
    cached: bool = False
    timestamp: float = 0.0
    confidence: float = 0.8  # Added for compatibility
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# ✅ MISSING: Abstract base class that other modules expect
class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def analyze_ui_state(self, ui_hierarchy: str, screenshot_path: str, task_context: str) -> LLMResponse:
        """Analyze UI state and provide insights"""
        pass
    
    @abstractmethod
    def plan_decomposition(self, high_level_goal: str, current_state: str) -> List[Dict[str, Any]]:
        """Decompose high-level goal into actionable steps"""
        pass

class CostEfficientLLMInterface(LLMInterface):
    """Cost-efficient LLM interface with caching and batching for Gemini free tier"""
    
    def __init__(self, api_key: str, cache_dir: str = "cache"):
        try:
            if GEMINI_AVAILABLE and api_key:
                genai.configure(api_key=api_key)
                self.api_configured = True
            else:
                self.api_configured = False
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            self.api_configured = False
            
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.request_count = 0
        self.daily_limit = 1500
        self.rate_limit_delay = 1.0
        self.last_request_time = 0.0
        
    def _get_cache_key(self, request: LLMRequest) -> str:
        content = f"{request.prompt}_{request.model}_{request.temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[LLMResponse]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return LLMResponse(
                    content=data['content'],
                    usage_tokens=data['usage_tokens'],
                    cached=True,
                    timestamp=data['timestamp']
                )
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, response: LLMResponse):
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'content': response.content,
                    'usage_tokens': response.usage_tokens,
                    'timestamp': response.timestamp
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _enforce_rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate response using Gemini API - implements LLMInterface"""
        request = LLMRequest(prompt=prompt, model="gemini-1.5-flash")
        
        # Use async generate internally
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate(request))
    
    def analyze_ui_state(self, ui_hierarchy: str, screenshot_path: str, task_context: str) -> LLMResponse:
        """Analyze UI state - implements LLMInterface"""
        prompt = f"""
        Analyze the Android UI state for QA testing.
        
        Task Context: {task_context}
        UI Hierarchy: {ui_hierarchy[:1000]}
        
        Provide analysis in JSON format with:
        - description: Current screen description
        - elements: Available interactive elements
        - actions: Suggested next actions
        - issues: Any detected issues
        """
        return self.generate_response(prompt)
    
    def plan_decomposition(self, high_level_goal: str, current_state: str) -> List[Dict[str, Any]]:
        """Decompose goal into actionable steps - implements LLMInterface"""
        prompt = f"""
        Create a detailed test plan for: {high_level_goal}
        Current State: {current_state}
        
        Provide a JSON list of steps with:
        - step: step number
        - action: action type (touch, type, verify, etc.)
        - description: what to do
        - success_criteria: how to verify success
        """
        
        response = self.generate_response(prompt)
        try:
            steps = json.loads(response.content)
            return steps if isinstance(steps, list) else []
        except json.JSONDecodeError:
            return [{"step": 1, "action": "touch", "description": "Execute task", "success_criteria": "Task completed"}]
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        cache_key = self._get_cache_key(request)
        
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            logger.info(f"Using cached response for {request.model}")
            return cached_response
        
        if not self.api_configured:
            logger.warning("API not configured, returning mock response")
            return LLMResponse(
                content="MOCK_RESPONSE_API_NOT_CONFIGURED",
                usage_tokens=0,
                cached=False,
                timestamp=time.time()
            )
        
        if self.request_count >= self.daily_limit:
            logger.error("Daily API limit reached")
            return LLMResponse(
                content="MOCK_RESPONSE_DUE_TO_RATE_LIMIT",
                usage_tokens=0,
                cached=False,
                timestamp=time.time()
            )
        
        self._enforce_rate_limit()
        
        try:
            # ✅ FIXED: Handle both old and new model names
            model_name = request.model
            if model_name == "gemini-pro":
                model_name = "gemini-1.5-flash"  # Auto-upgrade deprecated model
                logger.info(f"Auto-upgraded model from gemini-pro to {model_name}")
            
            model = genai.GenerativeModel(model_name)
            
            full_prompt = request.prompt
            if request.system_prompt:
                full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
            
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens,
                )
            )
            
            self.request_count += 1
            
            llm_response = LLMResponse(
                content=response.text,
                usage_tokens=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 100,
                cached=False,
                timestamp=time.time()
            )
            
            self._save_to_cache(cache_key, llm_response)
            
            logger.info(f"Generated response using {model_name} (tokens: {llm_response.usage_tokens})")
            return llm_response
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return LLMResponse(
                content=f"ERROR_MOCK_RESPONSE: {str(e)[:100]}",
                usage_tokens=0,
                cached=False,
                timestamp=time.time()
            )
    
    def batch_generate(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        responses = []
        for request in requests:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(self.generate(request))
            responses.append(response)
            time.sleep(0.5)
        return responses

class MockLLMInterface(LLMInterface):
    """ENHANCED Mock LLM interface with better agent detection and responses"""
    
    def __init__(self):
        self.call_count = 0
        
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate response - implements LLMInterface"""
        request = LLMRequest(prompt=prompt, model="mock")
        
        # Use async generate internally
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate(request))
    
    def analyze_ui_state(self, ui_hierarchy: str, screenshot_path: str, task_context: str) -> LLMResponse:
        """Mock UI state analysis"""
        analysis = {
            "description": "Mock analysis of Android UI state",
            "elements": [
                {"type": "button", "text": "Settings", "id": "settings_btn"},
                {"type": "toggle", "text": "Wi-Fi", "id": "wifi_toggle"}
            ],
            "actions": [
                {"action": "touch", "target": "settings_btn", "description": "Open settings"}
            ],
            "issues": []
        }
        
        return LLMResponse(
            content=json.dumps(analysis),
            usage_tokens=50,
            confidence=0.8,
            timestamp=time.time()
        )
    
    def plan_decomposition(self, high_level_goal: str, current_state: str) -> List[Dict[str, Any]]:
        """Mock plan decomposition"""
        if "wifi" in high_level_goal.lower():
            return [
                {"step": 1, "action": "touch", "description": "Open Settings", "success_criteria": "Settings app opens"},
                {"step": 2, "action": "touch", "description": "Tap Wi-Fi option", "success_criteria": "Wi-Fi settings visible"},
                {"step": 3, "action": "touch", "description": "Toggle Wi-Fi switch", "success_criteria": "Wi-Fi state changes"},
                {"step": 4, "action": "verify", "description": "Verify Wi-Fi is toggled", "success_criteria": "Wi-Fi shows new state"}
            ]
        else:
            return [
                {"step": 1, "action": "touch", "description": "Execute main action", "success_criteria": "Action completed"},
                {"step": 2, "action": "verify", "description": "Verify result", "success_criteria": "Expected outcome achieved"}
            ]
        
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate realistic, properly structured responses for each agent type"""
        self.call_count += 1
        
        prompt_text = request.prompt.lower()
        system_text = (request.system_prompt or "").lower()
        
        # Create deterministic but varied responses based on content
        content_hash = hashlib.md5(request.prompt.encode()).hexdigest()
        task_seed = int(content_hash[:8], 16) % 1000
        
        import random
        random.seed(task_seed)
        
        print(f"[MOCK LLM] Call #{self.call_count} - Model: {request.model} - Seed: {task_seed}")
        print(f"[MOCK LLM] First 80 chars: {request.prompt[:80]}...")
        
        # ✅ ENHANCED: Better agent detection patterns
        if any(pattern in prompt_text for pattern in [
            "create a detailed test plan", "test plan", "break down", "subgoals", 
            "decompose", "actionable subgoals", "sequence of actionable", "planning"
        ]) or "goal:" in prompt_text.lower():
            print("[MOCK LLM] ✅ PLANNER request detected")
            return self._generate_executable_plan(prompt_text, random, request.prompt)
        
        elif ("android ui automation specialist" in system_text or 
              "execute subgoal" in prompt_text or
              "subgoal to execute" in prompt_text or
              "current ui elements" in prompt_text or
              "grounding" in prompt_text):
            print("[MOCK LLM] ✅ EXECUTOR request detected")  
            return self._generate_executable_action(prompt_text, random, request.prompt)
        
        elif any(pattern in prompt_text for pattern in [
            "verify", "verification", "expected", "actual", "passed", "failed",
            "execution result", "meets expectations", "validate"
        ]):
            print("[MOCK LLM] ✅ VERIFIER request detected")
            return self._generate_verification_result(prompt_text, random)
        
        elif any(pattern in prompt_text for pattern in [
            "analyze", "analysis", "episode", "improvement", "supervisor",
            "test trace", "full test trace", "review"
        ]):
            print("[MOCK LLM] ✅ SUPERVISOR request detected")
            return self._generate_supervisor_analysis(prompt_text, random)
        
        else:
            print("[MOCK LLM] ⚠️ FALLBACK - treating as executor")
            return self._generate_executable_action(prompt_text, random, request.prompt)
    
    def _generate_executable_plan(self, prompt_text: str, random_gen, full_prompt: str) -> LLMResponse:
        """Generate executable plans that agents can actually follow"""
        
        print(f"[MOCK LLM] Generating plan for: {prompt_text[:50]}...")
        
        # Determine task type from the prompt
        if "wifi" in prompt_text or "wi-fi" in prompt_text:
            subgoals = [
                {
                    "id": 1, 
                    "action": "open_settings", 
                    "description": "Open Settings app from home screen", 
                    "priority": 1, 
                    "dependencies": []
                },
                {
                    "id": 2, 
                    "action": "navigate_wifi", 
                    "description": "Find and tap Wi-Fi option in Settings", 
                    "priority": 1, 
                    "dependencies": [1]
                },
                {
                    "id": 3, 
                    "action": "toggle_wifi", 
                    "description": "Toggle Wi-Fi switch to change state", 
                    "priority": 1, 
                    "dependencies": [2]
                },
                {
                    "id": 4, 
                    "action": "verify_wifi_state", 
                    "description": "Verify Wi-Fi state has changed successfully", 
                    "priority": 1, 
                    "dependencies": [3]
                }
            ]
            
        elif "calculator" in prompt_text or "calculation" in prompt_text:
            subgoals = [
                {
                    "id": 1, 
                    "action": "open_app_drawer", 
                    "description": "Open app drawer from home screen", 
                    "priority": 1, 
                    "dependencies": []
                },
                {
                    "id": 2, 
                    "action": "find_calculator", 
                    "description": "Locate Calculator app in app drawer", 
                    "priority": 1, 
                    "dependencies": [1]
                },
                {
                    "id": 3, 
                    "action": "open_calculator", 
                    "description": "Tap Calculator app to open it", 
                    "priority": 1, 
                    "dependencies": [2]
                },
                {
                    "id": 4, 
                    "action": "perform_calculation", 
                    "description": "Enter numbers and perform basic calculation", 
                    "priority": 1, 
                    "dependencies": [3]
                },
                {
                    "id": 5, 
                    "action": "verify_result", 
                    "description": "Verify calculation result is displayed", 
                    "priority": 1, 
                    "dependencies": [4]
                }
            ]
            
        elif "alarm" in prompt_text or "clock" in prompt_text:
            subgoals = [
                {
                    "id": 1, 
                    "action": "open_clock_app", 
                    "description": "Open Clock app", 
                    "priority": 1, 
                    "dependencies": []
                },
                {
                    "id": 2, 
                    "action": "navigate_to_alarms", 
                    "description": "Navigate to alarm section", 
                    "priority": 1, 
                    "dependencies": [1]
                },
                {
                    "id": 3, 
                    "action": "create_new_alarm", 
                    "description": "Create new alarm", 
                    "priority": 1, 
                    "dependencies": [2]
                },
                {
                    "id": 4, 
                    "action": "set_alarm_time", 
                    "description": "Set alarm time to 7:30 AM", 
                    "priority": 1, 
                    "dependencies": [3]
                },
                {
                    "id": 5, 
                    "action": "save_alarm", 
                    "description": "Save the new alarm", 
                    "priority": 1, 
                    "dependencies": [4]
                }
            ]
            
        else:
            # Generic fallback with proper structure
            subgoals = [
                {
                    "id": 1, 
                    "action": "execute_task", 
                    "description": f"Execute the requested task: {prompt_text[:30]}...", 
                    "priority": 1, 
                    "dependencies": []
                }
            ]
        
        # Create proper plan structure
        plan_content = {
            "subgoals": subgoals,
            "expected_screens": ["home", "settings"] if "settings" in str(subgoals) else ["home", "app"],
            "contingencies": ["handle_popup", "handle_permission_dialog", "retry_on_failure"],
            "estimated_steps": len(subgoals)
        }
        
        print(f"[MOCK LLM] Generated executable plan with {len(subgoals)} subgoals")
        
        return LLMResponse(
            content=json.dumps(plan_content, indent=2),
            usage_tokens=100,
            cached=False,
            timestamp=time.time()
        )
    
    def _generate_executable_action(self, prompt_text: str, random_gen, full_prompt: str) -> LLMResponse:
        """Generate executable actions with high success rate"""
        
        print(f"[MOCK LLM] Generating action for: {prompt_text[:50]}...")
        
        # High success rate for realistic execution (85% instead of previous variable rates)
        success_probability = 0.85
        
        if random_gen.random() < success_probability:
            # SUCCESS CASE - Generate realistic action based on subgoal
            confidence = random_gen.uniform(0.85, 0.95)
            
            if "open_settings" in full_prompt or "settings" in prompt_text:
                action_data = {
                    "action_type": "touch",
                    "element_id": "com.android.settings:id/settings_icon",
                    "coordinates": [200, 125],
                    "confidence": confidence,
                    "reasoning": "Located Settings app icon in home screen or app drawer"
                }
                
            elif "wifi" in prompt_text or "wi-fi" in prompt_text:
                action_data = {
                    "action_type": "touch", 
                    "element_id": "com.android.settings:id/wifi_toggle_switch",
                    "coordinates": [420, 180],
                    "confidence": confidence,
                    "reasoning": "Found Wi-Fi toggle switch in Settings > Wi-Fi"
                }
                
            else:
                # Generic successful action
                action_data = {
                    "action_type": "touch",
                    "element_id": "generic_ui_element",
                    "coordinates": [random_gen.randint(200, 400), random_gen.randint(300, 500)],
                    "confidence": confidence,
                    "reasoning": "Generic UI interaction for requested action"
                }
                
        else:
            # FAILURE CASE (15% of time)
            confidence = random_gen.uniform(0.15, 0.40)
            action_data = {
                "action_type": "failed",
                "element_id": "element_not_found",
                "coordinates": [0, 0],
                "confidence": confidence,
                "reasoning": "Target UI element not found or not accessible"
            }
        
        print(f"[MOCK LLM] Generated action: {action_data['action_type']} (confidence: {action_data['confidence']:.2f})")
        
        return LLMResponse(
            content=json.dumps(action_data, indent=2),
            usage_tokens=80,
            cached=False,
            timestamp=time.time()
        )
    
    def _generate_verification_result(self, prompt_text: str, random_gen) -> LLMResponse:
        """Generate verification results with realistic pass rates"""
        
        print(f"[MOCK LLM] Generating verification for: {prompt_text[:50]}...")
        
        # High verification pass rate for successful actions (88% instead of variable)
        pass_probability = 0.88
        
        if random_gen.random() < pass_probability:
            # VERIFICATION PASSED
            confidence = random_gen.uniform(0.85, 0.95)
            verification_data = {
                "passed": True,
                "expected_state": "UI action completed successfully",
                "actual_state": "Expected UI changes observed and verified",
                "issues": [],
                "confidence": confidence,
                "suggestions": []
            }
        else:
            # VERIFICATION FAILED
            confidence = random_gen.uniform(0.30, 0.65)
            verification_data = {
                "passed": False,
                "expected_state": "Expected UI state change",
                "actual_state": "UI state verification failed or ambiguous",
                "issues": ["Expected UI change not clearly visible", "State verification inconclusive"],
                "confidence": confidence,
                "suggestions": ["Retry action with longer wait", "Check UI element state more carefully"]
            }
        
        print(f"[MOCK LLM] Verification result: {'PASSED' if verification_data['passed'] else 'FAILED'} (confidence: {confidence:.2f})")
        
        return LLMResponse(
            content=json.dumps(verification_data, indent=2),
            usage_tokens=90,
            cached=False, 
            timestamp=time.time()
        )
    
    def _generate_supervisor_analysis(self, prompt_text: str, random_gen) -> LLMResponse:
        """Generate supervisor analysis with realistic quality assessments"""
        
        print(f"[MOCK LLM] Generating supervisor analysis...")
        
        execution_quality = random_gen.uniform(0.75, 0.92)
        
        supervisor_data = {
            "execution_quality": execution_quality,
            "improvement_areas": [
                "UI element detection accuracy could be enhanced",
                "Action timing optimization needed",
                "Error recovery mechanisms should be strengthened"
            ],
            "agent_feedback": {
                "planner": [
                    "Task decomposition was appropriate for complexity level",
                    "Consider adding more contingency steps for robustness"
                ],
                "executor": [
                    "Action execution showed good precision",
                    "UI element targeting can be improved for edge cases"
                ],
                "verifier": [
                    "Verification criteria were well-applied",
                    "Confidence scoring reflects actual success rates"
                ]
            },
            "key_insights": [
                f"Overall execution quality: {execution_quality:.1%}",
                "Agent coordination functioned effectively",
                "System demonstrates good task completion capability",
                "Some optimization opportunities identified for future improvements"
            ]
        }
        
        print(f"[MOCK LLM] Supervisor analysis generated (quality: {execution_quality:.2f})")
        
        return LLMResponse(
            content=json.dumps(supervisor_data, indent=2),
            usage_tokens=120,
            cached=False,
            timestamp=time.time()
        )

# ✅ MISSING: Factory function that other modules expect
def create_llm_interface() -> LLMInterface:
    """Factory function to create appropriate LLM interface"""
    from config.default_config import config
    
    if config.USE_MOCK_LLM:
        print("[LLM Factory] Creating Mock LLM Interface")
        return MockLLMInterface()
    else:
        try:
            if not GEMINI_AVAILABLE:
                print("[LLM Factory] Gemini not available, falling back to Mock LLM")
                return MockLLMInterface()
            
            if not config.GOOGLE_API_KEY:
                print("[LLM Factory] No API key configured, using Mock LLM")
                return MockLLMInterface()
            
            print("[LLM Factory] Creating Cost-Efficient Gemini Interface")
            return CostEfficientLLMInterface(config.GOOGLE_API_KEY)
            
        except Exception as e:
            print(f"[LLM Factory] Failed to create Gemini interface: {e}. Using Mock LLM")
            return MockLLMInterface()
