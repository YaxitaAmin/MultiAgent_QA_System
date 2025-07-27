# core/llm_interface.py - ULTIMATE ENHANCED VERSION WITH ALL PROVIDERS (CORRECTED)
import os
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum

# Provider availability checks
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    anthropic = None

from loguru import logger

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    MOCK = "mock"

@dataclass
class LLMRequest:
    prompt: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    provider: LLMProvider = LLMProvider.GEMINI

@dataclass
class LLMResponse:
    content: str
    usage_tokens: int = 0
    cached: bool = False
    timestamp: float = 0.0
    confidence: float = 0.8
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    provider: Optional[LLMProvider] = None
    cost: float = 0.0

# ✅ Abstract base class
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

class OpenAIInterface(LLMInterface):
    """OpenAI GPT interface with multiple model support"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", cache_dir: str = "cache"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.request_count = 0
        
        # Cost per 1K tokens (approximate)
        self.cost_per_1k_tokens = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API call cost"""
        costs = self.cost_per_1k_tokens.get(self.model, {"input": 0.001, "output": 0.002})
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate response using OpenAI API"""
        request = LLMRequest(prompt=prompt, model=self.model, provider=LLMProvider.OPENAI)
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate(request))
    
    def analyze_ui_state(self, ui_hierarchy: str, screenshot_path: str, task_context: str) -> LLMResponse:
        """Analyze UI state using OpenAI"""
        prompt = f"""
        Analyze the Android UI state for QA testing.
        
        Task Context: {task_context}
        UI Hierarchy: {ui_hierarchy[:2000]}
        
        Provide analysis in JSON format with:
        - description: Current screen description
        - elements: Available interactive elements
        - actions: Suggested next actions
        - issues: Any detected issues
        """
        return self.generate_response(prompt)
    
    def plan_decomposition(self, high_level_goal: str, current_state: str) -> List[Dict[str, Any]]:
        """Decompose goal into actionable steps using OpenAI"""
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
        """Generate response using OpenAI API"""
        cache_key = self._get_cache_key(request)
        
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            logger.info(f"Using cached OpenAI response for {request.model}")
            return cached_response
        
        try:
            messages = [{"role": "user", "content": request.prompt}]
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            usage = response.usage
            cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
            
            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                usage_tokens=usage.total_tokens,
                cached=False,
                timestamp=time.time(),
                provider=LLMProvider.OPENAI,
                cost=cost,
                metadata={"model": self.model, "finish_reason": response.choices[0].finish_reason}
            )
            
            self._save_to_cache(cache_key, llm_response)
            logger.info(f"OpenAI {self.model} response generated (tokens: {usage.total_tokens}, cost: ${cost:.4f})")
            
            return llm_response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content=f"ERROR_OPENAI: {str(e)[:100]}",
                usage_tokens=0,
                cached=False,
                timestamp=time.time(),
                provider=LLMProvider.OPENAI
            )
    
    def _get_cache_key(self, request: LLMRequest) -> str:
        content = f"openai_{request.prompt}_{request.model}_{request.temperature}"
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
                    timestamp=data['timestamp'],
                    provider=LLMProvider.OPENAI,
                    cost=data.get('cost', 0.0)
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
                    'timestamp': response.timestamp,
                    'cost': response.cost
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

class ClaudeInterface(LLMInterface):
    """Anthropic Claude interface"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", cache_dir: str = "cache"):
        if not CLAUDE_AVAILABLE:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.request_count = 0
        
        # Cost per 1K tokens (approximate)
        self.cost_per_1k_tokens = {
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075}
        }
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API call cost"""
        costs = self.cost_per_1k_tokens.get(self.model, {"input": 0.003, "output": 0.015})
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate response using Claude API"""
        request = LLMRequest(prompt=prompt, model=self.model, provider=LLMProvider.CLAUDE)
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate(request))
    
    def analyze_ui_state(self, ui_hierarchy: str, screenshot_path: str, task_context: str) -> LLMResponse:
        """Analyze UI state using Claude"""
        prompt = f"""
        Analyze the Android UI state for QA testing.
        
        Task Context: {task_context}
        UI Hierarchy: {ui_hierarchy[:2000]}
        
        Provide analysis in JSON format with:
        - description: Current screen description
        - elements: Available interactive elements
        - actions: Suggested next actions
        - issues: Any detected issues
        """
        return self.generate_response(prompt)
    
    def plan_decomposition(self, high_level_goal: str, current_state: str) -> List[Dict[str, Any]]:
        """Decompose goal into actionable steps using Claude"""
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
        """Generate response using Claude API"""
        cache_key = self._get_cache_key(request)
        
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            logger.info(f"Using cached Claude response for {request.model}")
            return cached_response
        
        try:
            system_prompt = request.system_prompt or "You are a helpful AI assistant specialized in Android UI automation and testing."
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": request.prompt}]
            )
            
            usage = message.usage
            cost = self._calculate_cost(usage.input_tokens, usage.output_tokens)
            
            llm_response = LLMResponse(
                content=message.content[0].text,
                usage_tokens=usage.input_tokens + usage.output_tokens,
                cached=False,
                timestamp=time.time(),
                provider=LLMProvider.CLAUDE,
                cost=cost,
                metadata={"model": self.model, "stop_reason": message.stop_reason}
            )
            
            self._save_to_cache(cache_key, llm_response)
            logger.info(f"Claude {self.model} response generated (tokens: {llm_response.usage_tokens}, cost: ${cost:.4f})")
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return LLMResponse(
                content=f"ERROR_CLAUDE: {str(e)[:100]}",
                usage_tokens=0,
                cached=False,
                timestamp=time.time(),
                provider=LLMProvider.CLAUDE
            )
    
    def _get_cache_key(self, request: LLMRequest) -> str:
        content = f"claude_{request.prompt}_{request.model}_{request.temperature}"
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
                    timestamp=data['timestamp'],
                    provider=LLMProvider.CLAUDE,
                    cost=data.get('cost', 0.0)
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
                    'timestamp': response.timestamp,
                    'cost': response.cost
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

class GeminiInterface(LLMInterface):
    """Google Gemini interface (enhanced from previous version)"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", cache_dir: str = "cache"):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI package not installed. Run: pip install google-generativeai")
        
        try:
            genai.configure(api_key=api_key)
            self.api_configured = True
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            self.api_configured = False
            
        self.model_name = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.request_count = 0
        self.daily_limit = 1500
        self.rate_limit_delay = 1.0
        self.last_request_time = 0.0
        
        # Gemini is free/very low cost for many models
        self.cost_per_1k_tokens = {
            "gemini-1.5-flash": {"input": 0.0, "output": 0.0},  # Free tier
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
            "gemini-pro": {"input": 0.0005, "output": 0.0015}
        }
    
    def _calculate_cost(self, total_tokens: int) -> float:
        """Calculate API call cost (Gemini often free)"""
        costs = self.cost_per_1k_tokens.get(self.model_name, {"input": 0.0, "output": 0.0})
        # Rough estimate since Gemini doesn't separate input/output clearly
        return (total_tokens / 1000) * max(costs["input"], costs["output"])
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate response using Gemini API"""
        request = LLMRequest(prompt=prompt, model=self.model_name, provider=LLMProvider.GEMINI)
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate(request))
    
    def analyze_ui_state(self, ui_hierarchy: str, screenshot_path: str, task_context: str) -> LLMResponse:  # ✅ FIXED: LLMResponse not LLLResponse
        """Analyze UI state using Gemini"""
        prompt = f"""
        Analyze the Android UI state for QA testing.
        
        Task Context: {task_context}
        UI Hierarchy: {ui_hierarchy[:2000]}
        
        Provide analysis in JSON format with:
        - description: Current screen description
        - elements: Available interactive elements
        - actions: Suggested next actions
        - issues: Any detected issues
        """
        return self.generate_response(prompt)
    
    def plan_decomposition(self, high_level_goal: str, current_state: str) -> List[Dict[str, Any]]:
        """Decompose goal into actionable steps using Gemini"""
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
        """Generate response using Gemini API"""
        cache_key = self._get_cache_key(request)
        
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            logger.info(f"Using cached Gemini response for {request.model}")
            return cached_response
        
        if not self.api_configured:
            logger.warning("Gemini API not configured, returning error response")
            return LLMResponse(
                content="ERROR_GEMINI_NOT_CONFIGURED",
                usage_tokens=0,
                cached=False,
                timestamp=time.time(),
                provider=LLMProvider.GEMINI
            )
        
        if self.request_count >= self.daily_limit:
            logger.error("Gemini daily API limit reached")
            return LLMResponse(
                content="ERROR_GEMINI_RATE_LIMIT",
                usage_tokens=0,
                cached=False,
                timestamp=time.time(),
                provider=LLMProvider.GEMINI
            )
        
        self._enforce_rate_limit()
        
        try:
            model_name = request.model
            if model_name == "gemini-pro":
                model_name = "gemini-1.5-flash"
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
            
            usage_tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 100
            cost = self._calculate_cost(usage_tokens)
            
            llm_response = LLMResponse(
                content=response.text,
                usage_tokens=usage_tokens,
                cached=False,
                timestamp=time.time(),
                provider=LLMProvider.GEMINI,
                cost=cost,
                metadata={"model": model_name}
            )
            
            self._save_to_cache(cache_key, llm_response)
            logger.info(f"Gemini {model_name} response generated (tokens: {usage_tokens}, cost: ${cost:.4f})")
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return LLMResponse(
                content=f"ERROR_GEMINI: {str(e)[:100]}",
                usage_tokens=0,
                cached=False,
                timestamp=time.time(),
                provider=LLMProvider.GEMINI
            )
    
    def _enforce_rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, request: LLMRequest) -> str:
        content = f"gemini_{request.prompt}_{request.model}_{request.temperature}"
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
                    timestamp=data['timestamp'],
                    provider=LLMProvider.GEMINI,
                    cost=data.get('cost', 0.0)
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
                    'timestamp': response.timestamp,
                    'cost': response.cost
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

class MockLLMInterface(LLMInterface):
    """Enhanced Mock LLM interface (from previous version)"""
    
    def __init__(self):
        self.call_count = 0
        
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate response - implements LLMInterface"""
        request = LLMRequest(prompt=prompt, model="mock", provider=LLMProvider.MOCK)
        
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
            timestamp=time.time(),
            provider=LLMProvider.MOCK,
            cost=0.0
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
        """Generate realistic mock responses (enhanced from previous version)"""
        self.call_count += 1
        
        prompt_text = request.prompt.lower()
        system_text = (request.system_prompt or "").lower()
        
        # Create deterministic but varied responses
        content_hash = hashlib.md5(request.prompt.encode()).hexdigest()
        task_seed = int(content_hash[:8], 16) % 1000
        
        import random
        random.seed(task_seed)
        
        print(f"[MOCK LLM] Call #{self.call_count} - Model: {request.model} - Seed: {task_seed}")
        
        # Enhanced agent detection patterns
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
        """Generate executable plans (from previous enhanced version)"""
        print(f"[MOCK LLM] Generating plan for: {prompt_text[:50]}...")
        
        if "wifi" in prompt_text or "wi-fi" in prompt_text:
            subgoals = [
                {"id": 1, "action": "open_settings", "description": "Open Settings app from home screen", "priority": 1, "dependencies": []},
                {"id": 2, "action": "navigate_wifi", "description": "Find and tap Wi-Fi option in Settings", "priority": 1, "dependencies": [1]},
                {"id": 3, "action": "toggle_wifi", "description": "Toggle Wi-Fi switch to change state", "priority": 1, "dependencies": [2]},
                {"id": 4, "action": "verify_wifi_state", "description": "Verify Wi-Fi state has changed successfully", "priority": 1, "dependencies": [3]}
            ]
        else:
            subgoals = [
                {"id": 1, "action": "execute_task", "description": f"Execute the requested task: {prompt_text[:30]}...", "priority": 1, "dependencies": []}
            ]
        
        plan_content = {
            "subgoals": subgoals,
            "expected_screens": ["home", "settings"] if "settings" in str(subgoals) else ["home", "app"],
            "contingencies": ["handle_popup", "handle_permission_dialog", "retry_on_failure"],
            "estimated_steps": len(subgoals)
        }
        
        return LLMResponse(
            content=json.dumps(plan_content, indent=2),
            usage_tokens=100,
            cached=False,
            timestamp=time.time(),
            provider=LLMProvider.MOCK,
            cost=0.0
        )
    
    def _generate_executable_action(self, prompt_text: str, random_gen, full_prompt: str) -> LLMResponse:
        """Generate executable actions (from previous enhanced version)"""
        success_probability = 0.85
        
        if random_gen.random() < success_probability:
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
                action_data = {
                    "action_type": "touch",
                    "element_id": "generic_ui_element",
                    "coordinates": [random_gen.randint(200, 400), random_gen.randint(300, 500)],
                    "confidence": confidence,
                    "reasoning": "Generic UI interaction for requested action"
                }
        else:
            confidence = random_gen.uniform(0.15, 0.40)
            action_data = {
                "action_type": "failed",
                "element_id": "element_not_found",
                "coordinates": [0, 0],
                "confidence": confidence,
                "reasoning": "Target UI element not found or not accessible"
            }
        
        return LLMResponse(
            content=json.dumps(action_data, indent=2),
            usage_tokens=80,
            cached=False,
            timestamp=time.time(),
            provider=LLMProvider.MOCK,
            cost=0.0
        )
    
    def _generate_verification_result(self, prompt_text: str, random_gen) -> LLMResponse:
        """Generate verification results (from previous enhanced version)"""
        pass_probability = 0.88
        
        if random_gen.random() < pass_probability:
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
            confidence = random_gen.uniform(0.30, 0.65)
            verification_data = {
                "passed": False,
                "expected_state": "Expected UI state change",
                "actual_state": "UI state verification failed or ambiguous",
                "issues": ["Expected UI change not clearly visible", "State verification inconclusive"],
                "confidence": confidence,
                "suggestions": ["Retry action with longer wait", "Check UI element state more carefully"]
            }
        
        return LLMResponse(
            content=json.dumps(verification_data, indent=2),
            usage_tokens=90,
            cached=False, 
            timestamp=time.time(),
            provider=LLMProvider.MOCK,
            cost=0.0
        )
    
    def _generate_supervisor_analysis(self, prompt_text: str, random_gen) -> LLMResponse:
        """Generate supervisor analysis (from previous enhanced version)"""
        execution_quality = random_gen.uniform(0.75, 0.92)
        
        supervisor_data = {
            "execution_quality": execution_quality,
            "improvement_areas": [
                "UI element detection accuracy could be enhanced",
                "Action timing optimization needed",
                "Error recovery mechanisms should be strengthened"
            ],
            "agent_feedback": {
                "planner": ["Task decomposition was appropriate for complexity level"],
                "executor": ["Action execution showed good precision"],
                "verifier": ["Verification criteria were well-applied"]
            },
            "key_insights": [
                f"Overall execution quality: {execution_quality:.1%}",
                "Agent coordination functioned effectively"
            ]
        }
        
        return LLMResponse(
            content=json.dumps(supervisor_data, indent=2),
            usage_tokens=120,
            cached=False,
            timestamp=time.time(),
            provider=LLMProvider.MOCK,
            cost=0.0
        )

class MultiProviderLLMInterface(LLMInterface):
    """Multi-provider LLM interface with fallback and load balancing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.providers = {}
        self.primary_provider = None
        self.fallback_order = []
        self.total_cost = 0.0
        
        # Initialize providers based on config
        if config.get("openai_api_key") and OPENAI_AVAILABLE:
            self.providers[LLMProvider.OPENAI] = OpenAIInterface(
                config["openai_api_key"], 
                config.get("openai_model", "gpt-4o-mini")
            )
            self.fallback_order.append(LLMProvider.OPENAI)
        
        if config.get("claude_api_key") and CLAUDE_AVAILABLE:
            self.providers[LLMProvider.CLAUDE] = ClaudeInterface(
                config["claude_api_key"],
                config.get("claude_model", "claude-3-5-sonnet-20241022")
            )
            self.fallback_order.append(LLMProvider.CLAUDE)
        
        if config.get("gemini_api_key") and GEMINI_AVAILABLE:
            self.providers[LLMProvider.GEMINI] = GeminiInterface(
                config["gemini_api_key"],
                config.get("gemini_model", "gemini-1.5-flash")
            )
            self.fallback_order.append(LLMProvider.GEMINI)
        
        # Always have mock as final fallback
        self.providers[LLMProvider.MOCK] = MockLLMInterface()
        self.fallback_order.append(LLMProvider.MOCK)
        
        # Set primary provider
        self.primary_provider = self.fallback_order[0] if self.fallback_order else LLMProvider.MOCK
        
        logger.info(f"MultiProvider initialized with: {list(self.providers.keys())}")
        logger.info(f"Primary provider: {self.primary_provider}")
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate response with fallback logic"""
        for provider in self.fallback_order:
            try:
                interface = self.providers[provider]
                response = interface.generate_response(prompt, context)
                
                if response.content and not response.content.startswith("ERROR"):
                    self.total_cost += response.cost
                    logger.info(f"Successfully used {provider.value} (cost: ${response.cost:.4f})")
                    return response
                else:
                    logger.warning(f"{provider.value} returned error response, trying next provider")
                    
            except Exception as e:
                logger.error(f"{provider.value} failed: {e}, trying next provider")
                continue
        
        # If all providers fail, return mock response
        return LLMResponse(
            content="All LLM providers failed, using mock response",
            usage_tokens=0,
            timestamp=time.time(),
            provider=LLMProvider.MOCK,
            cost=0.0
        )
    
    def analyze_ui_state(self, ui_hierarchy: str, screenshot_path: str, task_context: str) -> LLMResponse:
        """Analyze UI state with provider fallback"""
        primary_interface = self.providers[self.primary_provider]
        try:
            response = primary_interface.analyze_ui_state(ui_hierarchy, screenshot_path, task_context)
            self.total_cost += response.cost
            return response
        except Exception as e:
            logger.error(f"Primary provider {self.primary_provider} failed for UI analysis: {e}")
            # Fallback to mock
            return self.providers[LLMProvider.MOCK].analyze_ui_state(ui_hierarchy, screenshot_path, task_context)
    
    def plan_decomposition(self, high_level_goal: str, current_state: str) -> List[Dict[str, Any]]:
        """Plan decomposition with provider fallback"""
        primary_interface = self.providers[self.primary_provider]
        try:
            response = primary_interface.plan_decomposition(high_level_goal, current_state)
            return response
        except Exception as e:
            logger.error(f"Primary provider {self.primary_provider} failed for plan decomposition: {e}")
            # Fallback to mock
            return self.providers[LLMProvider.MOCK].plan_decomposition(high_level_goal, current_state)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary across all providers"""
        return {
            "total_cost": self.total_cost,
            "providers_used": list(self.providers.keys()),
            "primary_provider": self.primary_provider.value
        }

# ✅ Enhanced factory function
def create_llm_interface(provider: Optional[Union[str, LLMProvider]] = None) -> LLMInterface:
    """Enhanced factory function to create appropriate LLM interface"""
    from config.default_config import config
    
    # If specific provider requested
    if provider:
        if isinstance(provider, str):
            provider = LLMProvider(provider)
        
        if provider == LLMProvider.OPENAI:
            if config.OPENAI_API_KEY and OPENAI_AVAILABLE:
                print("[LLM Factory] Creating OpenAI Interface")
                return OpenAIInterface(config.OPENAI_API_KEY, config.OPENAI_MODEL)
            else:
                print("[LLM Factory] OpenAI not available, falling back to Mock")
                return MockLLMInterface()
        
        elif provider == LLMProvider.CLAUDE:
            if config.CLAUDE_API_KEY and CLAUDE_AVAILABLE:
                print("[LLM Factory] Creating Claude Interface")
                return ClaudeInterface(config.CLAUDE_API_KEY, config.CLAUDE_MODEL)
            else:
                print("[LLM Factory] Claude not available, falling back to Mock")
                return MockLLMInterface()
        
        elif provider == LLMProvider.GEMINI:
            if config.GOOGLE_API_KEY and GEMINI_AVAILABLE:
                print("[LLM Factory] Creating Gemini Interface")
                return GeminiInterface(config.GOOGLE_API_KEY, config.GEMINI_MODEL)
            else:
                print("[LLM Factory] Gemini not available, falling back to Mock")
                return MockLLMInterface()
        
        elif provider == LLMProvider.MOCK:
            print("[LLM Factory] Creating Mock LLM Interface")
            return MockLLMInterface()
    
    # Auto-selection based on config
    if config.USE_MOCK_LLM:
        print("[LLM Factory] Using Mock LLM (config setting)")
        return MockLLMInterface()
    
    # Try to create multi-provider interface
    provider_config = {
        "openai_api_key": getattr(config, 'OPENAI_API_KEY', ''),
        "openai_model": getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini'),
        "claude_api_key": getattr(config, 'CLAUDE_API_KEY', ''),
        "claude_model": getattr(config, 'CLAUDE_MODEL', 'claude-3-5-sonnet-20241022'),
        "gemini_api_key": getattr(config, 'GOOGLE_API_KEY', ''),
        "gemini_model": getattr(config, 'GEMINI_MODEL', 'gemini-1.5-flash')
    }
    
    # Check if any real providers are available
    real_providers_available = any([
        provider_config["openai_api_key"] and OPENAI_AVAILABLE,
        provider_config["claude_api_key"] and CLAUDE_AVAILABLE,
        provider_config["gemini_api_key"] and GEMINI_AVAILABLE
    ])
    
    if real_providers_available:
        print("[LLM Factory] Creating Multi-Provider LLM Interface")
        return MultiProviderLLMInterface(provider_config)
    else:
        print("[LLM Factory] No real providers available, using Mock LLM")
        return MockLLMInterface()
