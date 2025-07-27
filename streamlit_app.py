"""
Enhanced Streamlit Web Interface for Multi-Agent QA System
With Advanced LLM Provider Selection, API Key Management, and Sophisticated Prompt Refinement
"""

import streamlit as st
import asyncio
import json
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from typing import Dict, List, Optional, Any

# Import your corrected modules
from env_manager import EnvironmentManager
from config.default_config import config
from core.logger import QATestResult
from core.llm_interface import create_llm_interface, LLMProvider

# Page configuration
st.set_page_config(
    page_title="Multi-Agent QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedPromptRefiner:
    """Advanced prompt refinement engine with multiple strategies"""
    
    def __init__(self, llm_interface=None):
        self.llm_interface = llm_interface
        self.refinement_templates = {
            "android_ui": {
                "system_prompt": """You are an expert Android UI automation engineer. Transform user requests into precise, actionable test instructions.""",
                "template": """
                Transform this user request into a detailed Android UI automation script:
                
                Original: "{original_prompt}"
                
                Requirements:
                1. Break into specific UI interaction steps
                2. Use precise Android terminology (buttons, toggles, menus, text fields)
                3. Include verification steps
                4. Add error handling considerations
                5. Specify exact navigation paths
                6. Include success criteria
                
                Format as: Step-by-step instructions with clear actions and expected outcomes.
                """
            },
            "settings_navigation": {
                "system_prompt": """You are a specialist in Android Settings app navigation and system configurations.""",
                "template": """
                Create detailed Settings navigation instructions for: "{original_prompt}"
                
                Focus on:
                - Exact menu paths (Settings → Network & Internet → WiFi)
                - Toggle switch locations and states
                - Verification of state changes
                - Fallback navigation options
                - Expected UI elements at each step
                
                Return precise step-by-step navigation with verification.
                """
            },
            "app_testing": {
                "system_prompt": """You are an expert in Android app testing and interaction patterns.""",
                "template": """
                Generate comprehensive app testing instructions for: "{original_prompt}"
                
                Include:
                - App launch methods (home screen, app drawer, recent apps)
                - Core functionality testing steps
                - Input validation scenarios
                - UI element interactions
                - Expected behaviors and outputs
                - Error case handling
                
                Provide detailed testing workflow.
                """
            },
            "accessibility": {
                "system_prompt": """You are an accessibility testing expert for Android applications.""",
                "template": """
                Create accessibility-focused test instructions for: "{original_prompt}"
                
                Emphasize:
                - Screen reader compatibility
                - Touch target sizes
                - Contrast and visibility
                - Navigation without visual cues
                - Voice command alternatives
                - Assistive technology integration
                
                Return accessibility-comprehensive test steps.
                """
            }
        }
    
    def analyze_prompt_intent(self, prompt: str) -> str:
        """Analyze prompt to determine the best refinement strategy"""
        prompt_lower = prompt.lower()
        
        # Settings-related keywords
        if any(keyword in prompt_lower for keyword in ['settings', 'wifi', 'bluetooth', 'airplane', 'network', 'display', 'sound', 'battery']):
            return "settings_navigation"
        
        # App-specific keywords
        elif any(keyword in prompt_lower for keyword in ['calculator', 'calendar', 'camera', 'contacts', 'email', 'browser', 'clock', 'alarm']):
            return "app_testing"
        
        # Accessibility keywords
        elif any(keyword in prompt_lower for keyword in ['accessibility', 'voice', 'screen reader', 'talkback', 'contrast', 'magnify']):
            return "accessibility"
        
        # Default to general UI
        else:
            return "android_ui"
    
    def extract_ui_elements(self, prompt: str) -> List[str]:
        """Extract UI elements mentioned in the prompt"""
        ui_patterns = {
            'buttons': r'\b(button|btn|tap|click|press)\b',
            'toggles': r'\b(toggle|switch|on|off|enable|disable)\b',
            'menus': r'\b(menu|dropdown|select|option|choose)\b',
            'text_fields': r'\b(type|enter|input|text|field)\b',
            'navigation': r'\b(navigate|go to|open|access|find)\b'
        }
        
        elements = []
        for element_type, pattern in ui_patterns.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                elements.append(element_type)
        
        return elements
    
    def generate_context_aware_refinement(self, original_prompt: str, style: str = "detailed", focus: str = "general") -> str:
        """Generate context-aware prompt refinement"""
        if not self.llm_interface:
            return self._fallback_refinement(original_prompt)
        
        try:
            # Analyze intent
            intent = self.analyze_prompt_intent(original_prompt)
            ui_elements = self.extract_ui_elements(original_prompt)
            
            # Select appropriate template
            template_config = self.refinement_templates.get(intent, self.refinement_templates["android_ui"])
            
            # Build refinement prompt
            refinement_prompt = template_config["template"].format(original_prompt=original_prompt)
            
            # Add style and focus modifiers
            if style == "step_by_step":
                refinement_prompt += "\n\nFormat: Number each step clearly (1., 2., 3., etc.)"
            elif style == "concise":
                refinement_prompt += "\n\nFormat: Keep instructions concise but complete"
            elif style == "technical":
                refinement_prompt += "\n\nFormat: Use technical Android terminology and specific element IDs when possible"
            
            if focus != "general":
                refinement_prompt += f"\n\nSpecial focus: Emphasize {focus} aspects"
            
            # Add UI elements context
            if ui_elements:
                refinement_prompt += f"\n\nDetected UI elements: {', '.join(ui_elements)}. Ensure these are addressed."
            
            # Generate refined prompt
            response = self.llm_interface.generate_response(
                refinement_prompt,
                context={"system_prompt": template_config["system_prompt"]}
            )
            
            refined = response.content.strip()
            
            # Post-process the refinement
            refined = self._post_process_refinement(refined, original_prompt)
            
            return refined if refined and len(refined) > len(original_prompt) * 0.5 else original_prompt
            
        except Exception as e:
            st.warning(f"Advanced refinement failed: {e}")
            return self._fallback_refinement(original_prompt)
    
    def _fallback_refinement(self, original_prompt: str) -> str:
        """Fallback refinement when LLM is not available"""
        # Basic rule-based enhancement
        enhanced = original_prompt
        
        # Add structure if missing
        if not re.search(r'\d+\.|\-|\*', enhanced):
            steps = enhanced.split(',') if ',' in enhanced else [enhanced]
            enhanced = "\n".join([f"{i+1}. {step.strip()}" for i, step in enumerate(steps)])
        
        # Add verification if missing
        if 'verify' not in enhanced.lower():
            enhanced += "\n\nVerification: Confirm the action completed successfully"
        
        # Add Android-specific terms
        android_replacements = {
            'click': 'tap',
            'turn on': 'enable',
            'turn off': 'disable',
            'settings': 'Settings app'
        }
        
        for old, new in android_replacements.items():
            enhanced = enhanced.replace(old, new)
        
        return enhanced
    
    def _post_process_refinement(self, refined: str, original: str) -> str:
        """Post-process the refined prompt for quality"""
        # Ensure minimum length improvement
        if len(refined) < len(original) * 1.2:
            refined = f"{refined}\n\nAdditional verification: Ensure all UI changes are visible and the task objective is met."
        
        # Add Android-specific formatting
        if not refined.startswith("Android UI Test:"):
            refined = f"Android UI Test: {refined}"
        
        # Ensure proper step formatting
        lines = refined.split('\n')
        formatted_lines = []
        step_count = 1
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Step', 'Android UI Test:', 'Verification:', 'Expected:')):
                if not re.match(r'^\d+\.', line):
                    line = f"{step_count}. {line}"
                    step_count += 1
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

def init_session_state():
    """Initialize session state variables"""
    if 'qa_manager' not in st.session_state:
        st.session_state.qa_manager = None
    if 'execution_history' not in st.session_state:
        st.session_state.execution_history = []
    if 'current_execution' not in st.session_state:
        st.session_state.current_execution = None
    if 'android_wild_results' not in st.session_state:
        st.session_state.android_wild_results = None
    if 'llm_interface' not in st.session_state:
        st.session_state.llm_interface = None
    if 'refined_prompts' not in st.session_state:
        st.session_state.refined_prompts = {}
    if 'prompt_refiner' not in st.session_state:
        st.session_state.prompt_refiner = AdvancedPromptRefiner()

def get_llm_interface_with_config(provider: str, api_key: str = None, model: str = None):
    """Create LLM interface with user-provided configuration"""
    try:
        if provider == "mock":
            from core.llm_interface import MockLLMInterface
            interface = MockLLMInterface()
        elif provider == "openai":
            if not api_key:
                st.error("OpenAI API key is required")
                return None
            from core.llm_interface import OpenAIInterface
            interface = OpenAIInterface(api_key, model=model or "gpt-4o-mini")
        elif provider == "claude":
            if not api_key:
                st.error("Claude API key is required")
                return None
            from core.llm_interface import ClaudeInterface
            interface = ClaudeInterface(api_key, model=model or "claude-3-5-sonnet-20241022")
        elif provider == "gemini":
            if not api_key:
                st.error("Gemini API key is required")
                return None
            from core.llm_interface import GeminiInterface
            interface = GeminiInterface(api_key, model=model or "gemini-1.5-flash")
        else:
            st.error(f"Unknown provider: {provider}")
            return None
        
        # Test the connection
        try:
            test_response = interface.generate_response("Test connection: respond with 'OK'")
            if test_response.content:
                return interface
        except Exception as e:
            st.error(f"Connection test failed: {e}")
            return None
            
    except ImportError as e:
        st.error(f"Failed to import {provider} interface: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to create {provider} interface: {e}")
        return None

@st.cache_resource
def get_environment_manager():
    """Get cached environment manager"""
    try:
        return EnvironmentManager()
    except Exception as e:
        st.error(f"Failed to initialize EnvironmentManager: {e}")
        return None

def check_agent_s_status():
    """Check if Agent-S architecture is active"""
    try:
        if st.session_state.qa_manager is None:
            return {"active": False, "reason": "QA Manager not initialized"}
        
        from agents.base_agents import QAAgentS2
        
        agents = {
            "planner": st.session_state.qa_manager.planner_agent,
            "executor": st.session_state.qa_manager.executor_agent,
            "verifier": st.session_state.qa_manager.verifier_agent,
            "supervisor": st.session_state.qa_manager.supervisor_agent
        }
        
        agent_status = {}
        for name, agent in agents.items():
            agent_status[name] = isinstance(agent, QAAgentS2)
        
        all_agent_s = all(agent_status.values())
        
        return {
            "active": all_agent_s,
            "details": agent_status,
            "total_agents": len(agents),
            "agent_s_agents": sum(agent_status.values())
        }
        
    except Exception as e:
        return {"active": False, "error": str(e)}

def get_system_status():
    """Get comprehensive system status"""
    try:
        agent_s_status = check_agent_s_status()
        
        if st.session_state.qa_manager:
            try:
                metrics = st.session_state.qa_manager.get_system_metrics()
                test_summary = metrics.get('test_summary', {})
                system_integration = metrics.get('system_integration', {})
                
                return {
                    "initialized": True,
                    "agent_s": agent_s_status["active"],
                    "agent_s_details": agent_s_status,
                    "android_world": system_integration.get('android_world_connected', True),
                    "llm_interface": system_integration.get('llm_interface', 'Mock').title(),
                    "tests_completed": test_summary.get('total_tests', len(st.session_state.execution_history)),
                    "pass_rate": test_summary.get('pass_rate', 0.0),
                    "coordination_active": getattr(st.session_state.qa_manager, 'agent_coordination_active', True)
                }
            except Exception as e:
                return {
                    "initialized": True,
                    "agent_s": agent_s_status["active"],
                    "agent_s_details": agent_s_status,
                    "android_world": True,
                    "llm_interface": "Mock",
                    "tests_completed": len(st.session_state.execution_history),
                    "error": str(e)
                }
        else:
            return {
                "initialized": False,
                "agent_s": False,
                "agent_s_details": agent_s_status,
                "android_world": False,
                "llm_interface": "Not Available",
                "tests_completed": 0
            }
            
    except Exception as e:
        return {
            "initialized": False,
            "agent_s": False,
            "android_world": False,
            "llm_interface": "Error",
            "tests_completed": 0,
            "error": str(e)
        }

def main():
    """Main Streamlit application"""
    init_session_state()
    
    st.title("🤖 Enhanced Multi-Agent QA System")
    st.subheader("Agent-S + Android World + Advanced LLM Integration")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration Panel")
        
        # System Status
        status = get_system_status()
        
        if status["initialized"]:
            st.success("**System Status:** ✅ INITIALIZED")
            
            agent_s_details = status["agent_s_details"]
            if status["agent_s"]:
                st.success(f"**Agent-S:** ✅ Architecture Active ({agent_s_details.get('agent_s_agents', 0)}/4 agents)")
            else:
                st.error("**Agent-S:** ❌ Not Active")
                if "error" in agent_s_details:
                    st.caption(f"Error: {agent_s_details['error']}")
            
            st.success(f"**Android World:** {'✅' if status['android_world'] else '❌'}")
            st.info(f"**LLM Interface:** {status['llm_interface']}")
            st.info(f"**Tests Completed:** {status['tests_completed']}")
            
            if status["agent_s"] and "details" in agent_s_details:
                with st.expander("🤖 Agent-S Details"):
                    for agent_name, is_active in agent_s_details["details"].items():
                        status_icon = "✅" if is_active else "❌"
                        st.write(f"{status_icon} {agent_name.title()}Agent: {'QAAgentS2' if is_active else 'BaseAgent'}")
            
            if status.get("coordination_active"):
                st.success("**Coordination:** ✅ Active")
            else:
                st.warning("**Coordination:** ⚠️ Limited")
        else:
            st.warning("**System Status:** ⚠️ NOT INITIALIZED")
            if "error" in status:
                st.error(f"Error: {status['error']}")
        
        st.divider()
        
        # ✅ ENHANCED: Advanced LLM Provider Configuration
        with st.expander("🧠 Advanced LLM Configuration", expanded=True):
            st.subheader("LLM Provider Setup")
            
            # Provider selection with descriptions
            provider_options = {
                "Mock LLM (Free, Always Available)": "mock",
                "OpenAI GPT (Paid, High Quality)": "openai", 
                "Claude (Paid, Excellent Reasoning)": "claude",
                "Google Gemini (Free Tier Available)": "gemini"
            }
            
            selected_provider_name = st.selectbox(
                "Select LLM Provider:",
                list(provider_options.keys()),
                index=0,
                help="Choose your preferred LLM provider for prompt refinement and agent communication"
            )
            
            selected_provider = provider_options[selected_provider_name]
            
            # API Key input with validation
            api_key = None
            if selected_provider != "mock":
                api_key = st.text_input(
                    f"{selected_provider_name.split('(')[0].strip()} API Key:",
                    type="password",
                    help=f"Enter your API key for {selected_provider_name.split('(')[0].strip()}",
                    placeholder="sk-... or your API key"
                )
                
                if api_key:
                    # Basic validation
                    if selected_provider == "openai" and not api_key.startswith("sk-"):
                        st.warning("⚠️ OpenAI API keys typically start with 'sk-'")
                    elif selected_provider == "claude" and not api_key.startswith("sk-ant-"):
                        st.warning("⚠️ Claude API keys typically start with 'sk-ant-'")
                    
                    st.success("✅ API key entered")
                else:
                    st.warning(f"⚠️ {selected_provider_name.split('(')[0].strip()} API key required")
                    st.info("💡 Will fall back to Mock LLM if no key provided")
            
            # Enhanced model selection
            model_options = {
                "openai": {
                    "gpt-4o-mini": "Fast, cost-effective, good for most tasks",
                    "gpt-4o": "Highest quality, slower, more expensive",
                    "gpt-4-turbo": "Balanced speed and quality",
                    "gpt-3.5-turbo": "Fastest, lowest cost"
                },
                "claude": {
                    "claude-3-5-sonnet-20241022": "Best overall performance, balanced",
                    "claude-3-5-haiku-20241022": "Fastest, most cost-effective",
                    "claude-3-opus-20240229": "Highest quality, most expensive"
                },
                "gemini": {
                    "gemini-1.5-flash": "Free tier, fast responses",
                    "gemini-1.5-pro": "Higher quality, rate limited",
                    "gemini-pro": "Legacy model, stable"
                }
            }
            
            selected_model = None
            if selected_provider in model_options:
                model_info = model_options[selected_provider]
                selected_model = st.selectbox(
                    f"Select {selected_provider_name.split('(')[0].strip()} Model:",
                    list(model_info.keys()),
                    index=0,
                    format_func=lambda x: f"{x} - {model_info[x]}"
                )
                st.caption(f"ℹ️ {model_info[selected_model]}")
            
            # Connection and testing
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔗 Connect Provider", type="primary"):
                    with st.spinner("Connecting and testing..."):
                        interface = get_llm_interface_with_config(selected_provider, api_key, selected_model)
                        if interface:
                            st.session_state.llm_interface = interface
                            st.session_state.prompt_refiner = AdvancedPromptRefiner(interface)
                            st.success(f"✅ Connected to {selected_provider_name.split('(')[0].strip()}")
                            
                            # Update config
                            config.USE_MOCK_LLM = (selected_provider == "mock")
                            if selected_provider == "openai":
                                config.OPENAI_API_KEY = api_key or ""
                                config.OPENAI_MODEL = selected_model
                            elif selected_provider == "claude":
                                config.CLAUDE_API_KEY = api_key or ""
                                config.CLAUDE_MODEL = selected_model
                            elif selected_provider == "gemini":
                                config.GOOGLE_API_KEY = api_key or ""
                                config.GEMINI_MODEL = selected_model
                        else:
                            st.error(f"❌ Failed to connect to {selected_provider_name.split('(')[0].strip()}")
            
            with col2:
                if st.button("🧪 Test Connection"):
                    if st.session_state.llm_interface:
                        with st.spinner("Testing..."):
                            try:
                                test_response = st.session_state.llm_interface.generate_response(
                                    "Respond with: 'Connection test successful for Android QA system'"
                                )
                                if test_response.content:
                                    st.success("✅ Connection working!")
                                    st.info(f"Response: {test_response.content[:100]}...")
                                else:
                                    st.error("❌ No response received")
                            except Exception as e:
                                st.error(f"❌ Test failed: {e}")
                    else:
                        st.warning("⚠️ No LLM interface connected")
            
            # Show current status
            if st.session_state.llm_interface:
                current_provider = st.session_state.llm_interface.__class__.__name__
                st.success(f"🔗 Active: {current_provider}")
                
                # Show capabilities
                with st.expander("🔍 Provider Capabilities"):
                    if "Mock" in current_provider:
                        st.info("• Always available\n• Deterministic responses\n• No API costs\n• Good for testing")
                    elif "OpenAI" in current_provider:
                        st.info("• High-quality responses\n• Fast processing\n• Usage-based pricing\n• Excellent prompt refinement")
                    elif "Claude" in current_provider:
                        st.info("• Superior reasoning\n• Long context window\n• Usage-based pricing\n• Great for complex tasks")
                    elif "Gemini" in current_provider:
                        st.info("• Free tier available\n• Good performance\n• Google integration\n• Rate limited")
        
        st.divider()
        
        # Agent Configuration
        with st.expander("🤖 Agent Settings"):
            st.subheader("Agent Configuration")
            
            max_steps = st.number_input(
                "Max Plan Steps", 
                min_value=5, 
                max_value=50, 
                value=getattr(config, 'MAX_PLAN_STEPS', 20),
                help="Maximum number of steps in execution plan"
            )
            config.MAX_PLAN_STEPS = max_steps
            
            timeout = st.number_input(
                "Timeout (seconds)", 
                min_value=30, 
                max_value=600, 
                value=120,
                help="Maximum time to wait for task completion"
            )
            
            verification_threshold = st.slider(
                "Verification Threshold", 
                min_value=0.5, 
                max_value=1.0, 
                value=getattr(config, 'VERIFICATION_THRESHOLD', 0.7),
                step=0.1,
                help="Confidence threshold for verification steps"
            )
            config.VERIFICATION_THRESHOLD = verification_threshold
        
        # Environment Configuration
        with st.expander("📱 Environment Settings"):
            st.subheader("Android World Settings")
            
            android_tasks = getattr(config, 'ANDROID_WORLD_TASKS', [
                "settings_wifi", "clock_alarm", "calculator_basic", 
                "contacts_add", "email_search"
            ])
            
            android_task = st.selectbox(
                "Android World Task",
                android_tasks,
                index=0,
                help="Select the underlying Android World task type"
            )
            
            device_id = st.text_input(
                "Android Device ID",
                value=getattr(config, 'ANDROID_DEVICE_ID', "emulator-5554"),
                help="ADB device identifier (e.g., emulator-5554 or device serial)"
            )
            config.ANDROID_DEVICE_ID = device_id
        
        # Initialize system
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing QA system..."):
                try:
                    st.session_state.qa_manager = get_environment_manager()
                    if st.session_state.qa_manager:
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        success = loop.run_until_complete(st.session_state.qa_manager.initialize())
                        
                        if success:
                            st.success("✅ System initialized successfully!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ System initialization failed")
                    else:
                        st.error("❌ Failed to create EnvironmentManager")
                except Exception as e:
                    st.error(f"❌ Initialization error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Execute Task", 
        "✨ Advanced Prompt Studio", 
        "📊 Dashboard", 
        "📋 History", 
        "⚙️ Advanced"
    ])
    
    with tab1:
        execute_task_tab()
    
    with tab2:
        advanced_prompt_studio_tab()
    
    with tab3:
        dashboard_tab()
    
    with tab4:
        history_tab()
    
    with tab5:
        advanced_tab()

def advanced_prompt_studio_tab():
    """✅ ENHANCED: Advanced prompt refinement and enhancement studio"""
    st.header("✨ Advanced Prompt Studio")
    st.subheader("AI-Powered Prompt Refinement with Context-Aware Intelligence")
    
    # Check LLM availability
    if not st.session_state.llm_interface:
        st.warning("⚠️ Please connect to an LLM provider first to use Advanced Prompt Studio.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 Advanced Prompt Studio uses AI to transform simple requests into comprehensive, actionable Android UI test instructions.")
            st.markdown("""
            **Features:**
            - Context-aware refinement strategies
            - Android UI pattern recognition
            - Multi-step instruction generation
            - Verification step inclusion
            - Error handling suggestions
            """)
        with col2:
            if st.button("🔗 Setup LLM Provider", type="primary"):
                st.info("👈 Use the sidebar to configure your LLM provider")
        return
    
    # Show current capabilities
    st.success(f"🔗 Connected to: {st.session_state.llm_interface.__class__.__name__}")
    
    # Enhanced prompt input section
    st.subheader("📝 Prompt Input & Analysis")
    
    # Example prompts organized by category
    example_categories = {
        "Settings & System": [
            "test wifi",
            "turn on airplane mode", 
            "check battery settings",
            "adjust display brightness",
            "enable bluetooth"
        ],
        "App Interactions": [
            "open calculator and do 5+3",
            "set alarm for 7am",
            "create new contact",
            "take a photo with camera",
            "check calendar events"
        ],
        "Navigation Tests": [
            "open notification panel",
            "access recent apps",
            "navigate to app drawer",
            "use back button navigation",
            "test home screen widgets"
        ],
        "Accessibility": [
            "test with screen reader",
            "check high contrast mode",
            "verify voice commands",
            "test magnification gestures",
            "validate touch targets"
        ]
    }
    
    # Category selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox(
            "Choose example category or write custom:",
            ["Custom prompt..."] + list(example_categories.keys()),
            index=0
        )
    
    with col2:
        if selected_category != "Custom prompt...":
            st.metric("Examples Available", len(example_categories[selected_category]))
    
    # Prompt input
    if selected_category == "Custom prompt...":
        original_prompt = st.text_area(
            "Enter your task description:",
            placeholder="Describe what you want the Android QA system to test...\nExample: 'Test WiFi toggle functionality' or 'Verify calculator basic operations'",
            height=120,
            help="Enter any description - the AI will transform it into detailed, actionable test instructions"
        )
        st.caption("💡 Pro tip: Even simple descriptions like 'test wifi' will be transformed into comprehensive test procedures")
    else:
        # Show examples from selected category
        example_options = example_categories[selected_category]
        selected_example = st.selectbox(
            f"Select example from {selected_category}:",
            example_options + ["Custom..."]
        )
        
        if selected_example == "Custom...":
            original_prompt = st.text_area(
                f"Custom {selected_category} prompt:",
                placeholder=f"Enter your custom {selected_category.lower()} testing request...",
                height=100
            )
        else:
            original_prompt = st.text_area(
                "Selected example (editable):",
                value=selected_example,
                height=100,
                help="You can edit this example or use it as-is"
            )
    
    # Real-time prompt analysis
    if original_prompt and original_prompt.strip():
        with st.expander("🔍 Prompt Analysis", expanded=False):
            refiner = st.session_state.prompt_refiner
            intent = refiner.analyze_prompt_intent(original_prompt)
            ui_elements = refiner.extract_ui_elements(original_prompt)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Detected Intent:**")
                intent_icons = {
                    "settings_navigation": "⚙️ Settings Navigation",
                    "app_testing": "📱 App Testing", 
                    "accessibility": "♿ Accessibility",
                    "android_ui": "🤖 General UI"
                }
                st.info(intent_icons.get(intent, "🤖 General UI"))
            
            with col2:
                st.write("**UI Elements Found:**")
                if ui_elements:
                    for element in ui_elements:
                        st.write(f"• {element.replace('_', ' ').title()}")
                else:
                    st.write("• General interactions")
            
            with col3:
                st.write("**Complexity Score:**")
                complexity = min(len(original_prompt.split()) / 5, 5)
                st.progress(complexity / 5)
                st.write(f"{complexity:.1f}/5.0")
    
    # Enhanced refinement options
    st.subheader("🎨 Refinement Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        refinement_style = st.selectbox(
            "Refinement Style:",
            [
                "detailed", # Detailed & Comprehensive
                "step_by_step", # Clear Step-by-Step
                "concise", # Concise & Focused
                "technical" # Technical & Specific
            ],
            format_func=lambda x: {
                "detailed": "Detailed & Comprehensive",
                "step_by_step": "Clear Step-by-Step", 
                "concise": "Concise & Focused",
                "technical": "Technical & Specific"
            }[x],
            index=0,
            help="Choose how the AI should structure the refined instructions"
        )
    
    with col2:
        focus_area = st.selectbox(
            "Focus Area:",
            [
                "general", # General UI Testing
                "settings", # Settings Navigation
                "apps", # App Interactions
                "system", # System Features
                "accessibility" # Accessibility Testing
            ],
            format_func=lambda x: {
                "general": "General UI Testing",
                "settings": "Settings Navigation", 
                "apps": "App Interactions",
                "system": "System Features",
                "accessibility": "Accessibility Testing"
            }[x],
            index=0,
            help="Specify the primary focus for refinement"
        )
    
    with col3:
        include_advanced = st.checkbox(
            "Include Advanced Features",
            value=True,
            help="Add error handling, edge cases, and verification steps"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Refinement Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_verification = st.checkbox("Include verification steps", value=True)
            include_error_handling = st.checkbox("Add error handling", value=include_advanced)
            include_timing = st.checkbox("Add timing considerations", value=False)
        
        with col2:
            use_device_specific = st.checkbox("Use device-specific instructions", value=True)
            add_screenshots = st.checkbox("Suggest screenshot points", value=False)
            include_alternatives = st.checkbox("Include alternative approaches", value=False)
    
    # Main refinement action
    if st.button("✨ Refine Prompt with AI", type="primary", disabled=not original_prompt.strip()):
        if original_prompt.strip():
            with st.spinner("🤖 AI is analyzing and refining your prompt..."):
                try:
                    # Use advanced prompt refiner
                    refined_prompt = st.session_state.prompt_refiner.generate_context_aware_refinement(
                        original_prompt.strip(),
                        style=refinement_style,
                        focus=focus_area
                    )
                    
                    # Apply advanced options
                    if include_verification and "verification" not in refined_prompt.lower():
                        refined_prompt += "\n\nVerification Steps:\n• Confirm all UI changes are visible\n• Validate expected outcomes\n• Check for error messages or unexpected behavior"
                    
                    if include_error_handling and "error" not in refined_prompt.lower():
                        refined_prompt += "\n\nError Handling:\n• Handle potential permission dialogs\n• Account for network connectivity issues\n• Provide fallback navigation paths"
                    
                    if include_timing:
                        refined_prompt += "\n\nTiming Considerations:\n• Allow 2-3 seconds for UI transitions\n• Wait for animations to complete\n• Use explicit waits for dynamic content"
                    
                    # Store in session state
                    prompt_id = f"prompt_{len(st.session_state.refined_prompts)}"
                    refinement_metadata = {
                        "original": original_prompt,
                        "refined": refined_prompt,
                        "style": refinement_style,
                        "focus": focus_area,
                        "advanced_features": include_advanced,
                        "timestamp": datetime.now(),
                        "provider": st.session_state.llm_interface.__class__.__name__,
                        "intent": st.session_state.prompt_refiner.analyze_prompt_intent(original_prompt),
                        "ui_elements": st.session_state.prompt_refiner.extract_ui_elements(original_prompt)
                    }
                    st.session_state.refined_prompts[prompt_id] = refinement_metadata
                    
                    # Display results
                    st.success("✅ Prompt refined successfully with AI!")
                    
                    # Enhanced comparison view
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📝 Original Prompt")
                        st.text_area("", value=original_prompt, height=200, disabled=True, key="orig_display")
                        
                        # Original metrics
                        st.write("**Original Metrics:**")
                        orig_words = len(original_prompt.split())
                        orig_chars = len(original_prompt)
                        st.write(f"• Words: {orig_words}")
                        st.write(f"• Characters: {orig_chars}")
                        st.write(f"• Complexity: Simple")
                    
                    with col2:
                        st.subheader("✨ AI-Refined Prompt")
                        st.text_area("", value=refined_prompt, height=200, disabled=True, key="refined_display")
                        
                        # Refined metrics
                        st.write("**Refined Metrics:**")
                        refined_words = len(refined_prompt.split())
                        refined_chars = len(refined_prompt)
                        improvement_ratio = refined_words / orig_words if orig_words > 0 else 1
                        st.write(f"• Words: {refined_words}")
                        st.write(f"• Characters: {refined_chars}")
                        st.write(f"• Enhancement: {improvement_ratio:.1f}x more detailed")
                        
                        # Quality indicators
                        quality_score = min(
                            (refined_words / orig_words) * 20 + 
                            (50 if "verification" in refined_prompt.lower() else 0) +
                            (30 if any(step in refined_prompt.lower() for step in ["1.", "step", "•"]) else 0),
                            100
                        )
                        st.progress(quality_score / 100)
                        st.write(f"Quality Score: {quality_score:.0f}/100")
                    
                    # Action buttons
                    st.subheader("🚀 Actions")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("🚀 Execute Refined Task", type="primary"):
                            st.session_state.current_refined_prompt = refined_prompt
                            st.session_state.refined_prompts[prompt_id]["executed"] = True
                            st.success("✅ Refined prompt loaded for execution!")
                            st.info("👈 Switch to 'Execute Task' tab to run this refined prompt")
                    
                    with col2:
                        if st.button("📋 Copy Refined"):
                            st.code(refined_prompt, language="markdown")
                            st.success("📋 Use Ctrl+C to copy from the code block above")
                    
                    with col3:
                        if st.button("💾 Save to History"):
                            st.session_state.refined_prompts[prompt_id]["saved"] = True
                            st.success("💾 Saved to prompt history!")
                    
                    with col4:
                        if st.button("🔄 Re-refine"):
                            # Re-run refinement with different settings
                            st.info("🔄 Adjust settings above and click 'Refine' again")
                
                except Exception as e:
                    st.error(f"❌ Refinement failed: {e}")
                    st.info("💡 Try a different LLM provider or check your API connection")
                    
                    # Show fallback option
                    if st.button("🔧 Use Basic Refinement"):
                        basic_refined = st.session_state.prompt_refiner._fallback_refinement(original_prompt)
                        st.info("Using rule-based fallback refinement:")
                        st.code(basic_refined)
    
    # Enhanced prompt history
    if st.session_state.refined_prompts:
        st.divider()
        st.subheader("📚 Prompt History & Library")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            history_filter = st.selectbox(
                "Filter by:",
                ["All", "Recent", "Executed", "Saved", "High Quality"],
                index=0
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Newest", "Oldest", "Quality", "Complexity"],
                index=0
            )
        
        with col3:
            show_count = st.number_input(
                "Show items:",
                min_value=5,
                max_value=50,
                value=10
            )
        
        # Apply filters and sorting
        filtered_prompts = list(st.session_state.refined_prompts.items())
        
        if history_filter == "Recent":
            filtered_prompts = filtered_prompts[-7:]  # Last 7 days worth
        elif history_filter == "Executed":
            filtered_prompts = [(k, v) for k, v in filtered_prompts if v.get("executed", False)]
        elif history_filter == "Saved":
            filtered_prompts = [(k, v) for k, v in filtered_prompts if v.get("saved", False)]
        elif history_filter == "High Quality":
            # Filter by quality indicators
            filtered_prompts = [(k, v) for k, v in filtered_prompts 
                              if len(v.get("refined", "").split()) > len(v.get("original", "").split()) * 2]
        
        # Sort
        if sort_by == "Newest":
            filtered_prompts = sorted(filtered_prompts, key=lambda x: x[1].get("timestamp", datetime.now()), reverse=True)
        elif sort_by == "Oldest":
            filtered_prompts = sorted(filtered_prompts, key=lambda x: x[1].get("timestamp", datetime.now()))
        elif sort_by == "Quality":
            filtered_prompts = sorted(filtered_prompts, 
                                    key=lambda x: len(x[1].get("refined", "").split()) / max(len(x[1].get("original", "").split()), 1), 
                                    reverse=True)
        
        filtered_prompts = filtered_prompts[:show_count]
        
        # Display history items
        for prompt_id, prompt_data in filtered_prompts:
            timestamp_str = prompt_data['timestamp'].strftime('%Y-%m-%d %H:%M')
            original_preview = prompt_data['original'][:50] + "..." if len(prompt_data['original']) > 50 else prompt_data['original']
            
            # Status indicators
            status_icons = []
            if prompt_data.get("executed", False):
                status_icons.append("🚀")
            if prompt_data.get("saved", False):
                status_icons.append("💾")
            
            status_str = " ".join(status_icons) if status_icons else "📝"
            
            with st.expander(f"{status_str} {timestamp_str} - {original_preview}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original:**")
                    st.text_area("", value=prompt_data['original'], height=100, disabled=True, key=f"hist_orig_{prompt_id}")
                    
                with col2:
                    st.write("**Refined:**")
                    st.text_area("", value=prompt_data['refined'], height=100, disabled=True, key=f"hist_refined_{prompt_id}")
                
                # Enhanced metadata
                st.write("**Refinement Details:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"• Style: {prompt_data.get('style', 'N/A')}")
                    st.write(f"• Focus: {prompt_data.get('focus', 'N/A')}")
                
                with col2:
                    st.write(f"• Provider: {prompt_data.get('provider', 'N/A')}")
                    st.write(f"• Intent: {prompt_data.get('intent', 'N/A')}")
                
                with col3:
                    improvement = len(prompt_data['refined'].split()) / max(len(prompt_data['original'].split()), 1)
                    st.write(f"• Improvement: {improvement:.1f}x")
                    st.write(f"• UI Elements: {len(prompt_data.get('ui_elements', []))}")
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🚀 Execute", key=f"exec_hist_{prompt_id}"):
                        st.session_state.current_refined_prompt = prompt_data['refined']
                        st.session_state.refined_prompts[prompt_id]["executed"] = True
                        st.success("✅ Loaded for execution!")
                
                with col2:
                    if st.button("📋 Copy", key=f"copy_hist_{prompt_id}"):
                        st.code(prompt_data['refined'])
                        st.success("📋 Ready to copy!")
                
                with col3:
                    if st.button("🔄 Re-refine", key=f"rerefine_hist_{prompt_id}"):
                        st.session_state.rerefine_prompt = prompt_data['original']
                        st.info("🔄 Original prompt loaded for re-refinement")
                        st.rerun()
                
                with col4:
                    if st.button("🗑️ Delete", key=f"del_hist_{prompt_id}"):
                        del st.session_state.refined_prompts[prompt_id]
                        st.success("🗑️ Deleted from history")
                        st.rerun()

def execute_task_tab():
    """Enhanced task execution interface with advanced prompt refinement integration"""
    st.header("🎯 Execute QA Task")
    
    if st.session_state.qa_manager is None:
        st.warning("⚠️ Please initialize the system first using the sidebar.")
        return
    
    # System overview with enhanced metrics
    status = get_system_status()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_text = "READY" if status["initialized"] else "NOT READY"
        delta_text = "Operational" if status["initialized"] else "Initialize First"
        st.metric("System Status", status_text, delta=delta_text)
    
    with col2:
        mode = status["llm_interface"]
        agent_s_status = "Active" if status["agent_s"] else "Mock"
        st.metric("LLM Mode", mode, delta=f"Agent-S: {agent_s_status}")
    
    with col3:
        agent_count = status["agent_s_details"].get("agent_s_agents", 0) if status["agent_s"] else 4
        st.metric("Agents", f"{agent_count} Active", delta="Multi-Agent")
    
    with col4:
        total_tests = status["tests_completed"]
        st.metric("Tests Completed", total_tests)
    
    st.divider()
    
    # Enhanced task definition with refined prompt integration
    st.subheader("📋 Define QA Task")
    
    # Check if we have a refined prompt from Advanced Prompt Studio
    if hasattr(st.session_state, 'current_refined_prompt'):
        st.success("✨ Using AI-refined prompt from Advanced Prompt Studio!")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            task_description = st.text_area(
                "AI-Refined task description:",
                value=st.session_state.current_refined_prompt,
                height=200,
                help="This prompt was refined by AI for optimal execution accuracy"
            )
        
        with col2:
            st.write("**Refined Prompt Benefits:**")
            st.info("✅ Detailed steps\n✅ Clear objectives\n✅ Verification included\n✅ Error handling\n✅ Android-optimized")
            
            if st.button("🔄 Clear Refined Prompt"):
                delattr(st.session_state, 'current_refined_prompt')
                st.rerun()
            
            if st.button("🎨 Edit in Studio"):
                st.info("👈 Use 'Advanced Prompt Studio' tab to modify")
    
    else:
        # Enhanced input methods
        input_method = st.radio(
            "How would you like to define your task?",
            [
                "📝 Write Custom Prompt", 
                "📋 Choose Predefined Task",
                "📚 Load from Prompt History"
            ],
            horizontal=True
        )
        
        if input_method == "📝 Write Custom Prompt":
            col1, col2 = st.columns([3, 1])
            
            with col1:
                task_description = st.text_area(
                    "Enter your custom task description:",
                    placeholder="Describe the QA task you want to execute...\nExample: 'Test WiFi toggle in Settings' or 'Verify calculator operations'",
                    height=120,
                    help="Enter any description - use AI refinement for best results"
                )
            
            with col2:
                st.write("**AI Enhancement**")
                
                if st.button("✨ Quick Refine", help="Fast AI improvement", type="primary"):
                    if task_description.strip() and st.session_state.llm_interface:
                        with st.spinner("Quick refining..."):
                            refined = st.session_state.prompt_refiner.generate_context_aware_refinement(
                                task_description, style="step_by_step", focus="general"
                            )
                            if refined != task_description:
                                st.session_state.current_refined_prompt = refined
                                st.success("✅ Quick refinement complete!")
                                st.rerun()
                            else:
                                st.info("Prompt is already well-structured!")
                    elif not st.session_state.llm_interface:
                        st.warning("Connect LLM provider first")
                    else:
                        st.warning("Enter a task description first")
                
                if st.button("🎨 Advanced Studio", help="Full refinement options"):
                    if task_description.strip():
                        # Store current prompt for studio
                        st.session_state.studio_input_prompt = task_description
                        st.info("👈 Switch to 'Advanced Prompt Studio' tab")
                    else:
                        st.warning("Enter a task description first")
                
                st.divider()
                
                if st.button("🤖 Auto-Enhance", help="Automatic best-effort refinement"):
                    if task_description.strip():
                        with st.spinner("Auto-enhancing..."):
                            # Use fallback refinement if LLM not available
                            if st.session_state.llm_interface:
                                refined = st.session_state.prompt_refiner.generate_context_aware_refinement(
                                    task_description, style="detailed", focus="general"
                                )
                            else:
                                refined = st.session_state.prompt_refiner._fallback_refinement(task_description)
                            
                            task_description = refined
                            st.success("✅ Auto-enhanced!")
        
        elif input_method == "📋 Choose Predefined Task":
            # Enhanced predefined tasks with categories
            predefined_categories = {
                "Settings & System": [
                    "Test turning Wi-Fi on and off in Settings → Network & Internet → Internet",
                    "Navigate to Settings and open Display settings to adjust brightness",
                    "Toggle airplane mode on and off via quick settings panel",
                    "Navigate to Bluetooth settings and verify connection status",
                    "Test battery optimization settings and power management"
                ],
                "App Testing": [
                    "Open Calculator app and perform basic arithmetic calculation (5+3)",
                    "Create a new alarm for 7:30 AM using Clock app",
                    "Navigate to Storage settings and check available space",
                    "Test Camera app basic photo capture functionality",
                    "Verify Contacts app add new contact workflow"
                ],
                "System Navigation": [
                    "Test notification panel access and quick settings toggles",
                    "Verify home screen navigation and app drawer access",
                    "Test recent apps functionality and app switching",
                    "Validate back button navigation across multiple screens",
                    "Test split-screen multitasking functionality"
                ]
            }
            
            # Category selection
            selected_category = st.selectbox(
                "Select task category:",
                list(predefined_categories.keys()),
                index=0
            )
            
            # Task selection within category
            selected_task = st.selectbox(
                f"Select task from {selected_category}:",
                predefined_categories[selected_category],
                index=0
            )
            
            task_description = st.text_area(
                "Task description (editable):",
                value=selected_task,
                height=120,
                help="You can edit this predefined task or use it as-is"
            )
            
            # Option to refine predefined tasks
            if st.button("✨ Enhance Predefined Task") and st.session_state.llm_interface:
                with st.spinner("Enhancing predefined task..."):
                    refined = st.session_state.prompt_refiner.generate_context_aware_refinement(
                        task_description, style="detailed", focus="general"
                    )
                    if refined != task_description:
                        st.session_state.current_refined_prompt = refined
                        st.success("✅ Predefined task enhanced!")
                        st.rerun()
        
        else:  # Load from Prompt History
            if st.session_state.refined_prompts:
                # Show available refined prompts
                prompt_options = {}
                for prompt_id, prompt_data in st.session_state.refined_prompts.items():
                    timestamp = prompt_data['timestamp'].strftime('%m/%d %H:%M')
                    original_preview = prompt_data['original'][:30] + "..." if len(prompt_data['original']) > 30 else prompt_data['original']
                    prompt_options[f"{timestamp} - {original_preview}"] = prompt_id
                
                if prompt_options:
                    selected_prompt_display = st.selectbox(
                        "Select from refined prompt history:",
                        list(prompt_options.keys()),
                        index=0
                    )
                    
                    selected_prompt_id = prompt_options[selected_prompt_display]
                    selected_prompt_data = st.session_state.refined_prompts[selected_prompt_id]
                    
                    task_description = st.text_area(
                        "Selected refined prompt:",
                        value=selected_prompt_data['refined'],
                        height=150,
                        help="Loaded from prompt history - ready to execute"
                    )
                    
                    # Show metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"Style: {selected_prompt_data.get('style', 'N/A')}")
                    with col2:
                        st.info(f"Focus: {selected_prompt_data.get('focus', 'N/A')}")
                    with col3:
                        st.info(f"Provider: {selected_prompt_data.get('provider', 'N/A')}")
                else:
                    st.info("No refined prompts in history yet")
                    task_description = ""
            else:
                st.info("No prompt history available. Create some refined prompts first!")
                task_description = ""
    
    # Android World task configuration
    st.subheader("⚙️ Execution Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        android_tasks = getattr(config, 'ANDROID_WORLD_TASKS', [
            "settings_wifi", "clock_alarm", "calculator_basic", 
            "contacts_add", "email_search"
        ])
        
        android_world_task = st.selectbox(
            "Android World Task Type:",
            android_tasks,
            index=0,
            help="Select the underlying android_world task type that best matches your test"
        )
    
    with col2:
        execution_mode = st.selectbox(
            "Execution Mode:",
            ["Standard", "Detailed Logging", "Fast Mode", "Safe Mode"],
            index=0,
            help="Choose execution mode based on your testing needs"
        )
    
    # Execution parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_steps = st.number_input(
            "Max Steps", 
            min_value=5, 
            max_value=50, 
            value=20,
            help="Maximum number of execution steps"
        )
    
    with col2:
        timeout = st.number_input(
            "Timeout (seconds)", 
            min_value=30, 
            max_value=600, 
            value=120,
            help="Maximum execution time"
        )
    
    with col3:
        retry_attempts = st.number_input(
            "Retry Attempts",
            min_value=0,
            max_value=5,
            value=2,
            help="Number of retry attempts on failure"
        )
    
    with col4:
        enable_debug = st.checkbox(
            "Debug Mode", 
            value=False,
            help="Enable detailed debug logging"
        )
    
    # Enhanced execution controls
    st.subheader("🚀 Execution Controls")
    
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        execute_button = st.button(
            "🚀 Execute Task", 
            type="primary", 
            disabled=not task_description.strip(),
            help="Execute the defined task with current configuration"
        )
    
    with col2:
        if st.button("⚡ Quick Test", help="Run with minimal steps for quick testing"):
            max_steps = 10
            timeout = 60
            execute_button = True
    
    with col3:
        if st.button("🔍 Dry Run", help="Simulate execution without actual device interaction"):
            st.info("🔍 Dry run mode - would execute with current settings")
            # TODO: Implement dry run functionality
    
    with col4:
        if st.button("💾 Save Config", help="Save current execution configuration"):
            config_data = {
                "task_description": task_description,
                "android_world_task": android_world_task,
                "max_steps": max_steps,
                "timeout": timeout,
                "execution_mode": execution_mode
            }
            st.session_state.saved_configs = getattr(st.session_state, 'saved_configs', [])
            st.session_state.saved_configs.append(config_data)
            st.success("💾 Configuration saved!")
    
    with col5:
        if st.button("🗑️ Clear History"):
            st.session_state.execution_history = []
            st.success("🗑️ History cleared!")
            st.rerun()
    
    # Execute task with enhanced error handling
    if execute_button and task_description.strip():
        execute_task_with_enhanced_system(
            task_description, android_world_task, max_steps, timeout, 
            execution_mode, retry_attempts, enable_debug
        )

def execute_task_with_enhanced_system(task_description: str, android_world_task: str, 
                                     max_steps: int, timeout: int, execution_mode: str,
                                     retry_attempts: int, debug: bool):
    """Execute task with enhanced multi-agent system and advanced error handling"""
    
    # Create containers for enhanced progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.info(f"🔄 Executing task with {execution_mode.lower()} mode...")
        
        # Enhanced progress indicators
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            progress_bar = st.progress(0)
        with col2:
            status_text = st.empty()
        with col3:
            phase_timer = st.empty()
        
        phase_container = st.container()
        phase_details = phase_container.expander("Execution Details", expanded=debug)
        
        start_time = time.time()
        
        try:
            # Phase 1: Enhanced Planning
            phase_start = time.time()
            status_text.text("Phase 1/5: Planning")
            progress_bar.progress(0.2)
            phase_timer.text(f"⏱️ 0s")
            
            with phase_details:
                st.write("**Enhanced Planning**: Analyzing task and creating detailed execution plan")
                if debug:
                    st.code(f"DEBUG: Planning '{task_description}' with {max_steps} max steps")
                    st.code(f"DEBUG: Mode: {execution_mode}, Retries: {retry_attempts}")
            
            # Handle asyncio properly
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Phase 2: Environment Setup
            phase_start = time.time()
            status_text.text("Phase 2/5: Setup")
            progress_bar.progress(0.4)
            
            with phase_details:
                st.write("**Environment Setup**: Initializing agents and environment")
                if debug:
                    st.code(f"DEBUG: Using android_world task '{android_world_task}'")
                    st.code(f"DEBUG: LLM Provider: {st.session_state.llm_interface.__class__.__name__}")
            
            # Phase 3: Enhanced Execution
            phase_start = time.time()
            status_text.text("Phase 3/5: Executing")
            progress_bar.progress(0.6)
            
            with phase_details:
                st.write("**Enhanced Execution**: Performing UI interactions with retry logic")
            
            # Configure enhanced test settings
            test_config = {
                "goal": task_description,
                "android_world_task": android_world_task,
                "max_steps": max_steps,
                "timeout": timeout,
                "execution_mode": execution_mode,
                "retry_attempts": retry_attempts,
                "debug_mode": debug,
                "enhanced_features": True
            }
            
            # Execute with retry logic
            result = None
            last_error = None
            
            for attempt in range(retry_attempts + 1):
                try:
                    if attempt > 0:
                        with phase_details:
                            st.warning(f"Retry attempt {attempt}/{retry_attempts}")
                        time.sleep(2)  # Brief delay between retries
                    
                    result = loop.run_until_complete(
                        st.session_state.qa_manager.run_qa_test(test_config)
                    )
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    if attempt < retry_attempts:
                        with phase_details:
                            st.warning(f"Attempt {attempt + 1} failed: {e}")
                        continue
                    else:
                        raise e
            
            # Phase 4: Enhanced Verification
            phase_start = time.time()
            status_text.text("Phase 4/5: Verifying")
            progress_bar.progress(0.8)
            
            with phase_details:
                st.write("**Enhanced Verification**: Validating results with comprehensive checks")
                if debug and result:
                    st.code(f"DEBUG: Final result: {result.final_result}")
                    st.code(f"DEBUG: Actions executed: {len(getattr(result, 'actions', []))}")
            
            # Phase 5: Enhanced Analysis
            phase_start = time.time()
            status_text.text("Phase 5/5: Analyzing")
            progress_bar.progress(1.0)
            
            with phase_details:
                st.write("**Enhanced Analysis**: Generating comprehensive report with insights")
            
            # Process result with enhanced fields
            processed_result = {
                "test_id": getattr(result, 'test_id', f"test_{int(time.time())}"),
                "task_description": task_description,
                "task_name": getattr(result, 'task_name', task_description),
                "android_world_task": android_world_task,
                "success": result.final_result == "PASS",
                "final_result": result.final_result,
                "total_time": time.time() - start_time,
                "total_steps": len(getattr(result, 'actions', [])),
                "bug_detected": getattr(result, 'bug_detected', False),
                "supervisor_feedback": getattr(result, 'supervisor_feedback', "No feedback available") or "No feedback available",
                "actions_summary": {
                    "total": len(getattr(result, 'actions', [])),
                    "successful": sum(1 for a in getattr(result, 'actions', []) if getattr(a, 'success', False)),
                    "success_rate": sum(1 for a in getattr(result, 'actions', []) if getattr(a, 'success', False)) / len(getattr(result, 'actions', [])) if getattr(result, 'actions', []) else 0
                },
                "timestamp": datetime.now(),
                "start_time": start_time,
                "end_time": time.time(),
                "actions": getattr(result, 'actions', []),
                "agent_s_enhanced": getattr(result, 'agent_s_enhanced', True),
                "llm_provider": st.session_state.llm_interface.__class__.__name__ if st.session_state.llm_interface else "Mock",
                "prompt_refined": hasattr(st.session_state, 'current_refined_prompt'),
                "execution_mode": execution_mode,
                "retry_attempts_used": attempt,
                "debug_mode": debug,
                "enhanced_execution": True
            }
            
            elapsed_time = time.time() - start_time
            phase_timer.text(f"⏱️ {elapsed_time:.1f}s")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            st.error(f"❌ Enhanced execution failed: {e}")
            
            if debug:
                import traceback
                st.code(traceback.format_exc())
            
            # Create enhanced error result
            processed_result = {
                "test_id": f"error_test_{int(time.time())}",
                "task_description": task_description,
                "task_name": f"Failed: {task_description}",
                "android_world_task": android_world_task,
                "success": False,
                "final_result": "ERROR",
                "total_time": elapsed_time,
                "total_steps": 0,
                "bug_detected": False,
                "supervisor_feedback": f"Enhanced execution error: {str(e)}",
                "actions_summary": {"total": 0, "successful": 0, "success_rate": 0},
                "error": str(e),
                "timestamp": datetime.now(),
                "start_time": start_time,
                "end_time": time.time(),
                "actions": [],
                "agent_s_enhanced": False,
                "llm_provider": "Error",
                "prompt_refined": False,
                "execution_mode": execution_mode,
                "retry_attempts_used": retry_attempts,
                "debug_mode": debug,
                "enhanced_execution": True
            }
    
    # Add result to history
    st.session_state.execution_history.append(processed_result)
    
    # Clean up refined prompt after execution
    if hasattr(st.session_state, 'current_refined_prompt'):
        delattr(st.session_state, 'current_refined_prompt')
    
    # Display enhanced results
    with results_container:
        if processed_result["success"]:
            st.success("✅ Task executed successfully with enhanced system!")
        else:
            st.error("❌ Enhanced task execution failed!")
            if "error" in processed_result:
                st.error(f"Error: {processed_result['error']}")
        
        # Enhanced metrics display
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Duration", f"{processed_result['total_time']:.2f}s")
        with col2:
            st.metric("Steps", processed_result['total_steps'])
        with col3:
            success_rate = processed_result['actions_summary']['success_rate']
            st.metric("Success Rate", f"{success_rate:.1%}")
        with col4:
            mode_badge = {"Standard": "⚙️", "Detailed Logging": "📝", "Fast Mode": "⚡", "Safe Mode": "🛡️"}[execution_mode]
            st.metric("Mode", f"{mode_badge} {execution_mode}")
        with col5:
            retry_badge = "🔄" if processed_result.get("retry_attempts_used", 0) > 0 else "✅"
            st.metric("Retries", f"{retry_badge} {processed_result.get('retry_attempts_used', 0)}")
        with col6:
            enhanced_badge = "🚀" if processed_result.get("enhanced_execution", False) else "⚙️"
            llm_name = processed_result.get('llm_provider', 'Mock')[:8]
            st.metric("LLM", f"{enhanced_badge} {llm_name}")
        
        # Enhanced detailed results
        with st.expander("Enhanced Detailed Results", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Execution Summary")
                st.write(f"**Test ID**: {processed_result['test_id']}")
                st.write(f"**Task**: {processed_result['task_description'][:100]}{'...' if len(processed_result['task_description']) > 100 else ''}")
                st.write(f"**Android Task**: {processed_result['android_world_task']}")
                st.write(f"**Execution Mode**: {processed_result.get('execution_mode', 'Standard')}")
                st.write(f"**Enhanced Features**: {'Yes' if processed_result.get('enhanced_execution', False) else 'No'}")
                st.write(f"**Agent-S Enhanced**: {'Yes' if processed_result.get('agent_s_enhanced', False) else 'No'}")
                st.write(f"**LLM Provider**: {processed_result.get('llm_provider', 'Mock')}")
                st.write(f"**Prompt Refined**: {'Yes' if processed_result.get('prompt_refined', False) else 'No'}")
                st.write(f"**Debug Mode**: {'Yes' if processed_result.get('debug_mode', False) else 'No'}")
                st.write(f"**Retry Attempts**: {processed_result.get('retry_attempts_used', 0)}")
                
                if processed_result['supervisor_feedback']:
                    st.write("**Enhanced Supervisor Feedback**:")
                    st.info(processed_result['supervisor_feedback'])
            
            with col2:
                st.subheader("📈 Performance Analysis")
                actions = processed_result['actions_summary']
                
                if actions['total'] > 0:
                    success_pct = actions['successful'] / actions['total'] * 100
                    st.progress(success_pct / 100)
                    st.write(f"**Successful Actions**: {actions['successful']}/{actions['total']} ({success_pct:.1f}%)")
                    
                    # Performance rating
                    if success_pct >= 90:
                        st.success("🌟 Excellent Performance")
                    elif success_pct >= 75:
                        st.info("👍 Good Performance") 
                    elif success_pct >= 50:
                        st.warning("⚠️ Fair Performance")
                    else:
                        st.error("🔧 Needs Improvement")
                else:
                    st.write("No actions recorded")
                
                # Enhanced quality indicators
                quality_factors = []
                
                if processed_result.get('prompt_refined', False):
                    quality_factors.append("✅ AI-Refined Prompt")
                
                if processed_result.get('agent_s_enhanced', False):
                    quality_factors.append("✅ Agent-S Architecture")
                
                if processed_result.get('enhanced_execution', False):
                    quality_factors.append("✅ Enhanced Execution")
                
                if not processed_result.get('debug_mode', False):
                    quality_factors.append("✅ Production Mode")
                
                if quality_factors:
                    st.write("**Quality Factors:**")
                    for factor in quality_factors:
                        st.write(factor)
        
        # Show enhanced raw result data for debugging
        if debug:
            with st.expander("Enhanced Raw Execution Data"):
                st.json(processed_result)

# Keep the existing dashboard_tab, history_tab, and advanced_tab functions
# but add enhanced LLM provider and prompt refinement information

def dashboard_tab():
    """Enhanced dashboard with comprehensive metrics and advanced visualizations"""
    st.header("📊 Enhanced System Dashboard")
    
    if not st.session_state.execution_history:
        st.info("No execution history available. Execute some tasks first!")
        return
    
    # Convert history to DataFrame with enhanced columns
    df = pd.DataFrame(st.session_state.execution_history)
    
    # Enhanced key performance indicators
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        success_rate = df['success'].mean()
        st.metric("Overall Success Rate", f"{success_rate:.1%}")
    
    with col2:
        avg_duration = df['total_time'].mean()
        st.metric("Avg Duration", f"{avg_duration:.1f}s")
    
    with col3:
        avg_steps = df['total_steps'].mean()
        st.metric("Avg Steps", f"{avg_steps:.1f}")
    
    with col4:
        agent_s_tests = df.get('agent_s_enhanced', pd.Series([False] * len(df))).sum()
        st.metric("Agent-S Tests", f"{agent_s_tests}/{len(df)}")
    
    with col5:
        refined_prompts = df.get('prompt_refined', pd.Series([False] * len(df))).sum()
        st.metric("AI-Refined Prompts", f"{refined_prompts}/{len(df)}")
    
    with col6:
        enhanced_tests = df.get('enhanced_execution', pd.Series([False] * len(df))).sum()
        st.metric("Enhanced Tests", f"{enhanced_tests}/{len(df)}")
    
    st.divider()
    
    # Enhanced performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Success Rate by LLM Provider")
        if 'llm_provider' in df.columns:
            provider_success = df.groupby('llm_provider')['success'].mean().reset_index()
            fig1 = px.bar(
                provider_success,
                x='llm_provider',
                y='success',
                title="Success Rate by LLM Provider",
                labels={'llm_provider': 'LLM Provider', 'success': 'Success Rate'},
                color='success',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No LLM provider data available")
    
    with col2:
        st.subheader("Enhanced vs Standard Execution")
        if 'enhanced_execution' in df.columns:
            enhanced_comparison = df.groupby('enhanced_execution')['success'].mean().reset_index()
            enhanced_comparison['execution_type'] = enhanced_comparison['enhanced_execution'].map({True: 'Enhanced', False: 'Standard'})
            
            fig2 = px.bar(
                enhanced_comparison,
                x='execution_type',
                y='success',
                title="Success Rate: Enhanced vs Standard Execution",
                labels={'execution_type': 'Execution Type', 'success': 'Success Rate'},
                color='success',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No enhanced execution data available")
    
    # Additional enhanced charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Refined vs Original Prompts")
        if 'prompt_refined' in df.columns:
            refined_comparison = df.groupby('prompt_refined')['success'].mean().reset_index()
            refined_comparison['prompt_type'] = refined_comparison['prompt_refined'].map({True: 'AI-Refined', False: 'Original'})
            
            fig3 = px.pie(
                refined_comparison,
                values='success',
                names='prompt_type',
                title="Success Distribution: Refined vs Original Prompts"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No prompt refinement data available")
    
    with col2:
        st.subheader("Execution Time Trends")
        if len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            fig4 = px.line(
                df_sorted,
                x='timestamp',
                y='total_time',
                title="Execution Time Over Time",
                labels={'timestamp': 'Time', 'total_time': 'Duration (seconds)'}
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Need more data points for trend analysis")
    
    # Enhanced recent executions table
    st.subheader("Recent Enhanced Executions")
    display_df = df[['task_description', 'success', 'total_time', 'total_steps', 'timestamp']].copy()
    
    # Add enhanced columns if available
    if 'agent_s_enhanced' in df.columns:
        display_df['agent_s'] = df['agent_s_enhanced'].map({True: '🤖', False: '⚙️'})
    
    if 'llm_provider' in df.columns:
        display_df['llm'] = df['llm_provider']
    
    if 'prompt_refined' in df.columns:
        display_df['refined'] = df['prompt_refined'].map({True: '✨', False: '📝'})
    
    if 'enhanced_execution' in df.columns:
        display_df['enhanced'] = df['enhanced_execution'].map({True: '🚀', False: '⚙️'})
    
    if 'execution_mode' in df.columns:
        display_df['mode'] = df['execution_mode']
    
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['success'] = display_df['success'].map({True: '✅ Pass', False: '❌ Fail'})
    display_df = display_df.sort_values('timestamp', ascending=False).head(15)
    
    column_config = {
        "task_description": "Task Description",
        "success": st.column_config.TextColumn("Result"),
        "total_time": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
        "total_steps": "Steps",
        "timestamp": "Executed At"
    }
    
    if 'agent_s' in display_df.columns:
        column_config["agent_s"] = st.column_config.TextColumn("Agent")
    if 'llm' in display_df.columns:
        column_config["llm"] = st.column_config.TextColumn("LLM")
    if 'refined' in display_df.columns:
        column_config["refined"] = st.column_config.TextColumn("Prompt")
    if 'enhanced' in display_df.columns:
        column_config["enhanced"] = st.column_config.TextColumn("Enhanced")
    if 'mode' in display_df.columns:
        column_config["mode"] = st.column_config.TextColumn("Mode")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config=column_config
    )

def history_tab():
    """Enhanced execution history and analysis with comprehensive LLM information"""
    st.header("📋 Enhanced Execution History")
    
    if not st.session_state.execution_history:
        st.info("No execution history available.")
        return
    
    # Enhanced filters
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        success_filter = st.selectbox("Filter by result:", ["All", "Successful", "Failed"])
    
    with col2:
        # Enhanced LLM Provider filter
        df = pd.DataFrame(st.session_state.execution_history)
        if 'llm_provider' in df.columns:
            llm_providers = ['All'] + list(df['llm_provider'].unique())
            llm_filter = st.selectbox("Filter by LLM:", llm_providers)
        else:
            llm_filter = "All"
    
    with col3:
        # Execution mode filter
        if 'execution_mode' in df.columns:
            exec_modes = ['All'] + list(df['execution_mode'].unique())
            mode_filter = st.selectbox("Filter by mode:", exec_modes)
        else:
            mode_filter = "All"
    
    with col4:
        max_results = st.number_input("Max results:", min_value=5, max_value=100, value=20)
    
    with col5:
        sort_order = st.selectbox("Sort by:", ["Newest First", "Oldest First", "Duration", "Success Rate", "Enhanced Features"])
    
    # Apply enhanced filters
    filtered_history = st.session_state.execution_history.copy()
    
    if success_filter == "Successful":
        filtered_history = [h for h in filtered_history if h.get('success', False)]
    elif success_filter == "Failed":
        filtered_history = [h for h in filtered_history if not h.get('success', False)]
    
    if llm_filter != "All":
        filtered_history = [h for h in filtered_history if h.get('llm_provider') == llm_filter]
    
    if mode_filter != "All":
        filtered_history = [h for h in filtered_history if h.get('execution_mode') == mode_filter]
    
    # Apply enhanced sorting
    if sort_order == "Newest First":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
    elif sort_order == "Oldest First":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('timestamp', datetime.now()))
    elif sort_order == "Duration":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('total_time', 0), reverse=True)
    elif sort_order == "Success Rate":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('actions_summary', {}).get('success_rate', 0), reverse=True)
    elif sort_order == "Enhanced Features":
        filtered_history = sorted(filtered_history, key=lambda x: (
            x.get('enhanced_execution', False),
            x.get('prompt_refined', False),
            x.get('agent_s_enhanced', False)
        ), reverse=True)
    
    filtered_history = filtered_history[:max_results]
    
    # Display enhanced execution details
    for i, execution in enumerate(filtered_history):
        status_icon = "✅" if execution.get('success', False) else "❌"
        agent_s_icon = "🤖" if execution.get('agent_s_enhanced', False) else "⚙️"
        refined_icon = "✨" if execution.get('prompt_refined', False) else "📝"
        enhanced_icon = "🚀" if execution.get('enhanced_execution', False) else "⚙️"
        
        task_desc = execution.get('task_description', 'Unknown task')
        timestamp = execution.get('timestamp', datetime.now())
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp)
        llm_provider = execution.get('llm_provider', 'Unknown')
        exec_mode = execution.get('execution_mode', 'Standard')
        
        with st.expander(f"{status_icon} {enhanced_icon} {agent_s_icon} {refined_icon} {task_desc[:60]}... - {timestamp_str}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Basic Information:**")
                test_id = execution.get('test_id', execution.get('task_name', f'unknown_{i}'))
                st.write(f"Test ID: `{test_id}`")
                st.write(f"Duration: {execution.get('total_time', 0):.2f}s")
                st.write(f"Steps: {execution.get('total_steps', 0)}")
                st.write(f"Result: {execution.get('final_result', 'Unknown')}")
                st.write(f"Execution Mode: {exec_mode}")
                retry_count = execution.get('retry_attempts_used', 0)
                st.write(f"Retries Used: {retry_count}")
            
            with col2:
                st.write("**Performance Metrics:**")
                actions = execution.get('actions_summary', {})
                success_rate = actions.get('success_rate', 0)
                st.write(f"Action Success: {success_rate:.1%}")
                st.write(f"Bug Detected: {'Yes' if execution.get('bug_detected', False) else 'No'}")
                st.write(f"Android Task: {execution.get('android_world_task', 'Unknown')}")
                st.write(f"Agent-S Enhanced: {'Yes' if execution.get('agent_s_enhanced', False) else 'No'}")
                st.write(f"Enhanced Execution: {'Yes' if execution.get('enhanced_execution', False) else 'No'}")
                st.write(f"Debug Mode: {'Yes' if execution.get('debug_mode', False) else 'No'}")
            
            with col3:
                st.write("**AI & LLM Information:**")
                st.write(f"LLM Provider: {llm_provider}")
                st.write(f"Prompt Refined: {'Yes' if execution.get('prompt_refined', False) else 'No'}")
                
                # Enhanced quality score
                quality_score = 0
                if execution.get('enhanced_execution', False):
                    quality_score += 25
                if execution.get('prompt_refined', False):
                    quality_score += 25
                if execution.get('agent_s_enhanced', False):
                    quality_score += 25
                if success_rate > 0.8:
                    quality_score += 25
                
                st.progress(quality_score / 100)
                st.write(f"Quality Score: {quality_score}/100")
                
                feedback = execution.get('supervisor_feedback', '')
                if feedback:
                    st.write("**Feedback:**")
                    if len(feedback) > 100:
                        st.write(f"{feedback[:100]}...")
                    else:
                        st.write(feedback)
                else:
                    st.write("**Feedback:** No feedback available")
            
            # Enhanced task description display
            full_task = execution.get('task_description', '')
            if len(full_task) > 100:
                with st.expander("View Full Task Description"):
                    st.text_area("", value=full_task, height=100, disabled=True, key=f"full_task_{i}")
            
            # Enhanced details with performance analysis
            if st.checkbox(f"Show enhanced analysis", key=f"analysis_{i}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Performance Analysis:**")
                    if success_rate >= 0.9:
                        st.success("🌟 Excellent execution quality")
                    elif success_rate >= 0.7:
                        st.info("👍 Good execution quality")
                    elif success_rate >= 0.5:
                        st.warning("⚠️ Average execution quality")
                    else:
                        st.error("🔧 Poor execution quality")
                    
                    # Feature usage analysis
                    features_used = []
                    if execution.get('enhanced_execution'):
                        features_used.append("Enhanced Execution")
                    if execution.get('prompt_refined'):
                        features_used.append("AI Prompt Refinement")
                    if execution.get('agent_s_enhanced'):
                        features_used.append("Agent-S Architecture")
                    if execution.get('debug_mode'):
                        features_used.append("Debug Mode")
                    
                    if features_used:
                        st.write("**Features Used:**")
                        for feature in features_used:
                            st.write(f"• {feature}")
                
                with col2:
                    st.write("**Raw Data:**")
                    st.json(execution, expanded=False)

def advanced_tab():
    """Enhanced advanced features and system management with comprehensive LLM management"""
    st.header("⚙️ Advanced Features & System Management")
    
    # Enhanced Agent-S system information
    st.subheader("🤖 Enhanced Agent-S System Information")
    
    status = get_system_status()
    if status["agent_s"]:
        st.success("Agent-S Architecture: ✅ Active with Enhanced Features")
        
        agent_s_details = status["agent_s_details"]
        if "details" in agent_s_details:
            st.write("**Enhanced Agent Status:**")
            
            # Create agent status table
            agent_data = []
            for agent_name, is_active in agent_s_details["details"].items():
                agent_data.append({
                    "Agent": agent_name.title() + "Agent",
                    "Type": "QAAgentS2" if is_active else "BaseAgent",
                    "Status": "✅ Active" if is_active else "❌ Inactive",
                    "Enhanced": "Yes" if is_active else "No"
                })
            
            agent_df = pd.DataFrame(agent_data)
            st.dataframe(agent_df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Agent-S Agents", f"{agent_s_details.get('agent_s_agents', 0)}/4")
        with col2:
            coordination_status = status.get("coordination_active", False)
            st.metric("Coordination", "✅ Active" if coordination_status else "⚠️ Limited")
        with col3:
            llm_integration = "✅ Integrated" if st.session_state.llm_interface else "❌ Not Connected"
            st.metric("LLM Integration", llm_integration)
    else:
        st.warning("Agent-S Architecture: ❌ Not Active")
        if "error" in status["agent_s_details"]:
            st.error(f"Error: {status['agent_s_details']['error']}")
        
        st.info("💡 Initialize the system to enable Agent-S architecture")
    
    st.divider()
    
    # ✅ ENHANCED: Comprehensive LLM Provider Management Section
    st.subheader("🧠 Comprehensive LLM Provider Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current LLM Status & Capabilities:**")
        if st.session_state.llm_interface:
            provider_name = st.session_state.llm_interface.__class__.__name__
            st.success(f"✅ Connected: {provider_name}")
            
            # Enhanced connection details
            provider_info = {
                "MockLLMInterface": {
                    "name": "Mock LLM",
                    "cost": "Free",
                    "speed": "Instant",
                    "quality": "Deterministic",
                    "features": ["Always Available", "No API Limits", "Consistent Responses"]
                },
                "OpenAIInterface": {
                    "name": "OpenAI GPT",
                    "cost": "Pay-per-use",
                    "speed": "Fast",
                    "quality": "Excellent",
                    "features": ["High Quality", "Fast Processing", "Advanced Reasoning"]
                },
                "ClaudeInterface": {
                    "name": "Anthropic Claude",
                    "cost": "Pay-per-use", 
                    "speed": "Fast",
                    "quality": "Superior",
                    "features": ["Long Context", "Excellent Reasoning", "Safety Focused"]
                },
                "GeminiInterface": {
                    "name": "Google Gemini",
                    "cost": "Free Tier + Paid",
                    "speed": "Very Fast",
                    "quality": "Good",
                    "features": ["Free Tier", "Google Integration", "Multimodal"]
                }
            }
            
            current_info = provider_info.get(provider_name, {})
            if current_info:
                st.write(f"**Provider**: {current_info['name']}")
                st.write(f"**Cost Model**: {current_info['cost']}")
                st.write(f"**Speed**: {current_info['speed']}")
                st.write(f"**Quality**: {current_info['quality']}")
                
                st.write("**Key Features:**")
                for feature in current_info.get('features', []):
                    st.write(f"• {feature}")
            
            # Enhanced connection testing
            col1a, col1b = st.columns(2)
            
            with col1a:
                if st.button("🔍 Test Connection"):
                    with st.spinner("Testing connection..."):
                        try:
                            test_response = st.session_state.llm_interface.generate_response(
                                "Test connection: respond with 'Connection successful for Android QA system testing!'"
                            )
                            if test_response.content:
                                st.success("✅ Connection working perfectly!")
                                st.info(f"Response: {test_response.content[:100]}...")
                                
                                # Show response metadata
                                if hasattr(test_response, 'usage_tokens'):
                                    st.write(f"Tokens used: {test_response.usage_tokens}")
                                if hasattr(test_response, 'cost'):
                                    st.write(f"Cost: ${test_response.cost:.6f}")
                            else:
                                st.error("❌ No response received")
                        except Exception as e:
                            st.error(f"❌ Connection test failed: {e}")
            
            with col1b:
                if st.button("🧪 Test Refinement"):
                    with st.spinner("Testing prompt refinement..."):
                        try:
                            test_prompt = "test wifi settings"
                            refined = st.session_state.prompt_refiner.generate_context_aware_refinement(
                                test_prompt, style="detailed", focus="settings"
                            )
                            if refined != test_prompt:
                                st.success("✅ Refinement working!")
                                st.write("**Original**: test wifi settings")
                                st.write(f"**Refined**: {refined[:100]}...")
                            else:
                                st.warning("⚠️ Refinement not improving prompt")
                        except Exception as e:
                            st.error(f"❌ Refinement test failed: {e}")
        else:
            st.warning("⚠️ No LLM provider connected")
            st.info("👈 Use the sidebar to configure an LLM provider")
    
    with col2:
        st.write("**Enhanced Provider Statistics:**")
        if st.session_state.execution_history:
            df = pd.DataFrame(st.session_state.execution_history)
            
            # Provider usage stats
            if 'llm_provider' in df.columns:
                provider_stats = df['llm_provider'].value_counts()
                provider_success = df.groupby('llm_provider')['success'].mean()
                
                st.write("**Usage & Performance:**")
                for provider, count in provider_stats.items():
                    success_rate = provider_success.get(provider, 0)
                    st.write(f"• **{provider}**: {count} tests ({success_rate:.1%} success)")
            
            # Enhanced feature usage stats
            if 'prompt_refined' in df.columns:
                refined_count = df['prompt_refined'].sum()
                refined_success = df[df['prompt_refined'] == True]['success'].mean() if refined_count > 0 else 0
                st.write(f"**AI-Refined Prompts**: {refined_count} ({refined_success:.1%} success)")
            
            if 'enhanced_execution' in df.columns:
                enhanced_count = df['enhanced_execution'].sum()
                enhanced_success = df[df['enhanced_execution'] == True]['success'].mean() if enhanced_count > 0 else 0
                st.write(f"**Enhanced Executions**: {enhanced_count} ({enhanced_success:.1%} success)")
            
            # Cost tracking (if available)
            total_cost = 0
            if 'cost' in df.columns:
                total_cost = df['cost'].sum()
                st.write(f"**Total API Cost**: ${total_cost:.4f}")
        else:
            st.write("No execution history available for statistics")
    
    st.divider()
    
    # Enhanced benchmark testing section
    st.subheader("🏁 Advanced Benchmark Testing Suite")
    
    if st.session_state.qa_manager is None:
        st.warning("⚠️ Please initialize the system first.")
    else:
        st.write("Run comprehensive benchmark tests with advanced LLM provider comparison and prompt refinement analysis:")
        
        # Enhanced benchmark tasks with categories
        benchmark_categories = {
            "Settings & System Tests": [
                {"name": "Wi-Fi Settings", "task": "Test comprehensive Wi-Fi toggle functionality in Settings", "android_task": "settings_wifi"},
                {"name": "Display Settings", "task": "Navigate to Display settings and test brightness adjustment", "android_task": "settings_wifi"},
                {"name": "Airplane Mode", "task": "Test airplane mode toggle via quick settings panel", "android_task": "settings_wifi"},
                {"name": "Bluetooth Settings", "task": "Verify Bluetooth settings access and status check", "android_task": "settings_wifi"}
            ],
            "App Interaction Tests": [
                {"name": "Calculator", "task": "Open Calculator app and perform comprehensive mathematical operations", "android_task": "calculator_basic"},
                {"name": "Alarm Management", "task": "Test complete alarm creation and management workflow", "android_task": "clock_alarm"},
                {"name": "Contact Management", "task": "Test contact creation, editing, and management features", "android_task": "contacts_add"},
                {"name": "Email Search", "task": "Test email application search and filtering functionality", "android_task": "email_search"}
            ],
            "System Navigation Tests": [
                {"name": "Notification Panel", "task": "Test notification panel access and quick settings interaction", "android_task": "settings_wifi"},
                {"name": "App Switching", "task": "Test recent apps functionality and multitasking navigation", "android_task": "calculator_basic"},
                {"name": "Home Navigation", "task": "Test home screen navigation and app drawer access", "android_task": "settings_wifi"}
            ]
        }
        
        # Benchmark configuration
        col1, col2 = st.columns(2)
        
        with col1:
            selected_category = st.selectbox(
                "Select benchmark category:",
                list(benchmark_categories.keys()),
                index=0
            )
            
            selected_tasks = st.multiselect(
                f"Select tasks from {selected_category}:",
                [t["name"] for t in benchmark_categories[selected_category]],
                default=[t["name"] for t in benchmark_categories[selected_category][:2]]
            )
        
        with col2:
            st.write("**Advanced Benchmark Options:**")
            iterations = st.number_input("Iterations per task:", min_value=1, max_value=5, value=2)
            test_prompt_refinement = st.checkbox("Compare refined vs original prompts", value=True)
            test_execution_modes = st.checkbox("Test multiple execution modes", value=False)
            include_performance_analysis = st.checkbox("Include detailed performance analysis", value=True)
        
        # Execution mode options
        execution_modes = ["Standard"]
        if test_execution_modes:
            execution_modes.extend(["Fast Mode", "Detailed Logging", "Safe Mode"])
        
        # Advanced options
        with st.expander("🔧 Advanced Benchmark Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                benchmark_timeout = st.number_input("Benchmark timeout (minutes):", min_value=5, max_value=30, value=15)
                max_steps_override = st.number_input("Max steps override:", min_value=5, max_value=50, value=25)
                enable_retry_logic = st.checkbox("Enable retry logic", value=True)
            
            with col2:
                save_detailed_logs = st.checkbox("Save detailed execution logs", value=True)
                generate_report = st.checkbox("Generate comprehensive report", value=True)
                compare_providers = st.checkbox("Compare LLM providers (if multiple configured)", value=False)
        
        if st.button("🚀 Run Advanced Benchmark Suite", type="primary"):
            if selected_tasks:
                run_advanced_benchmark_suite(
                    selected_category, selected_tasks, benchmark_categories, 
                    iterations, test_prompt_refinement, execution_modes,
                    include_performance_analysis, save_detailed_logs, 
                    generate_report, benchmark_timeout, max_steps_override
                )
            else:
                st.warning("Please select at least one benchmark task.")
    
    st.divider()
    
    # Enhanced system configuration and diagnostics
    st.subheader("🔧 Enhanced System Configuration & Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Enhanced Configuration:**")
        config_data = {
            "Use Mock LLM": getattr(config, 'USE_MOCK_LLM', True),
            "Primary LLM Provider": getattr(config, 'PRIMARY_LLM_PROVIDER', 'auto'),
            "Max Plan Steps": getattr(config, 'MAX_PLAN_STEPS', 20),
            "Verification Threshold": getattr(config, 'VERIFICATION_THRESHOLD', 0.7),
            "Android Device ID": getattr(config, 'ANDROID_DEVICE_ID', "emulator-5554"),
            "Agent-S Active": status["agent_s"],
            "LLM Provider": st.session_state.llm_interface.__class__.__name__ if st.session_state.llm_interface else "None",
            "Enhanced Features": "Enabled",
            "Prompt Refinement": "Available" if st.session_state.llm_interface else "Unavailable",
            "Cost Tracking": getattr(config, 'ENABLE_COST_TRACKING', False),
            "Cache Enabled": getattr(config, 'CACHE_LLM_RESPONSES', True)
        }
        
        for key, value in config_data.items():
            if isinstance(value, bool):
                st.write(f"**{key}**: {'✅ Yes' if value else '❌ No'}")
            else:
                st.write(f"**{key}**: {value}")
        
        # Enhanced configuration export
        if st.button("📥 Export Enhanced Configuration"):
            enhanced_config = {
                **config_data,
                "export_timestamp": datetime.now().isoformat(),
                "system_version": "Enhanced Multi-Agent QA v2.0",
                "llm_capabilities": {
                    "connected": bool(st.session_state.llm_interface),
                    "provider": st.session_state.llm_interface.__class__.__name__ if st.session_state.llm_interface else None,
                    "refinement_available": bool(st.session_state.llm_interface)
                }
            }
            
            config_json = json.dumps(enhanced_config, indent=2, default=str)
            st.download_button(
                label="Download enhanced_config.json",
                data=config_json,
                file_name=f"enhanced_qa_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.write("**Enhanced System Diagnostics:**")
        
        # System health check
        health_checks = []
        
        # LLM connectivity
        if st.session_state.llm_interface:
            try:
                test_resp = st.session_state.llm_interface.generate_response("test")
                health_checks.append(("LLM Connectivity", "✅ Healthy"))
            except:
                health_checks.append(("LLM Connectivity", "❌ Failed"))
        else:
            health_checks.append(("LLM Connectivity", "⚠️ Not Connected"))
        
        # Agent-S status
        health_checks.append(("Agent-S Architecture", "✅ Active" if status["agent_s"] else "❌ Inactive"))
        
        # QA Manager
        health_checks.append(("QA Manager", "✅ Initialized" if st.session_state.qa_manager else "❌ Not Initialized"))
        
        # Prompt refiner
        health_checks.append(("Prompt Refiner", "✅ Available" if st.session_state.prompt_refiner else "❌ Unavailable"))
        
        # Display health checks
        for check_name, check_status in health_checks:
            st.write(f"**{check_name}**: {check_status}")
        
        # Enhanced metrics
        if st.session_state.qa_manager:
            try:
                metrics = st.session_state.qa_manager.get_system_metrics()
                if "message" not in metrics:
                    test_summary = metrics.get('test_summary', {})
                    st.write(f"**Total Tests**: {test_summary.get('total_tests', 0)}")
                    st.write(f"**Pass Rate**: {test_summary.get('pass_rate', 0):.1%}")
                    
                    integration = metrics.get('system_integration', {})
                    st.write(f"**Android World**: {'Connected' if integration.get('android_world_connected') else 'Mock'}")
                    
                    # Enhanced metrics
                    if st.session_state.execution_history:
                        df = pd.DataFrame(st.session_state.execution_history)
                        
                        if 'prompt_refined' in df.columns:
                            refined_rate = df['prompt_refined'].mean()
                            st.write(f"**Prompt Refinement Rate**: {refined_rate:.1%}")
                        
                        if 'enhanced_execution' in df.columns:
                            enhanced_rate = df['enhanced_execution'].mean()
                            st.write(f"**Enhanced Execution Rate**: {enhanced_rate:.1%}")
                        
                        avg_quality = df.get('actions_summary', pd.Series()).apply(
                            lambda x: x.get('success_rate', 0) if isinstance(x, dict) else 0
                        ).mean()
                        st.write(f"**Avg Action Success Rate**: {avg_quality:.1%}")
                else:
                    st.write("No enhanced metrics available yet")
            except Exception as e:
                st.write(f"Error getting enhanced metrics: {e}")
        else:
            st.write("System not initialized")

def run_advanced_benchmark_suite(category: str, selected_task_names: list, all_categories: dict,
                                iterations: int, test_refinement: bool, execution_modes: list,
                                include_analysis: bool, save_logs: bool, generate_report: bool,
                                timeout_minutes: int, max_steps: int):
    """Run advanced benchmark test suite with comprehensive analysis"""
    
    st.info("🔄 Running advanced benchmark test suite...")
    
    # Get selected tasks
    selected_tasks = [t for t in all_categories[category] if t["name"] in selected_task_names]
    
    # Calculate total tests
    total_tests = len(selected_tasks) * iterations * len(execution_modes)
    if test_refinement:
        total_tests *= 2  # Test both refined and original prompts
    
    # Create enhanced progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    current_test = 0
    start_time = time.time()
    
    # Run comprehensive test matrix
    for mode in execution_modes:
        for task_info in selected_tasks:
            for iteration in range(iterations):
                # Test with original prompt
                current_test += 1
                status_text.text(f"Running test {current_test}/{total_tests}: {task_info['name']} ({mode}) - Original")
                progress_bar.progress(current_test / total_tests)
                
                result = run_single_advanced_benchmark_test(
                    task_info, iteration + 1, mode, False, max_steps, timeout_minutes
                )
                results.append(result)
                
                # Test with refined prompt if enabled
                if test_refinement and st.session_state.llm_interface:
                    current_test += 1
                    status_text.text(f"Running test {current_test}/{total_tests}: {task_info['name']} ({mode}) - Refined")
                    progress_bar.progress(current_test / total_tests)
                    
                    result_refined = run_single_advanced_benchmark_test(
                        task_info, iteration + 1, mode, True, max_steps, timeout_minutes
                    )
                    results.append(result_refined)
    
    total_duration = time.time() - start_time
    status_text.text(f"✅ Advanced benchmark completed in {total_duration:.1f}s!")
    progress_bar.progress(1.0)
    
    # Comprehensive results analysis
    if results:
        df = pd.DataFrame(results)
        
        st.success(f"🎉 Advanced benchmark completed! {len(results)} tests executed in {total_duration:.1f}s.")
        
        # Enhanced metrics dashboard
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            overall_success = df['success'].mean()
            st.metric("Overall Success Rate", f"{overall_success:.1%}")
        
        with col2:
            avg_time = df['total_time'].mean()
            st.metric("Average Duration", f"{avg_time:.1f}s")
        
        with col3:
            if test_refinement and 'prompt_refined' in df.columns:
                refined_tests = df['prompt_refined'].sum()
                st.metric("AI-Refined Tests", f"{refined_tests}/{len(df)}")
            else:
                st.metric("Total Tests", len(df))
        
        with col4:
            unique_modes = df['execution_mode'].nunique()
            st.metric("Execution Modes", unique_modes)
        
        with col5:
            llm_provider = df['llm_provider'].iloc[0] if 'llm_provider' in df.columns else "Mock"
            st.metric("LLM Provider", llm_provider)
        
        # Advanced visualizations
        if include_analysis:
            st.subheader("📊 Comprehensive Performance Analysis")
            
            # Multi-dimensional analysis charts
            col1, col2 = st.columns(2)
            
            with col1:
                if test_refinement and 'prompt_refined' in df.columns:
                    st.subheader("Refined vs Original Prompt Performance")
                    comparison_df = df.groupby(['task_name', 'prompt_refined', 'execution_mode'])['success'].mean().reset_index()
                    comparison_df['prompt_type'] = comparison_df['prompt_refined'].map({True: 'AI-Refined', False: 'Original'})
                    
                    fig = px.bar(
                        comparison_df,
                        x='task_name',
                        y='success',
                        color='prompt_type',
                        facet_col='execution_mode',
                        barmode='group',
                        title="Success Rate by Task, Prompt Type, and Execution Mode",
                        labels={'task_name': 'Task', 'success': 'Success Rate'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Execution Time vs Success Rate")
                fig2 = px.scatter(
                    df,
                    x='total_time',
                    y='success',
                    color='execution_mode',
                    size='total_steps',
                    hover_data=['task_name', 'iteration'],
                    title="Performance Correlation Analysis"
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Execution mode comparison
            if len(execution_modes) > 1:
                st.subheader("Execution Mode Performance Comparison")
                mode_comparison = df.groupby('execution_mode').agg({
                    'success': 'mean',
                    'total_time': 'mean',
                    'total_steps': 'mean'
                }).reset_index()
                
                fig3 = px.bar(
                    mode_comparison,
                    x='execution_mode',
                    y='success',
                    title="Success Rate by Execution Mode",
                    labels={'execution_mode': 'Execution Mode', 'success': 'Success Rate'}
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        # Detailed results table with enhanced columns
        st.subheader("Detailed Advanced Benchmark Results")
        display_df = df.copy()
        
        # Add enhanced visual indicators
        display_df['success_icon'] = display_df['success'].map({True: '✅', False: '❌'})
        display_df['mode_icon'] = display_df['execution_mode'].map({
            'Standard': '⚙️',
            'Fast Mode': '⚡',
            'Detailed Logging': '📝',
            'Safe Mode': '🛡️'
        })
        
        if 'prompt_refined' in display_df.columns:
            display_df['refined_icon'] = display_df['prompt_refined'].map({True: '✨', False: '📝'})
        
        # Select enhanced columns for display
        columns_to_show = ['task_name', 'success_icon', 'mode_icon', 'total_time', 'total_steps', 'iteration']
        if 'refined_icon' in display_df.columns:
            columns_to_show.append('refined_icon')
        if 'llm_provider' in display_df.columns:
            columns_to_show.append('llm_provider')
        
        # Enhanced column configuration
        column_config = {
            "task_name": "Task",
            "success_icon": st.column_config.TextColumn("Result"),
            "mode_icon": st.column_config.TextColumn("Mode"),
            "total_time": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
            "total_steps": "Steps",
            "iteration": "Iteration"
        }
        
        if 'refined_icon' in display_df.columns:
            column_config["refined_icon"] = st.column_config.TextColumn("Prompt")
        if 'llm_provider' in display_df.columns:
            column_config["llm_provider"] = st.column_config.TextColumn("LLM")
        
        st.dataframe(
            display_df[columns_to_show],
            use_container_width=True,
            column_config=column_config
        )
        
        # Generate comprehensive report
        if generate_report:
            st.subheader("📋 Comprehensive Benchmark Report")
            
            report_data = {
                "benchmark_summary": {
                    "total_tests": len(results),
                    "overall_success_rate": overall_success,
                    "average_duration": avg_time,
                    "total_benchmark_time": total_duration,
                    "category_tested": category,
                    "tasks_tested": selected_task_names,
                    "execution_modes": execution_modes,
                    "iterations_per_task": iterations,
                    "prompt_refinement_tested": test_refinement
                },
                "performance_insights": {
                    "best_performing_task": df.loc[df['success'].idxmax(), 'task_name'] if len(df) > 0 else None,
                    "worst_performing_task": df.loc[df['success'].idxmin(), 'task_name'] if len(df) > 0 else None,
                    "fastest_execution": df['total_time'].min(),
                    "slowest_execution": df['total_time'].max(),
                    "most_steps": df['total_steps'].max(),
                    "least_steps": df['total_steps'].min()
                },
                "recommendations": []
            }
            
            # Generate recommendations
            if test_refinement and 'prompt_refined' in df.columns:
                refined_success = df[df['prompt_refined'] == True]['success'].mean()
                original_success = df[df['prompt_refined'] == False]['success'].mean()
                
                if refined_success > original_success + 0.1:
                    report_data["recommendations"].append("✅ AI prompt refinement significantly improves success rates - recommend using for all tests")
                elif refined_success > original_success:
                    report_data["recommendations"].append("💡 AI prompt refinement shows modest improvement - consider using for complex tasks")
                else:
                    report_data["recommendations"].append("⚠️ AI prompt refinement shows limited benefit - original prompts may be sufficient")
            
            if len(execution_modes) > 1:
                best_mode = df.groupby('execution_mode')['success'].mean().idxmax()
                report_data["recommendations"].append(f"🎯 {best_mode} execution mode showed best performance")
            
            if overall_success < 0.7:
                report_data["recommendations"].append("🔧 Overall success rate below 70% - consider system optimization")
            
            # Display report
            st.json(report_data, expanded=True)
            
            # Download report
            report_json = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                label="📥 Download Benchmark Report",
                data=report_json,
                file_name=f"advanced_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Add enhanced benchmark results to main history
        for result in results:
            result['timestamp'] = datetime.now()
            result['benchmark_test'] = True
            result['benchmark_category'] = category
            st.session_state.execution_history.append(result)

def run_single_advanced_benchmark_test(task_info: dict, iteration: int, execution_mode: str,
                                     refine_prompt: bool, max_steps: int, timeout_minutes: int) -> dict:
    """Run a single advanced benchmark test with comprehensive tracking"""
    
    task_description = task_info["task"]
    
    # Enhanced prompt refinement
    if refine_prompt and st.session_state.llm_interface:
        try:
            # Use context-aware refinement based on task type
            focus_area = "settings" if "settings" in task_description.lower() else "apps"
            task_description = st.session_state.prompt_refiner.generate_context_aware_refinement(
                task_description,
                style="detailed",
                focus=focus_area
            )
        except Exception as e:
            st.warning(f"Prompt refinement failed for {task_info['name']}: {e}")
            refine_prompt = False
    
    try:
        # Enhanced test configuration
        test_config = {
            "goal": task_description,
            "android_world_task": task_info["android_task"],
            "max_steps": max_steps,
            "timeout": timeout_minutes * 60,
            "execution_mode": execution_mode,
            "enhanced_features": True,
            "benchmark_test": True
        }
        
        # Run the test with timing
        start_time = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            st.session_state.qa_manager.run_qa_test(test_config)
        )
        end_time = time.time()
        
        # Process result with enhanced tracking
        return {
            "task_name": task_info["name"],
            "task_description": task_description,
            "android_world_task": task_info["android_task"],
            "success": result.final_result == "PASS",
            "total_time": end_time - start_time,
            "total_steps": len(getattr(result, 'actions', [])),
            "iteration": iteration,
            "execution_mode": execution_mode,
            "final_result": result.final_result,
            "agent_s_enhanced": getattr(result, 'agent_s_enhanced', True),
            "test_id": getattr(result, 'test_id', f"benchmark_{task_info['name']}_{iteration}"),
            "prompt_refined": refine_prompt,
            "llm_provider": st.session_state.llm_interface.__class__.__name__ if st.session_state.llm_interface else "Mock",
            "enhanced_execution": True,
            "benchmark_test": True,
            "benchmark_category": "Advanced"
        }
        
    except Exception as e:
        return {
            "task_name": task_info["name"],
            "task_description": task_description,
            "android_world_task": task_info["android_task"],
            "success": False,
            "total_time": 0.0,
            "total_steps": 0,
            "iteration": iteration,
            "execution_mode": execution_mode,
            "final_result": "ERROR",
            "error": str(e),
            "agent_s_enhanced": False,
            "test_id": f"error_benchmark_{task_info['name']}_{iteration}",
            "prompt_refined": refine_prompt,
            "llm_provider": "Error",
            "enhanced_execution": False,
            "benchmark_test": True,
            "benchmark_category": "Advanced"
        }

if __name__ == "__main__":
    main()
