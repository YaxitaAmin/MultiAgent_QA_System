"""
Complete Streamlit Web Interface for Multi-Agent QA System - CORRECTED
Integrates Agent-S and android_world with comprehensive UI and proper Agent-S detection
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

# Import your corrected modules
from env_manager import EnvironmentManager
from config.default_config import config
from core.logger import QATestResult

# Page configuration
st.set_page_config(
    page_title="Multi-Agent QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def get_environment_manager():
    """Get cached environment manager"""
    try:
        return EnvironmentManager()
    except Exception as e:
        st.error(f"Failed to initialize EnvironmentManager: {e}")
        return None

def check_agent_s_status():
    """✅ CORRECTED: Check if Agent-S architecture is active by examining agent classes"""
    try:
        # Check if we have a QA manager instance
        if st.session_state.qa_manager is None:
            return {"active": False, "reason": "QA Manager not initialized"}
        
        # Import base class for checking
        from agents.base_agents import QAAgentS2
        
        # Check if agents are QAAgentS2 instances (meaning they extend Agent-S)
        agents = {
            "planner": st.session_state.qa_manager.planner_agent,
            "executor": st.session_state.qa_manager.executor_agent,
            "verifier": st.session_state.qa_manager.verifier_agent,
            "supervisor": st.session_state.qa_manager.supervisor_agent
        }
        
        agent_status = {}
        for name, agent in agents.items():
            agent_status[name] = isinstance(agent, QAAgentS2)
        
        # Agent-S is active if all agents extend QAAgentS2
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
    """✅ CORRECTED: Get comprehensive system status with proper Agent-S detection"""
    try:
        # Check Agent-S status
        agent_s_status = check_agent_s_status()
        
        # Get system metrics if QA manager is available
        if st.session_state.qa_manager:
            try:
                metrics = st.session_state.qa_manager.get_system_metrics()
                test_summary = metrics.get('test_summary', {})
                system_integration = metrics.get('system_integration', {})
                
                return {
                    "initialized": True,
                    "agent_s": agent_s_status["active"],
                    "agent_s_details": agent_s_status,
                    "android_world": system_integration.get('android_world_connected', True),  # Mock is always available
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
                    "android_world": True,  # Mock environment
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
    
    st.title("🤖 Multi-Agent QA System")
    st.subheader("Agent-S + Android World Integration")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration Panel")
        
        # ✅ CORRECTED: System Status with proper Agent-S detection
        status = get_system_status()
        
        if status["initialized"]:
            st.success("**System Status:** ✅ INITIALIZED")
            
            # ✅ CORRECTED: Proper Agent-S status display
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
            
            # Show Agent-S details in expander
            if status["agent_s"] and "details" in agent_s_details:
                with st.expander("🤖 Agent-S Details"):
                    for agent_name, is_active in agent_s_details["details"].items():
                        status_icon = "✅" if is_active else "❌"
                        st.write(f"{status_icon} {agent_name.title()}Agent: {'QAAgentS2' if is_active else 'BaseAgent'}")
            
            # Show coordination status if available
            if status.get("coordination_active"):
                st.success("**Coordination:** ✅ Active")
            else:
                st.warning("**Coordination:** ⚠️ Limited")
        else:
            st.warning("**System Status:** ⚠️ NOT INITIALIZED")
            if "error" in status:
                st.error(f"Error: {status['error']}")
        
        st.divider()
        
        # API Configuration
        with st.expander("API Settings", expanded=True):
            use_mock = st.checkbox(
                "Use Mock LLM (for testing)", 
                value=config.USE_MOCK_LLM
            )
            
            if not use_mock:
                api_key = st.text_input(
                    "Gemini API Key", 
                    value=config.GOOGLE_API_KEY or "",
                    type="password",
                    help="Enter your Google Gemini API key"
                )
                if api_key:
                    config.GOOGLE_API_KEY = api_key
            
            # Update config
            config.USE_MOCK_LLM = use_mock
        
        # Agent Configuration
        with st.expander("Agent Settings"):
            st.subheader("Agent Configuration")
            
            max_steps = st.number_input(
                "Max Plan Steps", 
                min_value=5, 
                max_value=50, 
                value=getattr(config, 'MAX_PLAN_STEPS', 20)
            )
            config.MAX_PLAN_STEPS = max_steps
            
            timeout = st.number_input(
                "Timeout (seconds)", 
                min_value=30, 
                max_value=600, 
                value=120
            )
            
            verification_threshold = st.slider(
                "Verification Threshold", 
                min_value=0.5, 
                max_value=1.0, 
                value=getattr(config, 'VERIFICATION_THRESHOLD', 0.7),
                step=0.1
            )
            config.VERIFICATION_THRESHOLD = verification_threshold
        
        # Environment Configuration
        with st.expander("Environment Settings"):
            st.subheader("Android World Settings")
            
            android_tasks = getattr(config, 'ANDROID_WORLD_TASKS', [
                "settings_wifi", "clock_alarm", "calculator_basic", 
                "contacts_add", "email_search"
            ])
            
            android_task = st.selectbox(
                "Android World Task",
                android_tasks,
                index=0
            )
            
            device_id = st.text_input(
                "Android Device ID",
                value=getattr(config, 'ANDROID_DEVICE_ID', "emulator-5554"),
                help="ADB device identifier"
            )
            config.ANDROID_DEVICE_ID = device_id
        
        # ✅ CORRECTED: Initialize button with proper async handling
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing QA system..."):
                try:
                    st.session_state.qa_manager = get_environment_manager()
                    if st.session_state.qa_manager:
                        # Test initialization with proper async handling
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        success = loop.run_until_complete(st.session_state.qa_manager.initialize())
                        
                        if success:
                            st.success("✅ System initialized successfully!")
                            # Force rerun to update status
                            time.sleep(0.5)  # Brief pause for initialization
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
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Execute Task", "📊 Dashboard", "📋 History", "⚙️ Advanced"])
    
    with tab1:
        execute_task_tab()
    
    with tab2:
        dashboard_tab()
    
    with tab3:
        history_tab()
    
    with tab4:
        advanced_tab()

def execute_task_tab():
    """Task execution interface"""
    st.header("Execute QA Task")
    
    if st.session_state.qa_manager is None:
        st.warning("⚠️ Please initialize the system first using the sidebar.")
        return
    
    # ✅ CORRECTED: System overview with proper Agent-S status
    status = get_system_status()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "READY" if status["initialized"] else "NOT READY", 
                 delta="Operational" if status["initialized"] else "Initialize First")
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
    
    # Task input section
    st.subheader("Define QA Task")
    
    # Predefined tasks
    predefined_tasks = [
        "Test turning Wi-Fi on and off",
        "Open Settings and navigate to Display settings",
        "Test airplane mode toggle",
        "Navigate to Bluetooth settings and verify state",
        "Open Calculator app and perform basic calculation",
        "Test alarm creation and management",
        "Navigate to Storage settings and check usage",
        "Custom task..."
    ]
    
    selected_task = st.selectbox("Select a predefined task or choose custom:", predefined_tasks)
    
    if selected_task == "Custom task...":
        task_description = st.text_area(
            "Enter custom task description:",
            placeholder="Describe the QA task you want to execute...",
            height=100
        )
    else:
        task_description = st.text_area(
            "Task description:",
            value=selected_task,
            height=100
        )
    
    # Android World task selection
    android_tasks = getattr(config, 'ANDROID_WORLD_TASKS', [
        "settings_wifi", "clock_alarm", "calculator_basic", 
        "contacts_add", "email_search"
    ])
    
    android_world_task = st.selectbox(
        "Android World Task Type:",
        android_tasks,
        index=0,
        help="Select the underlying android_world task type"
    )
    
    # Execution parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        max_steps = st.number_input("Max Steps", min_value=5, max_value=50, value=20)
    with col2:
        timeout = st.number_input("Timeout (seconds)", min_value=30, max_value=600, value=120)
    with col3:
        enable_debug = st.checkbox("Enable Debug Mode", value=False)
    
    # Execution controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        execute_button = st.button("🚀 Execute Task", type="primary", disabled=not task_description.strip())
    with col2:
        if st.button("⚡ Quick Test", help="Run with minimal steps"):
            max_steps = 10
            timeout = 60
            execute_button = True
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.execution_history = []
            st.success("History cleared!")
            st.rerun()
    
    # Execute task
    if execute_button and task_description.strip():
        execute_task_with_real_system(task_description, android_world_task, max_steps, timeout, enable_debug)

def execute_task_with_real_system(task_description: str, android_world_task: str, 
                                 max_steps: int, timeout: int, debug: bool):
    """✅ CORRECTED: Execute task with REAL multi-agent system execution"""
    
    # Create containers for progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.info("🔄 Executing task with real multi-agent system...")
        
        # Progress indicators
        col1, col2 = st.columns([3, 1])
        with col1:
            progress_bar = st.progress(0)
        with col2:
            status_text = st.empty()
        
        phase_container = st.container()
        if debug:
            phase_details = phase_container.expander("Execution Details", expanded=True)
        else:
            phase_details = phase_container.expander("Execution Details", expanded=False)
        
        try:
            # Phase 1: Planning
            status_text.text("Phase 1/4: Planning")
            progress_bar.progress(0.25)
            with phase_details:
                st.write("**Planning**: Analyzing task and creating execution plan")
                if debug:
                    st.code(f"DEBUG: Planning task '{task_description}' with {max_steps} max steps")
            
            # ✅ CORRECTED: Handle asyncio properly in Streamlit
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Phase 2: Execution
            status_text.text("Phase 2/4: Executing")
            progress_bar.progress(0.5)
            with phase_details:
                st.write("**Executing**: Performing UI interactions and actions")
                if debug:
                    st.code(f"DEBUG: Using android_world task '{android_world_task}'")
            
            # ✅ CORRECTED: REAL EXECUTION CALL
            test_config = {
                "goal": task_description,
                "android_world_task": android_world_task,
                "max_steps": max_steps,
                "timeout": timeout
            }
            
            result = loop.run_until_complete(
                st.session_state.qa_manager.run_qa_test(test_config)
            )
            
            # Phase 3: Verification
            status_text.text("Phase 3/4: Verifying")
            progress_bar.progress(0.75)
            with phase_details:
                st.write("**Verifying**: Validating results and checking state")
                if debug:
                    st.code(f"DEBUG: Final result: {result.final_result}")
            
            # Phase 4: Analysis
            status_text.text("Phase 4/4: Analyzing")
            progress_bar.progress(1.0)
            with phase_details:
                st.write("**Analyzing**: Generating comprehensive report")
                if debug:
                    st.code(f"DEBUG: {len(result.actions)} actions executed")
            
            # ✅ CORRECTED: Process result with all required fields
            processed_result = {
                "test_id": getattr(result, 'test_id', f"test_{int(time.time())}"),
                "task_description": task_description,
                "task_name": getattr(result, 'task_name', task_description),
                "android_world_task": android_world_task,
                "success": result.final_result == "PASS",
                "final_result": result.final_result,
                "total_time": getattr(result, 'end_time', time.time()) - getattr(result, 'start_time', time.time()),
                "total_steps": len(getattr(result, 'actions', [])),
                "bug_detected": getattr(result, 'bug_detected', False),
                "supervisor_feedback": getattr(result, 'supervisor_feedback', "No feedback available") or "No feedback available",
                "actions_summary": {
                    "total": len(getattr(result, 'actions', [])),
                    "successful": sum(1 for a in getattr(result, 'actions', []) if getattr(a, 'success', False)),
                    "success_rate": sum(1 for a in getattr(result, 'actions', []) if getattr(a, 'success', False)) / len(getattr(result, 'actions', [])) if getattr(result, 'actions', []) else 0
                },
                "timestamp": datetime.now(),
                "start_time": getattr(result, 'start_time', time.time()),
                "end_time": getattr(result, 'end_time', time.time()),
                "actions": getattr(result, 'actions', []),
                "agent_s_enhanced": getattr(result, 'agent_s_enhanced', True)  # ✅ Track Agent-S usage
            }
            
        except Exception as e:
            st.error(f"❌ Execution failed: {e}")
            import traceback
            if debug:
                st.code(traceback.format_exc())
            
            # ✅ CORRECTED: Create error result with all required fields
            processed_result = {
                "test_id": f"error_test_{int(time.time())}",
                "task_description": task_description,
                "task_name": f"Failed: {task_description}",
                "android_world_task": android_world_task,
                "success": False,
                "final_result": "ERROR",
                "total_time": 0.0,
                "total_steps": 0,
                "bug_detected": False,
                "supervisor_feedback": f"Execution error: {str(e)}",
                "actions_summary": {"total": 0, "successful": 0, "success_rate": 0},
                "error": str(e),
                "timestamp": datetime.now(),
                "start_time": time.time(),
                "end_time": time.time(),
                "actions": [],
                "agent_s_enhanced": False
            }
    
    # Add result to history
    st.session_state.execution_history.append(processed_result)
    
    # ✅ CORRECTED: Display results with Agent-S information
    with results_container:
        if processed_result["success"]:
            st.success("✅ Task executed successfully!")
        else:
            st.error("❌ Task execution failed!")
            if "error" in processed_result:
                st.error(f"Error: {processed_result['error']}")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{processed_result['total_time']:.2f}s")
        with col2:
            st.metric("Steps", processed_result['total_steps'])
        with col3:
            success_rate = processed_result['actions_summary']['success_rate']
            st.metric("Action Success Rate", f"{success_rate:.1%}")
        with col4:
            agent_s_badge = "🤖" if processed_result.get("agent_s_enhanced", False) else "⚙️"
            st.metric("Final Result", f"{agent_s_badge} {processed_result['final_result']}")
        
        # Detailed results
        with st.expander("Detailed Results", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Execution Summary")
                st.write(f"**Test ID**: {processed_result['test_id']}")
                st.write(f"**Task**: {processed_result['task_description']}")
                st.write(f"**Android Task**: {processed_result['android_world_task']}")
                st.write(f"**Agent-S Enhanced**: {'Yes' if processed_result.get('agent_s_enhanced', False) else 'No'}")
                st.write(f"**Bug Detected**: {'Yes' if processed_result['bug_detected'] else 'No'}")
                
                if processed_result['supervisor_feedback']:
                    st.write("**Supervisor Feedback**:")
                    st.info(processed_result['supervisor_feedback'])
            
            with col2:
                st.subheader("Action Breakdown")
                actions = processed_result['actions_summary']
                if actions['total'] > 0:
                    success_pct = actions['successful'] / actions['total'] * 100
                    st.progress(success_pct / 100)
                    st.write(f"**Successful Actions**: {actions['successful']}/{actions['total']} ({success_pct:.1f}%)")
                else:
                    st.write("No actions recorded")
        
        # Show raw result data for debugging
        if debug:
            with st.expander("Raw Execution Data"):
                st.json(processed_result)

# ✅ Keep all other functions (dashboard_tab, history_tab, advanced_tab) the same as in your original code
# Just add the Agent-S status checks where needed

def dashboard_tab():
    """Dashboard with comprehensive metrics and visualizations"""
    st.header("📊 System Dashboard")
    
    if not st.session_state.execution_history:
        st.info("No execution history available. Execute some tasks first!")
        return
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.execution_history)
    
    # ✅ CORRECTED: Key performance indicators with Agent-S tracking
    col1, col2, col3, col4 = st.columns(4)
    
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
        # ✅ CORRECTED: Track Agent-S enhanced tests
        agent_s_tests = df.get('agent_s_enhanced', pd.Series([False] * len(df))).sum()
        st.metric("Agent-S Tests", f"{agent_s_tests}/{len(df)}")
    
    st.divider()
    
    # Performance charts (keep existing chart code)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Success Rate Trend")
        df_indexed = df.reset_index()
        fig1 = px.line(
            df_indexed, 
            x='index', 
            y='success',
            title="Success Rate Over Time",
            labels={'index': 'Execution #', 'success': 'Success (1=Pass, 0=Fail)'}
        )
        fig1.update_traces(mode='lines+markers')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Execution Duration Distribution")
        fig2 = px.histogram(
            df, 
            x='total_time',
            title="Duration Distribution",
            labels={'total_time': 'Duration (seconds)', 'count': 'Frequency'},
            nbins=20
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # ✅ CORRECTED: Advanced analytics with Agent-S breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Agent-S Usage Analysis")
        if 'agent_s_enhanced' in df.columns:
            agent_s_breakdown = df['agent_s_enhanced'].value_counts()
            fig3 = px.pie(
                values=agent_s_breakdown.values,
                names=['Agent-S Enhanced' if x else 'Standard Mode' for x in agent_s_breakdown.index],
                title="Test Type Distribution"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Agent-S tracking not available in historical data")
    
    with col2:
        st.subheader("Performance Correlation")
        fig4 = px.scatter(
            df,
            x='total_time',
            y='total_steps',
            color='success',
            title="Duration vs Steps",
            labels={'total_time': 'Duration (s)', 'total_steps': 'Steps'}
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Recent executions table (keep existing code but add Agent-S column)
    st.subheader("Recent Executions")
    display_df = df[['task_description', 'success', 'total_time', 'total_steps', 'timestamp']].copy()
    
    # ✅ CORRECTED: Add Agent-S column if available
    if 'agent_s_enhanced' in df.columns:
        display_df['agent_s'] = df['agent_s_enhanced'].map({True: '🤖', False: '⚙️'})
    
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['success'] = display_df['success'].map({True: '✅ Pass', False: '❌ Fail'})
    display_df = display_df.sort_values('timestamp', ascending=False).head(10)
    
    column_config = {
        "task_description": "Task Description",
        "success": st.column_config.TextColumn("Result"),
        "total_time": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
        "total_steps": "Steps",
        "timestamp": "Executed At"
    }
    
    if 'agent_s' in display_df.columns:
        column_config["agent_s"] = st.column_config.TextColumn("Mode")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config=column_config
    )

def history_tab():
    """Detailed execution history and analysis"""
    st.header("📋 Execution History")
    
    if not st.session_state.execution_history:
        st.info("No execution history available.")
        return
    
    # Filters and controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        success_filter = st.selectbox("Filter by result:", ["All", "Successful", "Failed"])
    
    with col2:
        max_results = st.number_input("Max results to show:", min_value=5, max_value=100, value=20)
    
    with col3:
        sort_order = st.selectbox("Sort by:", ["Newest First", "Oldest First", "Duration", "Success Rate"])
    
    # Apply filters
    filtered_history = st.session_state.execution_history.copy()
    
    if success_filter == "Successful":
        filtered_history = [h for h in filtered_history if h.get('success', False)]
    elif success_filter == "Failed":
        filtered_history = [h for h in filtered_history if not h.get('success', False)]
    
    # Apply sorting
    if sort_order == "Newest First":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
    elif sort_order == "Oldest First":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('timestamp', datetime.now()))
    elif sort_order == "Duration":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('total_time', 0), reverse=True)
    
    filtered_history = filtered_history[:max_results]
    
    # Display execution details
    for i, execution in enumerate(filtered_history):
        status_icon = "✅" if execution.get('success', False) else "❌"
        agent_s_icon = "🤖" if execution.get('agent_s_enhanced', False) else "⚙️"
        
        # ✅ CORRECTED: Safe access to fields with fallbacks
        task_desc = execution.get('task_description', 'Unknown task')
        timestamp = execution.get('timestamp', datetime.now())
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp)
        
        with st.expander(f"{status_icon} {agent_s_icon} {task_desc} - {timestamp_str}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Basic Information:**")
                test_id = execution.get('test_id', execution.get('task_name', f'unknown_{i}'))
                st.write(f"Test ID: `{test_id}`")
                st.write(f"Duration: {execution.get('total_time', 0):.2f}s")
                st.write(f"Steps: {execution.get('total_steps', 0)}")
                st.write(f"Result: {execution.get('final_result', 'Unknown')}")
            
            with col2:
                st.write("**Performance Metrics:**")
                actions = execution.get('actions_summary', {})
                success_rate = actions.get('success_rate', 0)
                st.write(f"Action Success: {success_rate:.1%}")
                st.write(f"Bug Detected: {'Yes' if execution.get('bug_detected', False) else 'No'}")
                st.write(f"Android Task: {execution.get('android_world_task', 'Unknown')}")
                st.write(f"Agent-S Enhanced: {'Yes' if execution.get('agent_s_enhanced', False) else 'No'}")
            
            with col3:
                st.write("**Feedback:**")
                feedback = execution.get('supervisor_feedback', '')
                if feedback:
                    if len(feedback) > 100:
                        st.write(f"{feedback[:100]}...")
                    else:
                        st.write(feedback)
                else:
                    st.write("No feedback available")
            
            # Advanced details
            if st.checkbox(f"Show detailed data", key=f"details_{i}"):
                st.json(execution, expanded=False)

def advanced_tab():
    """Advanced features and system management"""
    st.header("⚙️ Advanced Features")
    
    # ✅ CORRECTED: Show Agent-S system information
    st.subheader("🤖 Agent-S System Information")
    
    status = get_system_status()
    if status["agent_s"]:
        st.success("Agent-S Architecture: ✅ Active")
        
        agent_s_details = status["agent_s_details"]
        if "details" in agent_s_details:
            st.write("**Agent Status:**")
            for agent_name, is_active in agent_s_details["details"].items():
                status_badge = "✅ QAAgentS2" if is_active else "❌ BaseAgent"
                st.write(f"- {agent_name.title()}Agent: {status_badge}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Agent-S Agents", f"{agent_s_details.get('agent_s_agents', 0)}/4")
        with col2:
            coordination_status = status.get("coordination_active", False)
            st.metric("Coordination", "✅ Active" if coordination_status else "⚠️ Limited")
    else:
        st.warning("Agent-S Architecture: ❌ Not Active")
        if "error" in status["agent_s_details"]:
            st.error(f"Error: {status['agent_s_details']['error']}")
    
    st.divider()
    
    # Benchmark testing section
    st.subheader("🏁 Benchmark Testing")
    
    if st.session_state.qa_manager is None:
        st.warning("⚠️ Please initialize the system first.")
    else:
        st.write("Run predefined benchmark tests to evaluate system performance:")
        
        benchmark_tasks = [
            {"name": "Wi-Fi Settings", "task": "Test turning Wi-Fi on and off", "android_task": "settings_wifi"},
            {"name": "Display Settings", "task": "Open Settings and navigate to Display settings", "android_task": "settings_wifi"},
            {"name": "Airplane Mode", "task": "Test airplane mode toggle", "android_task": "settings_wifi"},
            {"name": "Bluetooth Settings", "task": "Navigate to Bluetooth settings and verify state", "android_task": "settings_wifi"},
            {"name": "Calculator", "task": "Open Calculator app and perform basic calculation", "android_task": "calculator_basic"},
            {"name": "Alarm Management", "task": "Test alarm creation and management", "android_task": "clock_alarm"}
        ]
        
        selected_benchmark_tasks = st.multiselect(
            "Select benchmark tasks:",
            [t["name"] for t in benchmark_tasks],
            default=[t["name"] for t in benchmark_tasks[:3]]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            iterations = st.number_input("Iterations per task:", min_value=1, max_value=5, value=2)
        with col2:
            quick_mode = st.checkbox("Quick mode (fewer steps)", value=True)
        
        if st.button("🚀 Run Benchmark", type="primary"):
            if selected_benchmark_tasks:
                run_real_benchmark_suite(selected_benchmark_tasks, benchmark_tasks, iterations, quick_mode)
            else:
                st.warning("Please select at least one benchmark task.")
    
    st.divider()
    
    # System configuration (keep existing code)
    st.subheader("🔧 System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Configuration:**")
        config_data = {
            "Use Mock LLM": getattr(config, 'USE_MOCK_LLM', True),
            "Max Plan Steps": getattr(config, 'MAX_PLAN_STEPS', 20),
            "Verification Threshold": getattr(config, 'VERIFICATION_THRESHOLD', 0.7),
            "Android Device ID": getattr(config, 'ANDROID_DEVICE_ID', "emulator-5554"),
            "Agent-S Active": status["agent_s"]
        }
        
        for key, value in config_data.items():
            st.write(f"**{key}**: {value}")
        
        if st.button("📥 Export Configuration"):
            config_json = json.dumps(config_data, indent=2, default=str)
            st.download_button(
                label="Download config.json",
                data=config_json,
                file_name=f"qa_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.write("**System Metrics:**")
        if st.session_state.qa_manager:
            try:
                metrics = st.session_state.qa_manager.get_system_metrics()
                if "message" not in metrics:
                    test_summary = metrics.get('test_summary', {})
                    st.write(f"**Total Tests**: {test_summary.get('total_tests', 0)}")
                    st.write(f"**Pass Rate**: {test_summary.get('pass_rate', 0):.1%}")
                    
                    integration = metrics.get('system_integration', {})
                    st.write(f"**Agent-S Active**: {'Yes' if integration.get('agent_s_active') else 'No'}")
                    st.write(f"**Android World**: {'Connected' if integration.get('android_world_connected') else 'Mock'}")
                else:
                    st.write("No metrics available yet")
            except Exception as e:
                st.write(f"Error getting metrics: {e}")
        else:
            st.write("System not initialized")
    
    # Keep rest of advanced_tab code the same...
    # (System logs, Android In The Wild Integration sections)

def run_real_benchmark_suite(selected_task_names: list, all_tasks: list, iterations: int, quick_mode: bool):
    """✅ CORRECTED: Run REAL benchmark test suite with Agent-S tracking"""
    
    st.info("🔄 Running benchmark tests...")
    
    # Filter selected tasks
    selected_tasks = [t for t in all_tasks if t["name"] in selected_task_names]
    
    # Create progress tracking
    total_tests = len(selected_tasks) * iterations
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, task_info in enumerate(selected_tasks):
        for j in range(iterations):
            current_test = (i * iterations) + j + 1
            status_text.text(f"Running test {current_test}/{total_tests}: {task_info['name']}")
            progress_bar.progress(current_test / total_tests)
            
            # Execute REAL test using your QA manager
            try:
                test_config = {
                    "goal": task_info["task"],
                    "android_world_task": task_info["android_task"],
                    "max_steps": 10 if quick_mode else 20,
                    "timeout": 60 if quick_mode else 120
                }
                
                # Run the actual test
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    st.session_state.qa_manager.run_qa_test(test_config)
                )
                
                # ✅ CORRECTED: Process result with Agent-S tracking
                processed_result = {
                    "task_name": task_info["name"],
                    "task_description": task_info["task"],
                    "android_world_task": task_info["android_task"],
                    "success": result.final_result == "PASS",
                    "total_time": getattr(result, 'end_time', time.time()) - getattr(result, 'start_time', time.time()),
                    "total_steps": len(getattr(result, 'actions', [])),
                    "iteration": j + 1,
                    "final_result": result.final_result,
                    "agent_s_enhanced": getattr(result, 'agent_s_enhanced', True),
                    "test_id": getattr(result, 'test_id', f"benchmark_{current_test}")
                }
                results.append(processed_result)
                
            except Exception as e:
                st.error(f"Benchmark test failed: {e}")
                # Add failed result
                results.append({
                    "task_name": task_info["name"],
                    "task_description": task_info["task"],
                    "android_world_task": task_info["android_task"],
                    "success": False,
                    "total_time": 0.0,
                    "total_steps": 0,
                    "iteration": j + 1,
                    "final_result": "ERROR",
                    "error": str(e),
                    "agent_s_enhanced": False,
                    "test_id": f"error_benchmark_{current_test}"
                })
    
    status_text.text("✅ Benchmark completed!")
    progress_bar.progress(1.0)
    
    # ✅ CORRECTED: Analyze REAL results with Agent-S metrics
    if results:
        df = pd.DataFrame(results)
        
        st.success(f"🎉 Benchmark completed! {len(results)} tests executed.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overall_success = df['success'].mean()
            st.metric("Overall Success Rate", f"{overall_success:.1%}")
        
        with col2:
            avg_time = df['total_time'].mean()
            st.metric("Average Duration", f"{avg_time:.1f}s")
        
        with col3:
            agent_s_tests = df['agent_s_enhanced'].sum() if 'agent_s_enhanced' in df else 0
            st.metric("Agent-S Enhanced", f"{agent_s_tests}/{len(df)}")
        
        # Benchmark results visualization
        fig = px.box(df, x='task_name', y='total_time', 
                    title="Execution Time Distribution by Task")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Benchmark Results")
        display_df = df[['task_name', 'success', 'total_time', 'total_steps', 'iteration', 'final_result']].copy()
        
        if 'agent_s_enhanced' in df:
            display_df['agent_s'] = df['agent_s_enhanced'].map({True: '🤖', False: '⚙️'})
        
        display_df['success'] = display_df['success'].map({True: '✅ Pass', False: '❌ Fail'})
        
        column_config = {
            "task_name": "Task",
            "success": "Result",  
            "total_time": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
            "total_steps": "Steps",
            "iteration": "Iteration",
            "final_result": "Status"
        }
        
        if 'agent_s' in display_df:
            column_config["agent_s"] = st.column_config.TextColumn("Mode")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config=column_config
        )
        
        # Add benchmark results to main history
        for result in results:
            result['timestamp'] = datetime.now()
            st.session_state.execution_history.append(result)

if __name__ == "__main__":
    main()
