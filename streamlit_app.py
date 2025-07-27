"""
Complete Streamlit Web Interface for Multi-Agent QA System
Integrates Agent-S and android_world with comprehensive UI
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
    """Get cached environment manager - CORRECTED (no typo)"""
    try:
        return EnvironmentManager()
    except Exception as e:
        st.error(f"Failed to initialize EnvironmentManager: {e}")
        return None

def main():
    """Main Streamlit application"""
    init_session_state()
    
    st.title("🤖 Multi-Agent QA System")
    st.subheader("Agent-S + Android World Integration")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration Panel")
        
        # System Status
        if st.session_state.qa_manager:
            try:
                metrics = st.session_state.qa_manager.get_system_metrics()
                if "message" not in metrics:
                    st.success("System Status: ✅ INITIALIZED")
                    integration = metrics.get('system_integration', {})
                    st.write(f"**Agent-S**: {'✅' if integration.get('agent_s_active') else '❌'}")
                    st.write(f"**Android World**: {'✅' if integration.get('android_world_connected') else '❌'}")
                    st.write(f"**LLM Interface**: {integration.get('llm_interface', 'unknown').title()}")
                    
                    test_summary = metrics.get('test_summary', {})
                    st.write(f"**Tests Completed**: {test_summary.get('total_tests', 0)}")
                else:
                    st.warning("System Status: READY (No tests yet)")
            except Exception as e:
                st.warning(f"System Status: ERROR - {e}")
        else:
            st.warning("System Status: NOT INITIALIZED")
        
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
                value=config.MAX_PLAN_STEPS
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
                value=config.VERIFICATION_THRESHOLD,
                step=0.1
            )
            config.VERIFICATION_THRESHOLD = verification_threshold
        
        # Environment Configuration
        with st.expander("Environment Settings"):
            st.subheader("Android World Settings")
            
            android_task = st.selectbox(
                "Android World Task",
                config.ANDROID_WORLD_TASKS,
                index=0
            )
            
            device_id = st.text_input(
                "Android Device ID",
                value=config.ANDROID_DEVICE_ID,
                help="ADB device identifier"
            )
            config.ANDROID_DEVICE_ID = device_id
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing QA system..."):
                try:
                    st.session_state.qa_manager = get_environment_manager()
                    if st.session_state.qa_manager:
                        # Test initialization
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success = loop.run_until_complete(st.session_state.qa_manager.initialize())
                        
                        if success:
                            st.success("✅ System initialized successfully!")
                            st.rerun()
                        else:
                            st.error("❌ System initialization failed")
                    else:
                        st.error("❌ Failed to create EnvironmentManager")
                except Exception as e:
                    st.error(f"❌ Initialization error: {e}")
    
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
    
    # System overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System Status", "READY", delta="Operational")
    with col2:
        mode = "Mock" if config.USE_MOCK_LLM else "Real"
        st.metric("LLM Mode", mode, delta="Active")
    with col3:
        st.metric("Agents", "4 Active", delta="Multi-Agent")
    with col4:
        total_tests = len(st.session_state.execution_history)
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
    android_world_task = st.selectbox(
        "Android World Task Type:",
        config.ANDROID_WORLD_TASKS,
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
    """Execute task with REAL multi-agent system execution"""
    
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
            # Phase 1: Initialization
            status_text.text("Phase 1/4: Planning")
            progress_bar.progress(0.25)
            with phase_details:
                st.write("**Planning**: Analyzing task and creating execution plan")
                if debug:
                    st.code(f"DEBUG: Planning task '{task_description}' with {max_steps} max steps")
            
            # Handle asyncio properly in Streamlit
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
            
            # REAL EXECUTION CALL - Use your corrected QA manager
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
            
            # CORRECTED: Process result into expected format with all required fields
            processed_result = {
                "test_id": result.test_id,  # ✅ FIXED: Ensure test_id is included
                "task_description": task_description,
                "task_name": result.task_name,  # Add this for compatibility
                "android_world_task": android_world_task,
                "success": result.final_result == "PASS",
                "final_result": result.final_result,
                "total_time": result.end_time - result.start_time,
                "total_steps": len(result.actions),
                "bug_detected": result.bug_detected,
                "supervisor_feedback": result.supervisor_feedback or "No feedback available",
                "actions_summary": {
                    "total": len(result.actions),
                    "successful": sum(1 for a in result.actions if a.success),
                    "success_rate": sum(1 for a in result.actions if a.success) / len(result.actions) if result.actions else 0
                },
                "timestamp": datetime.now(),
                # Additional fields for compatibility
                "start_time": result.start_time,
                "end_time": result.end_time,
                "actions": result.actions  # Keep the full actions list
            }
            
        except Exception as e:
            st.error(f"❌ Execution failed: {e}")
            import traceback
            if debug:
                st.code(traceback.format_exc())
            
            # CORRECTED: Create error result with all required fields
            processed_result = {
                "test_id": f"error_test_{int(time.time())}",  # ✅ FIXED: Include test_id
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
                # Additional error fields
                "start_time": time.time(),
                "end_time": time.time(),
                "actions": []
            }
    
    # Add result to history
    st.session_state.execution_history.append(processed_result)
    
    # Display results
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
            st.metric("Final Result", processed_result['final_result'])
        
        # Detailed results
        with st.expander("Detailed Results", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Execution Summary")
                st.write(f"**Test ID**: {processed_result['test_id']}")
                st.write(f"**Task**: {processed_result['task_description']}")
                st.write(f"**Android Task**: {processed_result['android_world_task']}")
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

def dashboard_tab():
    """Dashboard with comprehensive metrics and visualizations"""
    st.header("📊 System Dashboard")
    
    if not st.session_state.execution_history:
        st.info("No execution history available. Execute some tasks first!")
        return
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.execution_history)
    
    # Key performance indicators
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
        total_executions = len(df)
        st.metric("Total Executions", total_executions)
    
    st.divider()
    
    # Performance charts
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
    
    # Advanced analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Task Type Analysis")
        # Categorize tasks
        task_categories = []
        for task in df['task_description']:
            if 'wifi' in task.lower():
                task_categories.append('Wi-Fi')
            elif 'settings' in task.lower():
                task_categories.append('Settings')
            elif 'bluetooth' in task.lower():
                task_categories.append('Bluetooth')
            elif 'calculator' in task.lower():
                task_categories.append('Calculator')
            elif 'alarm' in task.lower():
                task_categories.append('Alarm')
            else:
                task_categories.append('Other')
        
        df['category'] = task_categories
        category_success = df.groupby('category')['success'].mean().reset_index()
        
        fig3 = px.bar(
            category_success,
            x='category',
            y='success',
            title="Success Rate by Task Category",
            labels={'success': 'Success Rate', 'category': 'Task Category'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
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
    
    # Recent executions table
    st.subheader("Recent Executions")
    display_df = df[['task_description', 'success', 'total_time', 'total_steps', 'timestamp']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['success'] = display_df['success'].map({True: '✅ Pass', False: '❌ Fail'})
    display_df = display_df.sort_values('timestamp', ascending=False).head(10)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "task_description": "Task Description",
            "success": st.column_config.TextColumn("Result"),
            "total_time": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
            "total_steps": "Steps",
            "timestamp": "Executed At"
        }
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
        
        # CORRECTED: Safe access to fields with fallbacks
        task_desc = execution.get('task_description', 'Unknown task')
        timestamp = execution.get('timestamp', datetime.now())
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp)
        
        with st.expander(f"{status_icon} {task_desc} - {timestamp_str}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Basic Information:**")
                # CORRECTED: Safe access with fallback values
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
    
    # System configuration
    st.subheader("🔧 System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Configuration:**")
        config_data = {
            "Use Mock LLM": config.USE_MOCK_LLM,
            "Max Plan Steps": config.MAX_PLAN_STEPS,
            "Verification Threshold": config.VERIFICATION_THRESHOLD,
            "Android Device ID": config.ANDROID_DEVICE_ID,
            "Available Tasks": len(config.ANDROID_WORLD_TASKS)
        }
        
        for key, value in config_data.items():
            st.write(f"**{key}**: {value}")
        
        if st.button("📥 Export Configuration"):
            config_json = json.dumps(config_data, indent=2)
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
    
    st.divider()
    
    # System logs and monitoring
    st.subheader("📋 System Logs")
    
    logs_dir = Path("logs")
    
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        json_files = list(logs_dir.glob("*.json"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if log_files:
                st.write("**Log Files:**")
                selected_log = st.selectbox("Select log file:", [f.name for f in log_files])
                
                if st.button("📖 View Log"):
                    log_path = logs_dir / selected_log
                    try:
                        with open(log_path, 'r') as f:
                            log_lines = f.readlines()
                        
                        # Show last 50 lines
                        st.text_area("Log Content (Last 50 lines):", 
                                   value=''.join(log_lines[-50:]), 
                                   height=300)
                    except Exception as e:
                        st.error(f"Failed to read log file: {e}")
            else:
                st.info("No log files found.")
        
        with col2:
            if json_files:
                st.write("**Test Result Files:**")
                selected_json = st.selectbox("Select test result:", [f.name for f in json_files])
                
                if st.button("📊 View Test Data"):
                    json_path = logs_dir / selected_json
                    try:
                        with open(json_path, 'r') as f:
                            test_data = json.load(f)
                        
                        st.json(test_data)
                    except Exception as e:
                        st.error(f"Failed to read test file: {e}")
            else:
                st.info("No test files found.")
    else:
        st.info("Logs directory not found.")
    
    st.divider()
    
    # Android In The Wild Integration (placeholder)
    st.subheader("🌍 Android In The Wild Integration")
    st.info("Android In The Wild evaluation functionality would be implemented here with the integration modules we discussed.")
    
    if st.button("🧪 Simulate Android Wild Evaluation"):
        st.success("This would run the Android In The Wild evaluation with your dataset!")

def run_real_benchmark_suite(selected_task_names: list, all_tasks: list, iterations: int, quick_mode: bool):
    """Run REAL benchmark test suite using your QA system"""
    
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
                
                # Process result
                processed_result = {
                    "task_name": task_info["name"],
                    "task_description": task_info["task"],
                    "android_world_task": task_info["android_task"],
                    "success": result.final_result == "PASS",
                    "total_time": result.end_time - result.start_time,
                    "total_steps": len(result.actions),
                    "iteration": j + 1,
                    "final_result": result.final_result
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
                    "error": str(e)
                })
    
    status_text.text("✅ Benchmark completed!")
    progress_bar.progress(1.0)
    
    # Analyze REAL results
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
            avg_steps = df['total_steps'].mean()
            st.metric("Average Steps", f"{avg_steps:.1f}")
        
        # Benchmark results visualization
        fig = px.box(df, x='task_name', y='total_time', 
                    title="Execution Time Distribution by Task")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Benchmark Results")
        display_df = df[['task_name', 'success', 'total_time', 'total_steps', 'iteration', 'final_result']].copy()
        display_df['success'] = display_df['success'].map({True: '✅ Pass', False: '❌ Fail'})
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "task_name": "Task",
                "success": "Result",  
                "total_time": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
                "total_steps": "Steps",
                "iteration": "Iteration",
                "final_result": "Status"
            }
        )
        
        # Add benchmark results to main history
        for result in results:
            result['timestamp'] = datetime.now()
            st.session_state.execution_history.append(result)

if __name__ == "__main__":
    main()
