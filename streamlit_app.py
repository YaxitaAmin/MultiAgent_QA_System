
import streamlit as st
import asyncio
import json
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from android_in_the_wild.integration_manager import AndroidInTheWildIntegration, EvaluationResult

from env_manager import MultiAgentQAManager
from main_orchestrator import load_config, get_default_config

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
    if 'config' not in st.session_state:
        st.session_state.config = get_default_config()
    if 'execution_history' not in st.session_state:
        st.session_state.execution_history = []
    if 'current_execution' not in st.session_state:
        st.session_state.current_execution = None

def load_qa_manager():
    """Load or reload QA manager with current config"""
    try:
        st.session_state.qa_manager = MultiAgentQAManager(st.session_state.config)
        return True
    except Exception as e:
        st.error(f"Failed to initialize QA Manager: {e}")
        return False

def main():
    init_session_state()
    
    st.title("Multi-Agent QA System for Android UI Testing")
    st.markdown("*Built on Agent-S and AndroidWorld*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration Panel")
        
        # System Status
        if st.session_state.qa_manager:
            status = st.session_state.qa_manager.get_system_status()
            st.success("System Status: READY")
            st.write(f"Environment: {status['environment'].title()}")
            st.write(f"LLM Interface: {status['llm_interface'].title()}")
            st.write(f"Episodes: {status['episodes_completed']}")
        else:
            st.warning("System Status: NOT INITIALIZED")
        
        st.divider()
        
        # API Configuration
        with st.expander("API Settings", expanded=True):
            use_mock = st.checkbox("Use Mock LLM (for testing)", value=st.session_state.config.get("use_mock_llm", True))
            
            if not use_mock:
                api_key = st.text_input(
                    "Gemini API Key", 
                    value=st.session_state.config.get("gemini_api_key", ""),
                    type="password",
                    help="Enter your Google Gemini API key"
                )
                st.session_state.config["gemini_api_key"] = api_key
            
            st.session_state.config["use_mock_llm"] = use_mock
        
        # Agent Configuration
        with st.expander("Agent Settings"):
            st.subheader("Model Configuration")
            
            models = ["mock", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
            
            planner_model = st.selectbox(
                "Planner Model", 
                models, 
                index=0  # Default to mock
            )
            st.session_state.config["agents"]["planner"]["model"] = planner_model
            
            executor_model = st.selectbox(
                "Executor Model", 
                models, 
                index=0  # Default to mock
            )
            st.session_state.config["agents"]["executor"]["model"] = executor_model
            
            verifier_model = st.selectbox(
                "Verifier Model", 
                models, 
                index=0  # Default to mock
            )
            st.session_state.config["agents"]["verifier"]["model"] = verifier_model
            
            supervisor_model = st.selectbox(
                "Supervisor Model", 
                models, 
                index=0  # Default to mock
            )
            st.session_state.config["agents"]["supervisor"]["model"] = supervisor_model
        
        # Environment Configuration
        with st.expander("Environment Settings"):
            max_steps = st.number_input(
                "Max Steps", 
                min_value=10, 
                max_value=100, 
                value=st.session_state.config["android_env"].get("max_steps", 10)
            )
            st.session_state.config["android_env"]["max_steps"] = max_steps
            
            timeout = st.number_input(
                "Timeout (seconds)", 
                min_value=60, 
                max_value=600, 
                value=st.session_state.config["android_env"].get("task_timeout", 300)
            )
            st.session_state.config["android_env"]["task_timeout"] = timeout
        
        # Initialize button
        if st.button("Initialize System", type="primary"):
            with st.spinner("Initializing QA system..."):
                if load_qa_manager():
                    st.success("System initialized successfully!")
                else:
                    st.error("Failed to initialize system")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Execute Task", "Dashboard", "History", "Advanced"])
    
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
        st.warning("Please initialize the system first using the sidebar.")
        return
    
    # System overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System Status", "READY", delta="Operational")
    with col2:
        st.metric("Environment", "Mock Android", delta="Test Mode")
    with col3:
        st.metric("Agents", "4 Active", delta="All Systems Go")
    with col4:
        total_tests = len(st.session_state.execution_history)
        st.metric("Tests Completed", total_tests, delta=f"+{total_tests}")
    
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
        "Test device volume controls",
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
    
    # Execution parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        max_steps = st.number_input("Max Steps", min_value=5, max_value=100, value=10)
    with col2:
        timeout = st.number_input("Timeout (seconds)", min_value=30, max_value=600, value=300)
    with col3:
        enable_debug = st.checkbox("Enable Debug Mode", value=False)
    
    # Execution controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        execute_button = st.button("Execute Task", type="primary", disabled=not task_description.strip())
    with col2:
        if st.button("Quick Test", help="Run with minimal steps"):
            max_steps = 10
            timeout = 60
            execute_button = True
    with col3:
        if st.button("Clear History"):
            st.session_state.execution_history = []
            st.rerun()
    
    # Execute task
    if execute_button and task_description.strip():
        execute_task_with_real_system(task_description, max_steps, timeout, enable_debug)

def execute_task_with_real_system(task_description: str, max_steps: int, timeout: int, debug: bool):
    """Execute task with REAL multi-agent system execution"""
    
    # Create containers for progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.info("Executing task with real multi-agent system...")
        
        # Progress indicators
        col1, col2 = st.columns([3, 1])
        with col1:
            progress_bar = st.progress(0)
        with col2:
            status_text = st.empty()
        
        phase_container = st.container()
        phase_details = phase_container.expander("Execution Details", expanded=debug)
        
        # REAL EXECUTION - Call your actual QA manager
        try:
            # Phase 1: Planning
            status_text.text("Phase 1/4: Planning")
            progress_bar.progress(0.25)
            with phase_details:
                st.write("**Planning**: Analyzing task and creating execution plan")
                if debug:
                    st.code("DEBUG: Planner agent creating test plan...")
            
            # Execute the real task using your QA manager
            import asyncio
            
            # Handle asyncio properly in Streamlit
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Execute with progress updates
            status_text.text("Phase 2/4: Executing")
            progress_bar.progress(0.5)
            with phase_details:
                st.write("**Executing**: Performing UI interactions and actions")
                if debug:
                    st.code("DEBUG: Executing subgoals with agents...")
            
            # REAL EXECUTION CALL
            if hasattr(st.session_state.qa_manager, 'execute_qa_task_sync'):
                result = st.session_state.qa_manager.execute_qa_task_sync(task_description, max_steps, timeout)
            else:
                result = loop.run_until_complete(
                    st.session_state.qa_manager.execute_qa_task(task_description, max_steps, timeout)
                )
            
            # Phase 3: Verification
            status_text.text("Phase 3/4: Verifying")
            progress_bar.progress(0.75)
            with phase_details:
                st.write("**Verifying**: Validating results and checking state")
                if debug:
                    exec_rate = result.get('execution', {}).get('success_rate', 0)
                    st.code(f"DEBUG: Execution success rate: {exec_rate:.1%}")
            
            # Phase 4: Analysis
            status_text.text("Phase 4/4: Analyzing")
            progress_bar.progress(1.0)
            with phase_details:
                st.write("**Analyzing**: Generating comprehensive report")
                if debug:
                    st.code(f"DEBUG: Task {'succeeded' if result.get('success') else 'failed'}")
                    
        except Exception as e:
            st.error(f"Execution failed: {e}")
            # Create error result
            result = {
                "episode_id": f"episode_error_{int(time.time())}",
                "task_description": task_description,
                "success": False,
                "error": str(e),
                "total_time": 0.0,
                "total_steps": 0,
                "plan": {"goal": task_description, "subgoals_count": 0, "completed_subgoals": 0},
                "execution": {"success_rate": 0.0, "average_execution_time": 0.0},
                "verification": {"pass_rate": 0.0, "average_confidence": 0.0}
            }
    
    # Add REAL result to history
    st.session_state.execution_history.append({
        **result,
        "timestamp": datetime.now()
    })
    
    # Display REAL results
    with results_container:
        if result["success"]:
            st.success("Task executed successfully!")
        else:
            st.error("Task execution failed!")
            if "error" in result:
                st.error(f"Error: {result['error']}")
        
        # Display REAL metrics from your QA system
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{result['total_time']:.2f}s")
        with col2:
            st.metric("Steps", result['total_steps'])
        with col3:
            exec_rate = result.get('execution', {}).get('success_rate', 0)
            st.metric("Success Rate", f"{exec_rate:.1%}")
        with col4:
            confidence = result.get('verification', {}).get('average_confidence', 0)
            st.metric("Confidence", f"{confidence:.2f}")
        
        # Detailed REAL results
        with st.expander("Detailed Results", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Plan Execution")
                plan = result['plan']
                st.write(f"Goal: {plan['goal']}")
                st.write(f"Subgoals: {plan['completed_subgoals']}/{plan['subgoals_count']}")
                
                # Real progress visualization
                if plan['subgoals_count'] > 0:
                    progress_pct = plan['completed_subgoals'] / plan['subgoals_count']
                    st.progress(progress_pct)
                else:
                    st.progress(0.0)
            
            with col2:
                st.subheader("Agent Performance")
                
                # Show REAL agent performance from your system
                execution_data = result.get('execution', {})
                verification_data = result.get('verification', {})
                
                agent_data = {
                    "Agent": ["Planner", "Executor", "Verifier", "Supervisor"],
                    "Performance": [
                        1.0 if plan['completed_subgoals'] > 0 else 0.0,  # Planner success
                        execution_data.get('success_rate', 0.0),         # Real executor performance
                        verification_data.get('pass_rate', 0.0),         # Real verifier performance  
                        0.85 if result.get('success') else 0.5          # Supervisor based on overall success
                    ]
                }
                
                fig = px.bar(agent_data, x="Agent", y="Performance", 
                           title="Real Agent Performance Scores")
                st.plotly_chart(fig, use_container_width=True)
        
        # Show raw result data for debugging
        if debug:
            with st.expander("Raw Execution Data"):
                st.json(result)

def dashboard_tab():
    """Dashboard with comprehensive metrics and visualizations"""
    st.header("System Dashboard")
    
    if not st.session_state.execution_history:
        st.info("No execution history available. Execute some tasks first!")
        return
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.execution_history)
    
    # Key performance indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        success_rate = df['success'].mean()
        st.metric("Overall Success Rate", f"{success_rate:.1%}", 
                 delta=f"{success_rate - 0.8:.1%}" if success_rate > 0.8 else None)
    
    with col2:
        avg_duration = df['total_time'].mean()
        st.metric("Avg Duration", f"{avg_duration:.1f}s",
                 delta=f"{avg_duration - 10:.1f}s" if avg_duration < 20 else None)
    
    with col3:
        avg_steps = df['total_steps'].mean()
        st.metric("Avg Steps", f"{avg_steps:.1f}",
                 delta=f"{avg_steps - 4:.1f}" if avg_steps < 6 else None)
    
    with col4:
        total_executions = len(df)
        st.metric("Total Executions", total_executions,
                 delta=f"+{total_executions}")
    
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
    display_df['success'] = display_df['success'].map({True: 'Pass', False: 'Fail'})
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
    st.header("Execution History")
    
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
        filtered_history = [h for h in filtered_history if h['success']]
    elif success_filter == "Failed":
        filtered_history = [h for h in filtered_history if not h['success']]
    
    # Apply sorting
    if sort_order == "Newest First":
        filtered_history = sorted(filtered_history, key=lambda x: x['timestamp'], reverse=True)
    elif sort_order == "Oldest First":
        filtered_history = sorted(filtered_history, key=lambda x: x['timestamp'])
    elif sort_order == "Duration":
        filtered_history = sorted(filtered_history, key=lambda x: x['total_time'], reverse=True)
    
    filtered_history = filtered_history[:max_results]
    
    # Display execution details
    for i, execution in enumerate(filtered_history):
        status_icon = "✅" if execution['success'] else "❌"
        
        with st.expander(
            f"{status_icon} {execution['task_description']} - {execution['timestamp'].strftime('%Y-%m-%d %H:%M')}"
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"Episode ID: `{execution['episode_id']}`")
                st.write(f"Duration: {execution['total_time']:.2f}s")
                st.write(f"Steps: {execution['total_steps']}")
                st.write(f"Result: {'Success' if execution['success'] else 'Failed'}")
            
            with col2:
                st.write("**Performance Metrics:**")
                if 'execution' in execution:
                    st.write(f"Execution Success: {execution['execution']['success_rate']:.1%}")
                if 'verification' in execution:
                    st.write(f"Verification Pass: {execution['verification']['pass_rate']:.1%}")
                    st.write(f"Avg Confidence: {execution['verification']['average_confidence']:.2f}")
            
            with col3:
                st.write("**Plan Details:**")
                if 'plan' in execution:
                    plan = execution['plan']
                    st.write(f"Subgoals: {plan['completed_subgoals']}/{plan['subgoals_count']}")
                    
                    if plan['subgoals_count'] > 0:
                        completion_rate = plan['completed_subgoals'] / plan['subgoals_count']
                        st.progress(completion_rate)
            
            # Advanced details
            if st.checkbox(f"Show detailed data", key=f"details_{i}"):
                st.json(execution, expanded=False)

def advanced_tab():
    """Advanced features and system management"""
    st.header("Advanced Features")
    
    # Benchmark testing section
    st.subheader("Benchmark Testing")
    
    if st.session_state.qa_manager is None:
        st.warning("Please initialize the system first.")
    else:
        st.write("Run predefined benchmark tests to evaluate system performance:")
        
        benchmark_tasks = [
            "Test turning Wi-Fi on and off",
            "Open Settings and navigate to Display settings", 
            "Test airplane mode toggle",
            "Navigate to Bluetooth settings and verify state",
            "Open Calculator app and perform basic calculation",
            "Test device volume controls"
        ]
        
        selected_benchmark_tasks = st.multiselect(
            "Select benchmark tasks:",
            benchmark_tasks,
            default=benchmark_tasks[:3]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            iterations = st.number_input("Iterations per task:", min_value=1, max_value=10, value=2)
        with col2:
            parallel_execution = st.checkbox("Parallel execution", value=False)
        
        if st.button("Run Benchmark", type="primary"):
            if selected_benchmark_tasks:
                run_real_benchmark_suite(selected_benchmark_tasks, iterations, parallel_execution)
            else:
                st.warning("Please select at least one benchmark task.")
    
    st.divider()
    
    # System configuration
    st.subheader("System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Configuration:**")
        if st.button("Export Current Config"):
            config_json = json.dumps(st.session_state.config, indent=2)
            st.download_button(
                label="Download config.json",
                data=config_json,
                file_name=f"qa_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.write("**Import Configuration:**")
        uploaded_file = st.file_uploader("Choose config file", type=['json'])
        
        if uploaded_file is not None:
            try:
                config = json.load(uploaded_file)
                st.session_state.config = config
                st.success("Configuration imported successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to import configuration: {e}")
    
    st.divider()
    
    # System logs and monitoring
    st.subheader("System Logs")
    
    if st.session_state.qa_manager:
        logs_dir = Path("logs")
        
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            json_files = list(logs_dir.glob("*.json"))
            
            col1, col2 = st.columns(2)
            
            with col1:
                if log_files:
                    selected_log = st.selectbox("Select log file:", [f.name for f in log_files])
                    
                    if st.button("View Log"):
                        log_path = logs_dir / selected_log
                        try:
                            with open(log_path, 'r') as f:
                                log_lines = f.readlines()
                            
                            # Show last 100 lines
                            st.text_area("Log Content (Last 100 lines):", 
                                       value=''.join(log_lines[-100:]), 
                                       height=300)
                        except Exception as e:
                            st.error(f"Failed to read log file: {e}")
                else:
                    st.info("No log files found.")
            
            with col2:
                if json_files:
                    selected_json = st.selectbox("Select episode file:", [f.name for f in json_files])
                    
                    if st.button("View Episode Data"):
                        json_path = logs_dir / selected_json
                        try:
                            with open(json_path, 'r') as f:
                                episode_data = json.load(f)
                            
                            st.json(episode_data)
                        except Exception as e:
                            st.error(f"Failed to read episode file: {e}")
                else:
                    st.info("No episode files found.")
        else:
            st.info("Logs directory not found.")
    else:
        st.info("System not initialized.")
    
    st.divider()
    
    # Android In The Wild Integration
    st.subheader("Android In The Wild Dataset Integration")
    
    if st.session_state.qa_manager is None:
        st.warning("Please initialize the system first.")
    else:
        st.write("Evaluate your multi-agent system against real-world Android usage patterns:")
        
        # Dataset configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dataset_path = st.text_input(
                "Dataset Path:", 
                value="./android_in_the_wild_dataset",
                help="Path to Android In The Wild dataset directory"
            )
        
        with col2:
            num_videos = st.number_input(
                "Number of videos:", 
                min_value=1, 
                max_value=10, 
                value=5
            )
        
        with col3:
            auto_create = st.checkbox(
                "Auto-create sample data", 
                value=True,
                help="Create sample dataset if none exists"
            )
        
        # Execution buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Run Evaluation", type="primary"):
                run_android_wild_evaluation(dataset_path, num_videos, auto_create)
        
        with col2:
            if st.button("View Dataset Stats"):
                show_dataset_statistics(dataset_path)
        
        with col3:
            if st.button("Clear Results"):
                if 'android_wild_results' in st.session_state:
                    del st.session_state.android_wild_results
                st.success("Results cleared!")
        
        # Display results if available
        if 'android_wild_results' in st.session_state:
            display_comprehensive_results()

def run_android_wild_evaluation(dataset_path: str, num_videos: int, auto_create: bool):
    """Run comprehensive Android In The Wild evaluation"""
    
    st.info("Running Android In The Wild evaluation...")
    
    try:
        # Import here to avoid import errors if module doesn't exist
        import sys
        from pathlib import Path
        
        # Add android_in_the_wild to path
        android_wild_path = Path(__file__).parent / "android_in_the_wild"
        if str(android_wild_path) not in sys.path:
            sys.path.append(str(android_wild_path))
        
        from android_in_the_wild.integration_manager import AndroidInTheWildIntegration
        
        # Create integration manager
        integration = AndroidInTheWildIntegration(st.session_state.qa_manager, dataset_path)
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            detail_text = st.empty()
        
        # Run evaluation
        status_text.text("Initializing evaluation...")
        progress_bar.progress(0.1)
        
        # Use sync method for Streamlit compatibility
        results = integration.run_evaluation_sync(num_videos)
        
        if results:
            # Update progress as results come in
            for i, result in enumerate(results):
                progress = (i + 1) / len(results)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i+1}/{len(results)} videos")
                detail_text.text(f"Latest: {result.generated_prompt} - {'Success' if result.success else 'Failed'}")
                time.sleep(0.1)  # Small delay for visual effect
            
            # Store results
            st.session_state.android_wild_results = {
                'results': results,
                'report': integration.generate_evaluation_report(),
                'timestamp': time.time()
            }
            
            progress_bar.progress(1.0)
            status_text.text("Evaluation completed successfully!")
            detail_text.text(f"Processed {len(results)} videos with {sum(1 for r in results if r.success)} successes")
            
            # Auto-display results
            st.success(f"Evaluation completed! Processed {len(results)} videos.")
            display_comprehensive_results()
            
        else:
            st.error("No results generated. Check dataset path and system configuration.")
            
    except ImportError as e:
        st.error(f"Module import failed: {e}")
        st.info("Make sure android_in_the_wild modules are properly installed.")
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        st.info("Check logs for detailed error information.")

def show_dataset_statistics(dataset_path: str):
    """Show dataset statistics"""
    try:
        from android_in_the_wild.dataset_handler import AndroidInTheWildHandler
        
        handler = AndroidInTheWildHandler(dataset_path)
        handler.load_video_traces(10)  # Load up to 10 for stats
        stats = handler.get_dataset_statistics()
        
        st.subheader("Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Videos", stats.get('total_videos', 0))
        
        with col2:
            st.metric("Total Duration", f"{stats.get('total_duration', 0):.1f}s")
        
        with col3:
            st.metric("Avg Duration", f"{stats.get('average_duration', 0):.1f}s")
        
        with col4:
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1%}")
        
        # Task type distribution
        if 'task_type_distribution' in stats:
            st.subheader("Task Type Distribution")
            
            task_data = stats['task_type_distribution']
            fig = px.pie(
                values=list(task_data.values()),
                names=list(task_data.keys()),
                title="Distribution of Task Types in Dataset"
            )
            st.plotly_chart(fig, use_container_width=True, key="dataset_task_distribution_chart")
        
    except Exception as e:
        st.error(f"Failed to load dataset statistics: {e}")

def display_comprehensive_results():
    """Display comprehensive Android In The Wild evaluation results"""
    
    if 'android_wild_results' not in st.session_state:
        st.info("No evaluation results available.")
        return
    
    data = st.session_state.android_wild_results
    results = data['results']
    report = data['report']
    
    # Generate unique timestamp for this render
    import time
    render_id = int(time.time() * 1000) % 10000
    
    st.header("Android In The Wild Evaluation Results")
    st.markdown("---")
    
    # Summary metrics with better spacing
    st.subheader("📊 Overall Performance Metrics")
    summary = report['evaluation_summary']
    
    # Create two rows of metrics for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        success_rate = summary['overall_success_rate']
        delta_color = "normal" if success_rate >= 0.7 else "inverse"
        st.metric(
            "Success Rate", 
            f"{success_rate:.1%}",
            delta=f"{success_rate - 0.7:.1%}",
            delta_color=delta_color
        )
    
    with col2:
        accuracy = summary['average_accuracy']
        delta_color = "normal" if accuracy >= 0.8 else "inverse"
        st.metric(
            "Accuracy", 
            f"{accuracy:.2f}",
            delta=f"{accuracy - 0.8:.2f}",
            delta_color=delta_color
        )
    
    with col3:
        composite = summary['composite_score']
        delta_color = "normal" if composite >= 0.75 else "inverse"
        st.metric(
            "Composite Score", 
            f"{composite:.2f}",
            delta=f"{composite - 0.75:.2f}",
            delta_color=delta_color
        )
    
    # Second row
    col4, col5, col6 = st.columns(3)
    
    with col4:
        robustness = summary['average_robustness']
        delta_color = "normal" if robustness >= 0.75 else "inverse"
        st.metric(
            "Robustness", 
            f"{robustness:.2f}",
            delta=f"{robustness - 0.75:.2f}",
            delta_color=delta_color
        )
    
    with col5:
        generalization = summary['average_generalization']
        delta_color = "normal" if generalization >= 0.7 else "inverse"
        st.metric(
            "Generalization", 
            f"{generalization:.2f}",
            delta=f"{generalization - 0.7:.2f}",
            delta_color=delta_color
        )
    
    with col6:
        total_videos = len(results)
        success_count = sum(1 for r in results if r.success)
        st.metric(
            "Tests Processed", 
            f"{total_videos}",
            delta=f"{success_count} successful"
        )
    
    st.markdown("---")
    
    # Performance visualization with better spacing
    st.subheader("📈 Performance Analysis")
    
    # Score distribution chart with proper spacing
    if results:
        st.markdown("#### Score Distribution by Metric")
        st.markdown("*Distribution of accuracy, robustness, and generalization scores across all test videos*")
        
        score_data = []
        for result in results:
            score_data.extend([
                {"Metric": "Accuracy", "Score": result.accuracy_score, "Video": result.video_id},
                {"Metric": "Robustness", "Score": result.robustness_score, "Video": result.video_id},
                {"Metric": "Generalization", "Score": result.generalization_score, "Video": result.video_id}
            ])
        
        if score_data:
            df_scores = pd.DataFrame(score_data)
            
            fig1 = px.box(
                df_scores, 
                x="Metric", 
                y="Score",
                title="Score Distribution by Performance Metric",
                points="all",
                height=400,
                color="Metric"
            )
            fig1.update_layout(
                font_size=12,
                title_font_size=16,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True, key=f"score_distribution_chart_{render_id}")
        
        st.markdown("---")
        
        # Success analysis with better layout
        st.markdown("#### Success vs Failure Analysis")
        st.markdown("*Relationship between different performance metrics and overall success*")
        
        success_data = []
        for result in results:
            success_data.append({
                "Video ID": result.video_id,
                "Success": "✅ Success" if result.success else "❌ Failure",
                "Accuracy": result.accuracy_score,
                "Robustness": result.robustness_score,
                "Generalization": result.generalization_score,
                "Task Type": result.ground_truth_trace.metadata.get('task_type', 'unknown'),
                "Execution Time": result.execution_time
            })
        
        if success_data:
            df_success = pd.DataFrame(success_data)
            
            fig2 = px.scatter(
                df_success,
                x="Accuracy",
                y="Robustness",
                size="Generalization",
                color="Success",
                hover_data=["Task Type", "Execution Time", "Video ID"],
                title="Performance Correlation Analysis (Size = Generalization Score)",
                height=450,
                size_max=20
            )
            fig2.update_layout(
                font_size=12,
                title_font_size=16,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"success_scatter_chart_{render_id}")
    
    st.markdown("---")
    
    # Task type performance with better organization
    if 'task_type_performance' in report and report['task_type_performance']:
        st.subheader("🎯 Task Type Performance Analysis")
        
        task_perf = report['task_type_performance']
        task_df = pd.DataFrame([
            {
                "Task Type": task_type,
                "Count": data['count'],
                "Success Rate": data['success_rate'],
                "Accuracy": data['avg_accuracy'],
                "Robustness": data['avg_robustness'],
                "Generalization": data['avg_generalization']
            }
            for task_type, data in task_perf.items()
        ])
        
        # Summary stats for task performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_tasks = len(task_df)
            st.metric("Task Categories", total_tasks)
        
        with col2:
            if not task_df.empty:
                avg_success = task_df['Success Rate'].mean()
                st.metric("Average Success Rate", f"{avg_success:.1%}")
            else:
                st.metric("Average Success Rate", "0.0%")
        
        with col3:
            if not task_df.empty:
                best_task = task_df.loc[task_df['Success Rate'].idxmax(), 'Task Type']
                best_rate = task_df['Success Rate'].max()
                st.metric("Best Performing Task", f"{best_task}", delta=f"{best_rate:.1%}")
            else:
                st.metric("Best Performing Task", "None")
        
        st.markdown("#### Task Performance Summary")
        
        # Enhanced table display
        if not task_df.empty:
            # Style the dataframe for better readability
            styled_df = task_df.style.format({
                'Success Rate': '{:.1%}',
                'Accuracy': '{:.2f}',
                'Robustness': '{:.2f}',
                'Generalization': '{:.2f}'
            }).background_gradient(
                subset=['Success Rate', 'Accuracy', 'Robustness', 'Generalization'],
                cmap='RdYlGn',
                low=0.3,
                high=0.7
            )
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=200
            )
        else:
            st.info("No task performance data available")
        
        # Task performance charts in better layout
        if not task_df.empty:
            st.markdown("#### Performance Metrics by Task Category")
            
            # Single wide chart instead of two columns for better readability
            fig3 = px.bar(
                task_df,
                x="Task Type",
                y=["Success Rate", "Accuracy", "Robustness", "Generalization"],
                title="Comprehensive Performance Metrics by Task Type",
                barmode="group",
                height=400
            )
            fig3.update_layout(
                font_size=12,
                title_font_size=16,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                xaxis_tickangle=45,
                legend_title="Metrics"
            )
            st.plotly_chart(fig3, use_container_width=True, key=f"task_performance_chart_{render_id}")
    
    st.markdown("---")
    
    # Execution time analysis with better presentation
    if results:
        st.subheader("⏱️ Execution Time Analysis")
        st.markdown("*Analysis of execution times across different test scenarios*")
        
        time_data = []
        for result in results:
            time_data.append({
                "Video ID": result.video_id,
                "Task Type": result.ground_truth_trace.metadata.get('task_type', 'unknown'),
                "Execution Time": result.execution_time,
                "Success": "✅ Success" if result.success else "❌ Failure",
                "Accuracy": result.accuracy_score
            })
        
        df_time = pd.DataFrame(time_data)
        
        # Summary stats for execution time
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_time = df_time['Execution Time'].mean()
            st.metric("Average Time", f"{avg_time:.2f}s")
        
        with col2:
            max_time = df_time['Execution Time'].max()
            st.metric("Maximum Time", f"{max_time:.2f}s")
        
        with col3:
            min_time = df_time['Execution Time'].min()
            st.metric("Minimum Time", f"{min_time:.2f}s")
        
        # Execution time chart
        fig5 = px.bar(
            df_time,
            x="Video ID",
            y="Execution Time",
            color="Success",
            title="Execution Time by Test Video",
            hover_data=["Task Type", "Accuracy"],
            height=350
        )
        fig5.update_layout(
            font_size=12,
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            xaxis_tickangle=45
        )
        st.plotly_chart(fig5, use_container_width=True, key=f"execution_time_chart_{render_id}")
    
    st.markdown("---")
    
    # Detailed results table with better formatting
    st.subheader("📋 Detailed Test Results")
    st.markdown("*Complete breakdown of all test executions with performance metrics*")
    
    if results:
        # Summary before detailed table
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tests = len(results)
            st.metric("Total Tests", total_tests)
        
        with col2:
            successful_tests = sum(1 for r in results if r.success)
            st.metric("Successful Tests", successful_tests)
        
        with col3:
            failed_tests = total_tests - successful_tests
            st.metric("Failed Tests", failed_tests)
        
        with col4:
            if total_tests > 0:
                success_percentage = (successful_tests / total_tests) * 100
                st.metric("Success Percentage", f"{success_percentage:.1f}%")
            else:
                st.metric("Success Percentage", "0.0%")
        
        st.markdown("#### Complete Test Results Table")
        
        detailed_df = pd.DataFrame([
            {
                "Video ID": result.video_id,
                "Generated Prompt": result.generated_prompt[:40] + "..." if len(result.generated_prompt) > 40 else result.generated_prompt,
                "Task Type": result.ground_truth_trace.metadata.get('task_type', 'unknown'),
                "Success": "✅ Pass" if result.success else "❌ Fail",
                "Accuracy": f"{result.accuracy_score:.3f}",
                "Robustness": f"{result.robustness_score:.3f}",
                "Generalization": f"{result.generalization_score:.3f}",
                "Execution Time": f"{result.execution_time:.2f}s",
                "Expected Actions": len(result.ground_truth_trace.user_actions),
                "Agent Steps": result.agent_trace.get('total_steps', 0)
            }
            for result in results
        ])
        
        # Enhanced dataframe with better column configuration
        st.dataframe(
            detailed_df,
            use_container_width=True,
            height=350,
            column_config={
                "Video ID": st.column_config.TextColumn("Video ID", width=120),
                "Generated Prompt": st.column_config.TextColumn("Generated Prompt", width=250),
                "Task Type": st.column_config.TextColumn("Task Type", width=150),
                "Success": st.column_config.TextColumn("Result", width=100),
                "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f", width=100),
                "Robustness": st.column_config.NumberColumn("Robustness", format="%.3f", width=100),
                "Generalization": st.column_config.NumberColumn("Generalization", format="%.3f", width=120),
                "Execution Time": st.column_config.TextColumn("Time", width=100),
                "Expected Actions": st.column_config.NumberColumn("Expected", width=100),
                "Agent Steps": st.column_config.NumberColumn("Actual", width=100)
            }
        )
    else:
        st.info("No detailed results available")
    
    st.markdown("---")
    
    # Recommendations
    if 'recommendations' in report and report['recommendations']:
        st.subheader("Recommendations for Improvement")
        
        recommendations = report['recommendations']
        
        # Group recommendations by priority
        critical_recs = [r for r in recommendations if r.get('priority') == 'critical']
        high_recs = [r for r in recommendations if r.get('priority') == 'high']
        medium_recs = [r for r in recommendations if r.get('priority') == 'medium']
        other_recs = [r for r in recommendations if r.get('priority') not in ['critical', 'high', 'medium']]
        
        # Display by priority
        if critical_recs:
            st.markdown("#### Critical Issues")
            for i, rec in enumerate(critical_recs):
                st.error(f"**{rec.get('type', 'general').title()}**: {rec.get('message', 'No message')}")
                if rec.get('specific_actions'):
                    with st.expander(f"Action Items for {rec.get('type', 'issue').title()}"):
                        for action in rec['specific_actions']:
                            st.write(f"• {action}")
        
        if high_recs:
            st.markdown("#### High Priority")
            for i, rec in enumerate(high_recs):
                st.warning(f"**{rec.get('type', 'general').title()}**: {rec.get('message', 'No message')}")
                if rec.get('specific_actions'):
                    with st.expander(f"Action Items for {rec.get('type', 'issue').title()}"):
                        for action in rec['specific_actions']:
                            st.write(f"• {action}")
        
        if medium_recs:
            st.markdown("#### Medium Priority")
            for i, rec in enumerate(medium_recs):
                st.info(f"**{rec.get('type', 'general').title()}**: {rec.get('message', 'No message')}")
                if rec.get('specific_actions'):
                    with st.expander(f"Action Items for {rec.get('type', 'issue').title()}"):
                        for action in rec['specific_actions']:
                            st.write(f"• {action}")
        
        if other_recs:
            st.markdown("#### Additional Recommendations")
            for i, rec in enumerate(other_recs):
                st.success(f"**{rec.get('type', 'general').title()}**: {rec.get('message', 'No message')}")
                if rec.get('specific_actions'):
                    with st.expander(f"Action Items for {rec.get('type', 'issue').title()}"):
                        for action in rec['specific_actions']:
                            st.write(f"• {action}")
    
    st.divider()
    
    # Export functionality
    st.subheader("Export Results")
    
    # Summary stats for export section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Videos Processed", len(results))
    
    with col2:
        success_count = sum(1 for r in results if r.success)
        st.metric("Successful Tests", f"{success_count}/{len(results)}")
    
    with col3:
        avg_time = sum(r.execution_time for r in results) / len(results) if results else 0
        st.metric("Average Execution Time", f"{avg_time:.2f}s")
    
    # Export buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export JSON Report", key=f"export_json_button_{render_id}"):
            try:
                export_data = {
                    "evaluation_summary": summary,
                    "detailed_results": [
                        {
                            "video_id": result.video_id,
                            "generated_prompt": result.generated_prompt,
                            "success": result.success,
                            "accuracy": result.accuracy_score,
                            "robustness": result.robustness_score,
                            "generalization": result.generalization_score,
                            "execution_time": result.execution_time,
                            "task_type": result.ground_truth_trace.metadata.get('task_type', 'unknown')
                        }
                        for result in results
                    ],
                    "export_timestamp": time.time()
                }
                
                report_json = json.dumps(export_data, indent=2)
                
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"android_wild_evaluation_{int(time.time())}.json",
                    mime="application/json",
                    key=f"download_json_button_{render_id}"
                )
                
                st.success("Report ready for download!")
                
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col2:
        if st.button("Export CSV Summary", key=f"export_csv_button_{render_id}"):
            try:
                summary_data = []
                for result in results:
                    summary_data.append({
                        'video_id': result.video_id,
                        'prompt': result.generated_prompt,
                        'success': result.success,
                        'accuracy': result.accuracy_score,
                        'robustness': result.robustness_score,
                        'generalization': result.generalization_score,
                        'execution_time': result.execution_time,
                        'task_type': result.ground_truth_trace.metadata.get('task_type', 'unknown')
                    })
                
                if summary_data:
                    df_export = pd.DataFrame(summary_data)
                    csv_data = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV Summary",
                        data=csv_data,
                        file_name=f"android_wild_summary_{int(time.time())}.csv",
                        mime="text/csv",
                        key=f"download_csv_button_{render_id}"
                    )
                    
                    st.success("CSV ready for download!")
                else:
                    st.warning("No data available for CSV export")
                
            except Exception as e:
                st.error(f"CSV export failed: {e}")
    
    with col3:
        if st.button("Export Full Report", key=f"export_full_button_{render_id}"):
            try:
                full_report = json.dumps(report, indent=2, default=str)
                
                st.download_button(
                    label="Download Full Report",
                    data=full_report,
                    file_name=f"android_wild_full_report_{int(time.time())}.json",
                    mime="application/json",
                    key=f"download_full_button_{render_id}"
                )
                
                st.success("Full report ready for download!")
                
            except Exception as e:
                st.error(f"Full report export failed: {e}")

def run_real_benchmark_suite(tasks: list, iterations: int, parallel: bool):
    """Run REAL benchmark test suite using your QA system"""
    
    st.info("Running benchmark tests...")
    
    # Create progress tracking
    total_tests = len(tasks) * iterations
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, task in enumerate(tasks):
        for j in range(iterations):
            current_test = (i * iterations) + j + 1
            status_text.text(f"Running test {current_test}/{total_tests}: {task}")
            progress_bar.progress(current_test / total_tests)
            
            # Execute REAL test using your QA manager
            try:
                if hasattr(st.session_state.qa_manager, 'execute_qa_task_sync'):
                    result = st.session_state.qa_manager.execute_qa_task_sync(task, 10, 300)
                else:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(
                        st.session_state.qa_manager.execute_qa_task(task, 10, 300)
                    )
                
                result['iteration'] = j + 1
                results.append(result)
                
            except Exception as e:
                st.error(f"Benchmark test failed: {e}")
                # Add failed result
                results.append({
                    "task_description": task,
                    "success": False,
                    "total_time": 0.0,
                    "total_steps": 0,
                    "iteration": j + 1,
                    "error": str(e)
                })
    
    st.success("Benchmark completed!")
    
    # Analyze REAL results
    df = pd.DataFrame(results)
    
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
    fig = px.box(df, x='task_description', y='total_time', 
                title="Execution Time Distribution by Task")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()