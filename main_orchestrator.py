# main_orchestrator.py - CORRECTED VERSION integrating with all previous fixes
import asyncio
import json
import yaml
from pathlib import Path
import argparse
import time
from typing import Dict, Any, List
from loguru import logger

from env_manager import EnvironmentManager
from config.default_config import config

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with fallback to default config"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    
    try:
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Merge with default config
        default_config = get_default_config()
        merged_config = {**default_config, **yaml_config}
        
        logger.info(f"Loaded configuration from {config_path}")
        return merged_config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration that matches our system setup"""
    return {
        "agents": {
            "planner": {
                "model": config.GEMINI_MODEL,
                "temperature": 0.1,
                "max_tokens": 1000
            },
            "executor": {
                "model": config.GEMINI_MODEL,
                "temperature": 0.0,
                "max_tokens": 500
            },
            "verifier": {
                "model": config.GEMINI_MODEL,
                "temperature": 0.1,
                "max_tokens": 600
            },
            "supervisor": {
                "model": config.GEMINI_MODEL,
                "temperature": 0.2,
                "max_tokens": 1000
            }
        },
        "android_env": {
            "task_name": "settings_wifi",
            "task_timeout": 300,
            "max_steps": config.MAX_PLAN_STEPS,
            "screenshot_dir": str(config.SCREENSHOTS_DIR),
            "device_id": config.ANDROID_DEVICE_ID
        },
        "logging": {
            "level": config.LOG_LEVEL,
            "log_dir": str(config.LOGS_DIR),
            "enable_screenshots": True
        },
        "use_mock_llm": config.USE_MOCK_LLM,
        "gemini_api_key": config.GOOGLE_API_KEY,
        "android_world_tasks": config.ANDROID_WORLD_TASKS
    }

async def run_single_test(env_manager: EnvironmentManager, task_description: str, 
                         android_world_task: str, max_steps: int, timeout: int) -> Dict[str, Any]:
    """Run a single QA test"""
    print(f"\n🎯 Executing QA Task: {task_description}")
    print(f"   Android World Task: {android_world_task}")
    print(f"   Max Steps: {max_steps}")
    print(f"   Timeout: {timeout}s")
    
    test_config = {
        "goal": task_description,
        "android_world_task": android_world_task,
        "max_steps": max_steps,
        "timeout": timeout
    }
    
    start_time = time.time()
    result = await env_manager.run_qa_test(test_config)
    duration = time.time() - start_time
    
    return {
        "success": result.final_result == "PASS",
        "result": result.final_result,
        "test_id": result.test_id,
        "task_name": result.task_name,
        "duration": duration,
        "total_steps": len(result.actions),
        "bug_detected": result.bug_detected,
        "supervisor_feedback": result.supervisor_feedback,
        "actions_summary": {
            "total": len(result.actions),
            "successful": sum(1 for a in result.actions if a.success),
            "success_rate": sum(1 for a in result.actions if a.success) / len(result.actions) if result.actions else 0
        }
    }

async def run_benchmark(env_manager: EnvironmentManager, tasks: List[Dict[str, str]], 
                       iterations: int = 1) -> Dict[str, Any]:
    """Run benchmark tests"""
    print(f"\n🚀 Running benchmark with {len(tasks)} task types, {iterations} iterations each")
    
    all_results = []
    task_results = {}
    
    for task_config in tasks:
        task_name = task_config["name"]
        task_description = task_config["description"]
        android_world_task = task_config["android_world_task"]
        
        print(f"\n📋 Testing: {task_name}")
        task_results[task_name] = []
        
        for iteration in range(iterations):
            print(f"   Iteration {iteration + 1}/{iterations}")
            
            try:
                result = await run_single_test(
                    env_manager, 
                    task_description, 
                    android_world_task,
                    config.MAX_PLAN_STEPS,
                    300
                )
                
                result["task_name"] = task_name
                result["iteration"] = iteration + 1
                all_results.append(result)
                task_results[task_name].append(result)
                
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(f"     {status} - {result['duration']:.2f}s - {result['total_steps']} steps")
                
            except Exception as e:
                print(f"     ❌ ERROR - {str(e)}")
                all_results.append({
                    "success": False,
                    "result": "ERROR", 
                    "task_name": task_name,
                    "iteration": iteration + 1,
                    "duration": 0,
                    "total_steps": 0,
                    "error": str(e)
                })
    
    # Calculate benchmark summary
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r["success"])
    total_duration = sum(r["duration"] for r in all_results)
    total_steps = sum(r["total_steps"] for r in all_results)
    
    summary = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": total_tests - successful_tests,
        "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
        "total_duration": total_duration,
        "average_duration": total_duration / total_tests if total_tests > 0 else 0,
        "total_steps": total_steps,
        "average_steps": total_steps / total_tests if total_tests > 0 else 0
    }
    
    return {
        "summary": summary,
        "all_results": all_results,
        "task_results": task_results
    }

def get_predefined_tasks() -> List[Dict[str, str]]:
    """Get predefined test tasks"""
    return [
        {
            "name": "WiFi Settings Test",
            "description": "Test turning Wi-Fi on and off in Settings",
            "android_world_task": "settings_wifi"
        },
        {
            "name": "Alarm Creation Test", 
            "description": "Test creating and managing alarms",
            "android_world_task": "clock_alarm"
        },
        {
            "name": "Calculator Test",
            "description": "Test basic calculator operations",
            "android_world_task": "calculator_basic"
        },
        {
            "name": "Contacts Test",
            "description": "Test adding and managing contacts",
            "android_world_task": "contacts_add"
        },
        {
            "name": "Email Search Test",
            "description": "Test email search functionality",
            "android_world_task": "email_search"
        }
    ]

async def main():
    """Main CLI interface with proper error handling"""
    parser = argparse.ArgumentParser(description="Multi-Agent QA System for Android UI Testing")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--task", "-t", default="Test turning Wi-Fi on and off", help="QA task description")
    parser.add_argument("--android-task", "-a", default="settings_wifi", 
                       choices=config.ANDROID_WORLD_TASKS, help="Android World task type")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM interface for testing")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--api-key", help="Gemini API key (overrides config)")
    parser.add_argument("--max-steps", type=int, default=config.MAX_PLAN_STEPS, help="Maximum execution steps")
    parser.add_argument("--timeout", type=int, default=120, help="Task timeout in seconds")
    parser.add_argument("--iterations", type=int, default=1, help="Number of benchmark iterations")
    parser.add_argument("--list-tasks", action="store_true", help="List available Android World tasks")
    parser.add_argument("--export-logs", action="store_true", help="Export logs after execution")
    
    args = parser.parse_args()
    
    # List available tasks and exit
    if args.list_tasks:
        print("\n📋 Available Android World Tasks:")
        tasks = get_predefined_tasks()
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task['name']} ({task['android_world_task']})")
            print(f"      {task['description']}")
        return
    
    # Load and merge configuration
    yaml_config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.mock_llm:
        config.USE_MOCK_LLM = True
        print("🔧 Using Mock LLM interface")
    
    if args.api_key:
        config.GOOGLE_API_KEY = args.api_key
        print("🔧 Using provided API key")
    
    # Print system status
    print(f"\n🤖 Multi-Agent QA System")
    print(f"   LLM Mode: {'Mock' if config.USE_MOCK_LLM else 'Gemini API'}")
    print(f"   Android Device: {config.ANDROID_DEVICE_ID}")
    print(f"   Screenshots: {config.SCREENSHOTS_DIR}")
    print(f"   Logs: {config.LOGS_DIR}")
    
    # Initialize environment manager
    env_manager = None
    
    try:
        print("\n🔧 Initializing Multi-Agent QA System...")
        env_manager = EnvironmentManager()
        
        initialization_success = await env_manager.initialize()
        if not initialization_success:
            print("❌ Failed to initialize QA system")
            return
        
        print("✅ QA System initialized successfully")
        
        # Get system metrics
        metrics = env_manager.get_system_metrics()
        if "message" not in metrics:
            print(f"   System Integration: Agent-S {'✅' if metrics['system_integration']['agent_s_active'] else '❌'}")
            print(f"   Android World: {'✅' if metrics['system_integration']['android_world_connected'] else '❌'}")
        
        if args.benchmark:
            # Run benchmark tests
            tasks = get_predefined_tasks()
            results = await run_benchmark(env_manager, tasks, args.iterations)
            
            print(f"\n📊 Benchmark Results Summary:")
            summary = results["summary"]
            print(f"   Total Tests: {summary['total_tests']}")
            print(f"   Success Rate: {summary['success_rate']:.1%}")
            print(f"   Average Duration: {summary['average_duration']:.2f}s")
            print(f"   Average Steps: {summary['average_steps']:.1f}")
            
            # Detailed results by task
            print(f"\n📋 Results by Task:")
            for task_name, task_results in results["task_results"].items():
                successes = sum(1 for r in task_results if r["success"])
                total = len(task_results)
                avg_duration = sum(r["duration"] for r in task_results) / total if total > 0 else 0
                print(f"   {task_name}: {successes}/{total} ({successes/total:.1%}) - {avg_duration:.2f}s avg")
        
        else:
            # Run single task
            result = await run_single_test(
                env_manager, 
                args.task, 
                args.android_task,
                args.max_steps, 
                args.timeout
            )
            
            print(f"\n✅ Task Execution Results:")
            print(f"   Result: {result['result']}")
            print(f"   Success: {'✓' if result['success'] else '✗'}")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Total Steps: {result['total_steps']}")
            print(f"   Bug Detected: {'Yes' if result['bug_detected'] else 'No'}")
            
            if result['actions_summary']['total'] > 0:
                actions = result['actions_summary']
                print(f"   Action Success Rate: {actions['success_rate']:.1%} ({actions['successful']}/{actions['total']})")
            
            if result.get('supervisor_feedback'):
                print(f"   Supervisor Feedback: {result['supervisor_feedback'][:100]}...")
            
            if not result['success']:
                print(f"   Test ID for debugging: {result['test_id']}")
        
        # Export logs if requested
        if args.export_logs:
            print(f"\n📤 Exporting logs...")
            try:
                export_path = env_manager.planner_agent.logger.export_logs("json")
                print(f"   Logs exported to: {export_path}")
            except Exception as e:
                print(f"   Export failed: {e}")
        
        # Show final system metrics
        final_metrics = env_manager.get_system_metrics()
        if "test_summary" in final_metrics:
            test_summary = final_metrics["test_summary"]
            print(f"\n📈 Session Summary:")
            print(f"   Tests Run: {test_summary['total_tests']}")
            if test_summary['total_tests'] > 0:
                print(f"   Pass Rate: {test_summary['pass_rate']:.1%}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        logger.error(f"Main execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env_manager:
            print("\n🔚 Shutting down QA system...")
            try:
                await env_manager.shutdown()
                print("✅ Shutdown completed")
            except Exception as e:
                print(f"⚠️  Shutdown warning: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
