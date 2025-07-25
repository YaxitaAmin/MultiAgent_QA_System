# main_orchestrator.py
import asyncio
import json
import yaml
from pathlib import Path
import argparse
from typing import Dict, Any
from loguru import logger

from env_manager import MultiAgentQAManager

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "agents": {
            "planner": {
                "model": "gemini-1.5-flash",
                "temperature": 0.1,
                "max_tokens": 1000
            },
            "executor": {
                "model": "gemini-1.5-flash",
                "temperature": 0.0,
                "max_tokens": 500
            },
            "verifier": {
                "model": "gemini-1.5-flash",
                "temperature": 0.1,
                "max_tokens": 600
            },
            "supervisor": {
                "model": "gemini-1.5-pro",
                "temperature": 0.2,
                "max_tokens": 1000
            }
        },
        "android_env": {
            "task_name": "settings_wifi",
            "task_timeout": 300,
            "max_steps": 50,
            "screenshot_dir": "screenshots"
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs",
            "enable_screenshots": True
        },
        "use_mock_llm": False
    }

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Multi-Agent QA System for Android UI Testing")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--task", "-t", default="Test turning Wi-Fi on and off", help="QA task description")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM interface for testing")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--api-key", help="Gemini API key (overrides config)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum execution steps")
    parser.add_argument("--timeout", type=int, default=300, help="Task timeout in seconds")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.mock_llm:
        config["use_mock_llm"] = True
    
    if args.api_key:
        config["gemini_api_key"] = args.api_key
    
    config["android_env"]["max_steps"] = args.max_steps
    config["android_env"]["task_timeout"] = args.timeout
    
    # Initialize QA manager
    qa_manager = MultiAgentQAManager(config)
    
    # Print system status
    status = qa_manager.get_system_status()
    print(f"\n🤖 Multi-Agent QA System Status:")
    print(f"   Environment: {status['environment']}")
    print(f"   LLM Interface: {status['llm_interface']}")
    print(f"   Episodes Completed: {status['episodes_completed']}")
    print(f"   Logs Directory: {status['logs_directory']}")
    
    try:
        if args.benchmark:
            # Run benchmark tests
            benchmark_tasks = [
                "Test turning Wi-Fi on and off",
                "Open Settings and navigate to Display settings",
                "Test airplane mode toggle"
            ]
            
            print(f"\n🚀 Running benchmark with {len(benchmark_tasks)} tasks...")
            results = await qa_manager.run_benchmark(benchmark_tasks, iterations=2)
            
            print(f"\n📊 Benchmark Results:")
            print(f"   Total Tasks: {results['summary']['total_tasks']}")
            print(f"   Success Rate: {results['summary']['success_rate']:.2%}")
            print(f"   Average Time: {results['summary']['average_task_time']:.2f}s")
            print(f"   Average Steps: {results['summary']['average_steps_per_task']:.1f}")
            
        else:
            # Run single task
            print(f"\n🎯 Executing QA Task: {args.task}")
            result = await qa_manager.execute_qa_task(args.task, args.max_steps, args.timeout)
            
            print(f"\n✅ Task Results:")
            print(f"   Success: {'✓' if result['success'] else '✗'}")
            print(f"   Duration: {result['total_time']:.2f}s")
            print(f"   Steps: {result['total_steps']}")
            print(f"   Episode ID: {result['episode_id']}")
            
            if 'plan' in result:
                plan = result['plan']
                print(f"   Plan: {plan['completed_subgoals']}/{plan['subgoals_count']} subgoals completed")
            
            if 'execution' in result:
                exec_summary = result['execution']
                print(f"   Execution: {exec_summary['success_rate']:.2%} success rate")
            
            if 'verification' in result:
                verif_summary = result['verification']
                print(f"   Verification: {verif_summary['pass_rate']:.2%} pass rate")
            
            if 'logs_exported_to' in result:
                print(f"   Logs: {result['logs_exported_to']}")
            
            # Show error if failed
            if not result['success'] and 'error' in result:
                print(f"   Error: {result['error']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        logger.error(f"Main execution failed: {e}")
    finally:
        print("\n🔚 Shutting down...")
        qa_manager.android_env.close()

if __name__ == "__main__":
    asyncio.run(main())
