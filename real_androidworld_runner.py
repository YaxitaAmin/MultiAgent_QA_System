# real_androidworld_runner.py - FIXED VERSION
import asyncio
import time
import json
import os
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger

from env_manager import MultiAgentQAManager
from config.default_config import get_default_config

class RealAndroidWorldRunner:
    """Runner for testing multi-agent system on real AndroidWorld"""
    
    def __init__(self, use_real_device: bool = True, adb_device_serial: str = None):
        self.use_real_device = use_real_device
        self.adb_device_serial = adb_device_serial
        self.results = []
        
        print(f"[REAL_RUNNER] Initializing Real AndroidWorld Runner")
        print(f"[REAL_RUNNER] Use real device: {use_real_device}")
        print(f"[REAL_RUNNER] ADB device serial: {adb_device_serial}")
    
    async def run_real_androidworld_tests(self, tasks: List[str]) -> Dict[str, Any]:
        """Run comprehensive tests on real AndroidWorld"""
        
        print(f"[REAL_RUNNER] Starting real AndroidWorld tests with {len(tasks)} tasks")
        
        overall_results = {
            "test_session_id": f"real_test_{int(time.time())}",
            "environment": "real_androidworld" if self.use_real_device else "mock_androidworld",
            "device_info": {},
            "tasks": tasks,
            "results": [],
            "summary": {}
        }
        
        for i, task in enumerate(tasks):
            print(f"[REAL_RUNNER] Running task {i+1}/{len(tasks)}: {task}")
            
            try:
                # Create real AndroidWorld configuration
                config = self._get_real_config(task)
                
                # Initialize QA Manager with configuration for real testing
                qa_manager = MultiAgentQAManager(config)
                
                # Try to initialize real AndroidWorld if requested
                if self.use_real_device:
                    real_init_success = qa_manager.android_env.initialize_real_androidworld(
                        adb_device_serial=self.adb_device_serial
                    )
                    
                    if real_init_success:
                        print(f"[REAL_RUNNER] ✅ Real AndroidWorld initialized")
                    else:
                        print(f"[REAL_RUNNER] ⚠️ Real AndroidWorld failed, using mock mode")
                
                # Get device status
                device_status = qa_manager.android_env.get_real_device_status()
                overall_results["device_info"] = device_status
                
                print(f"[REAL_RUNNER] Device status: {device_status}")
                
                # Execute task
                start_time = time.time()
                
                result = await qa_manager.execute_qa_task(
                    task_description=task,
                    max_steps=20,  # More steps for real environment
                    timeout=120    # Longer timeout for real execution
                )
                
                execution_time = time.time() - start_time
                
                # Enhanced result with real environment data
                enhanced_result = {
                    **result,
                    "task_description": task,
                    "execution_time": execution_time, 
                    "environment": "real_androidworld" if not qa_manager.android_env.mock_mode else "mock_androidworld",
                    "device_info": device_status,
                    "real_screenshots": self._collect_screenshots(qa_manager),
                    "performance_analysis": self._analyze_performance(result, execution_time, qa_manager.android_env.mock_mode)
                }
                
                overall_results["results"].append(enhanced_result)
                
                print(f"[REAL_RUNNER] ✅ Task completed - Success: {result.get('success', False)}")
                
                # Brief pause between tasks
                await asyncio.sleep(2.0)
                
            except Exception as e:
                error_result = {
                    "task_description": task,
                    "success": False,
                    "error": str(e),
                    "environment": "error",
                    "execution_time": 0.0
                }
                overall_results["results"].append(error_result)
                
                print(f"[REAL_RUNNER] ❌ Task failed: {e}")
        
        # Generate summary
        overall_results["summary"] = self._generate_test_summary(overall_results["results"])
        
        # Save results
        self._save_test_results(overall_results)
        
        print(f"[REAL_RUNNER] ✅ All tests completed")
        return overall_results
    
    def _get_real_config(self, task_name: str) -> Dict[str, Any]:
        """Get configuration for real AndroidWorld testing"""
        
        config = get_default_config()
        
        # Override for real AndroidWorld testing
        config.update({
            "use_mock_llm": os.getenv("GEMINI_API_KEY") is None,  # Use real LLM if API key available
            "android_env": {
                "task_name": self._map_task_to_androidworld(task_name),
                "screenshot_dir": "real_screenshots",
                "timeout": 120
            },
            "agents": {
                "planner": {
                    "model": "gemini-pro" if os.getenv("GEMINI_API_KEY") else "mock",
                    "temperature": 0.1,
                    "max_subgoals": 15
                },
                "executor": {
                    "model": "gemini-pro" if os.getenv("GEMINI_API_KEY") else "mock",
                    "temperature": 0.0,
                    "retry_limit": 5
                },
                "verifier": {
                    "model": "gemini-pro" if os.getenv("GEMINI_API_KEY") else "mock",
                    "temperature": 0.0,
                    "confidence_threshold": 0.6
                },
                "supervisor": {
                    "model": "gemini-pro" if os.getenv("GEMINI_API_KEY") else "mock",
                    "temperature": 0.2,
                    "analysis_depth": "comprehensive"
                }
            }
        })
        
        return config
    
    def _map_task_to_androidworld(self, task_description: str) -> str:
        """Map task description to AndroidWorld task name"""
        
        task_mapping = {
            "Test turning Wi-Fi on and off": "settings_wifi",
            "Navigate to Bluetooth settings": "settings_bluetooth", 
            "Open Calculator and perform calculation": "calculator_basic",
            "Set alarm for 7:30 AM": "clock_alarm",
            "Check device storage": "settings_storage",
            "Test airplane mode toggle": "settings_airplane_mode"
        }
        
        # Direct lookup
        if task_description in task_mapping:
            return task_mapping[task_description]
        
        # Fuzzy matching
        task_lower = task_description.lower()
        if "wifi" in task_lower or "wi-fi" in task_lower:
            return "settings_wifi"
        elif "bluetooth" in task_lower:
            return "settings_bluetooth"
        elif "calculator" in task_lower:
            return "calculator_basic"
        elif "alarm" in task_lower or "clock" in task_lower:
            return "clock_alarm"
        elif "storage" in task_lower:
            return "settings_storage"
        elif "airplane" in task_lower:
            return "settings_airplane_mode"
        else:
            return "settings_wifi"  # Default fallback
    
    def _collect_screenshots(self, qa_manager: MultiAgentQAManager) -> List[str]:
        """Collect screenshots from execution"""
        
        screenshot_dir = Path("real_screenshots")
        screenshot_dir.mkdir(exist_ok=True)
        
        screenshots = list(screenshot_dir.glob("*.png"))
        screenshots.sort(key=lambda x: x.stat().st_mtime)
        
        return [str(path) for path in screenshots[-10:]]  # Last 10 screenshots
    
    def _analyze_performance(self, result: Dict[str, Any], execution_time: float, is_mock: bool) -> Dict[str, Any]:
        """Analyze performance"""
        
        return {
            "execution_time": execution_time,
            "vs_mock_ratio": execution_time / 0.25 if not is_mock else 1.0,
            "device_interactions": result.get("total_steps", 0),
            "success_confidence": result.get("execution", {}).get("success_rate", 0.0),
            "environment_type": "mock" if is_mock else "real",
            "stability_score": self._assess_stability(result),
            "challenges_detected": self._identify_challenges(result, is_mock)
        }
    
    def _assess_stability(self, result: Dict[str, Any]) -> float:
        """Assess execution stability"""
        
        factors = []
        
        # Check execution consistency
        exec_data = result.get("execution", {})
        success_rate = exec_data.get("success_rate", 0)
        factors.append(success_rate)
        
        # Check verification confidence
        verif_data = result.get("verification", {})
        confidence = verif_data.get("average_confidence", 0)
        factors.append(confidence)
        
        # Check for errors
        if result.get("error"):
            factors.append(0.0)
        else:
            factors.append(1.0)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def _identify_challenges(self, result: Dict[str, Any], is_mock: bool) -> List[str]:
        """Identify execution challenges"""
        
        challenges = []
        
        if is_mock:
            challenges.append("Mock environment - limited real-world complexity")
        
        total_time = result.get("total_time", 0)
        if total_time > 30:
            challenges.append("Long execution time")
        
        total_steps = result.get("total_steps", 0)
        if total_steps > 15:
            challenges.append("High step count - complex navigation")
        
        if result.get("error"):
            challenges.append(f"Execution error: {result['error']}")
        
        success_rate = result.get("execution", {}).get("success_rate", 0)
        if success_rate < 0.6:
            challenges.append("Low success rate")
        
        return challenges
    
    def _generate_test_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate test summary"""
        
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.get("success", False))
        
        if total_tasks == 0:
            return {"error": "No tasks executed"}
        
        total_time = sum(r.get("execution_time", 0) for r in results)
        avg_execution_time = total_time / total_tasks
        
        # Calculate performance metrics
        performance_scores = []
        stability_scores = []
        
        for result in results:
            perf_analysis = result.get("performance_analysis", {})
            performance_scores.append(perf_analysis.get("success_confidence", 0))
            stability_scores.append(perf_analysis.get("stability_score", 0.5))
        
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0.5
        
        # Check environment types
        real_tests = sum(1 for r in results if r.get("environment") == "real_androidworld")
        mock_tests = sum(1 for r in results if r.get("environment") == "mock_androidworld")
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks,
            "average_execution_time": avg_execution_time,
            "performance_score": avg_performance,
            "stability_score": avg_stability,
            "environment_breakdown": {
                "real_tests": real_tests,
                "mock_tests": mock_tests,
                "real_percentage": real_tests / total_tasks if total_tasks > 0 else 0
            },
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations"""
        
        recommendations = []
        
        # Check if any real testing occurred
        real_tests = [r for r in results if r.get("environment") == "real_androidworld"]
        if not real_tests:
            recommendations.append("Set up real Android device/emulator for actual AndroidWorld testing")
        
        # Performance recommendations
        avg_time = sum(r.get("execution_time", 0) for r in results) / len(results) if results else 0
        if avg_time > 5.0:
            recommendations.append("Optimize action timing and execution efficiency")
        
        success_rate = sum(1 for r in results if r.get("success", False)) / len(results) if results else 0
        if success_rate < 0.7:
            recommendations.append("Improve agent coordination and task completion")
        
        # Environment-specific recommendations
        if real_tests:
            recommendations.append("Real AndroidWorld integration successful - consider expanding test coverage")
        
        return recommendations
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results"""
        
        output_dir = Path("real_androidworld_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"androidworld_test_{timestamp}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[REAL_RUNNER] Results saved to: {filepath}")
