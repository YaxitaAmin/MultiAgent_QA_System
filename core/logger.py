# core/logger.py - COMPLETE VERSION WITH ALL MISSING CLASSES
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from loguru import logger as loguru_logger


@dataclass
class AgentAction:
    """Represents an action taken by an agent"""
    agent_name: str  # Changed from agent_type for compatibility
    action_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: float
    success: bool
    duration: float  # Changed from execution_time for compatibility
    error_message: Optional[str] = None
    
    # Aliases for backward compatibility
    @property
    def agent_type(self) -> str:
        return self.agent_name
    
    @property
    def execution_time(self) -> float:
        return self.duration

@dataclass
class TestMetrics:
    """Test execution metrics"""
    total_actions: int
    successful_actions: int
    failed_actions: int
    total_duration: float
    average_action_time: float
    success_rate: float
    agent_performance: Dict[str, Any]

# ✅ MISSING CLASS: QATestResult that other modules expect
@dataclass
class QATestResult:
    """Complete QA test result"""
    test_id: str
    task_name: str
    start_time: float
    end_time: float
    actions: List[AgentAction]
    final_result: str  # "PASS", "FAIL", "ERROR"
    bug_detected: bool
    supervisor_feedback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "test_id": self.test_id,
            "task_name": self.task_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time,
            "actions": [asdict(action) for action in self.actions],
            "final_result": self.final_result,
            "bug_detected": self.bug_detected,
            "supervisor_feedback": self.supervisor_feedback,
            "metrics": {
                "total_actions": len(self.actions),
                "successful_actions": sum(1 for a in self.actions if a.success),
                "success_rate": sum(1 for a in self.actions if a.success) / len(self.actions) if self.actions else 0,
                "duration": self.end_time - self.start_time
            }
        }

@dataclass
class TestEpisode:
    """Test episode data structure"""
    episode_id: str
    task_description: str
    start_time: float
    end_time: Optional[float] = None
    total_steps: int = 0
    success: bool = False
    final_result: Optional[str] = None
    actions: List[AgentAction] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []

class QALogger:
    """Comprehensive logging system for multi-agent QA testing"""
    
    def __init__(self, component_name: str = "QALogger", log_dir: str = "logs"):
        self.component_name = component_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup structured logging
        self.setup_logging()
        
        # Current test tracking
        self.current_test_id: Optional[str] = None
        self.current_test_actions: List[AgentAction] = []
        self.current_test_start_time: Optional[float] = None
        self.current_task_name: Optional[str] = None
        
        # Episode tracking (for backward compatibility)
        self.current_episode: Optional[TestEpisode] = None
        self.episodes: List[TestEpisode] = []
        
        # All completed tests
        self.completed_tests: List[QATestResult] = []
        
        self.info(f"QALogger initialized: {component_name}")
        
    def setup_logging(self):
        """Setup loguru logging configuration"""
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        
        # Remove default handler
        loguru_logger.remove()
        
        # Add console handler
        loguru_logger.add(
            lambda msg: print(msg, end=""),
            format=log_format,
            level="INFO"
        )
        
        # Add file handler
        log_file = self.log_dir / f"qa_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        loguru_logger.add(
            str(log_file),
            format=log_format,
            level="DEBUG",
            rotation="10 MB"
        )
    
    # ✅ MISSING METHODS that other modules expect
    def start_test(self, test_id: str, task_name: str) -> str:
        """Start a new QA test"""
        self.current_test_id = test_id
        self.current_task_name = task_name
        self.current_test_start_time = time.time()
        self.current_test_actions = []
        
        self.info(f"Started QA test: {test_id} - {task_name}")
        return test_id
    
    def finish_test(self, task_name: str, final_result: str, bug_detected: bool, 
                   supervisor_feedback: str = "") -> QATestResult:
        """Finish current QA test and return result"""
        if not self.current_test_id:
            # Create emergency test result
            test_result = QATestResult(
                test_id=f"emergency_test_{int(time.time())}",
                task_name=task_name,
                start_time=time.time(),
                end_time=time.time(),
                actions=[],
                final_result=final_result,
                bug_detected=bug_detected,
                supervisor_feedback=supervisor_feedback
            )
        else:
            test_result = QATestResult(
                test_id=self.current_test_id,
                task_name=self.current_task_name or task_name,
                start_time=self.current_test_start_time or time.time(),
                end_time=time.time(),
                actions=self.current_test_actions.copy(),
                final_result=final_result,
                bug_detected=bug_detected,
                supervisor_feedback=supervisor_feedback
            )
        
        # Save test result
        self.completed_tests.append(test_result)
        self._save_test_result(test_result)
        
        # Reset current test state
        self.current_test_id = None
        self.current_task_name = None
        self.current_test_start_time = None
        self.current_test_actions = []
        
        duration = test_result.end_time - test_result.start_time
        self.info(f"Finished QA test: {test_result.test_id} - {final_result} - {duration:.2f}s")
        
        return test_result
    
    def log_agent_action(self, action: AgentAction):
        """Log an agent action (compatible with AgentAction object)"""
        if self.current_test_actions is not None:
            self.current_test_actions.append(action)
        
        success_str = "✅" if action.success else "❌"
        self.info(f"{success_str} {action.agent_name}: {action.action_type} ({action.duration:.3f}s)")
        
        if action.error_message:
            self.error(f"Agent {action.agent_name} error: {action.error_message}")
    
    def _save_test_result(self, test_result: QATestResult):
        """Save test result to JSON file"""
        result_file = self.log_dir / f"test_result_{test_result.test_id}.json"
        
        try:
            with open(result_file, 'w') as f:
                json.dump(test_result.to_dict(), f, indent=2, default=str)
            
            self.debug(f"Saved test result to {result_file}")
            
        except Exception as e:
            self.error(f"Failed to save test result {test_result.test_id}: {e}")
    
    # Logging convenience methods
    def info(self, message: str):
        """Log info message"""
        loguru_logger.info(f"[{self.component_name}] {message}")
    
    def debug(self, message: str):
        """Log debug message"""
        loguru_logger.debug(f"[{self.component_name}] {message}")
    
    def warning(self, message: str):
        """Log warning message"""
        loguru_logger.warning(f"[{self.component_name}] {message}")
    
    def error(self, message: str):
        """Log error message"""
        loguru_logger.error(f"[{self.component_name}] {message}")
    
    # ✅ BACKWARD COMPATIBILITY METHODS for existing code
    def start_episode(self, task_description: str) -> str:
        """Start a new test episode (backward compatibility)"""
        episode_id = f"episode_{int(time.time())}_{len(self.episodes)}"
        
        self.current_episode = TestEpisode(
            episode_id=episode_id,
            task_description=task_description,
            start_time=time.time()
        )
        
        self.info(f"Started episode {episode_id}: {task_description}")
        return episode_id
    
    def end_episode(self, success: bool, final_result: str = ""):
        """End current test episode (backward compatibility)"""
        if not self.current_episode:
            self.warning("No active episode to end")
            return
        
        self.current_episode.end_time = time.time()
        self.current_episode.success = success
        self.current_episode.final_result = final_result
        self.current_episode.total_steps = len(self.current_episode.actions)
        
        # Add to episodes list
        self.episodes.append(self.current_episode)
        
        duration = self.current_episode.end_time - self.current_episode.start_time
        self.info(f"Ended episode {self.current_episode.episode_id}: Success={success}, Duration={duration:.2f}s")
        
        self.current_episode = None
    
    def log_ui_interaction(self, action_type: str, target: str, result: str):
        """Log UI interaction"""
        self.info(f"UI Interaction: {action_type} on {target} -> {result}")
    
    def log_verification_result(self, expected: str, actual: str, passed: bool):
        """Log verification result"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.info(f"Verification {status}: Expected='{expected}', Actual='{actual}'")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all completed tests"""
        if not self.completed_tests:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "error_tests": 0,
                "success_rate": 0.0,
                "tests": []
            }
        
        total_tests = len(self.completed_tests)
        passed_tests = sum(1 for t in self.completed_tests if t.final_result == "PASS")
        failed_tests = sum(1 for t in self.completed_tests if t.final_result == "FAIL")
        error_tests = sum(1 for t in self.completed_tests if t.final_result == "ERROR")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "tests": [
                {
                    "test_id": t.test_id,
                    "task_name": t.task_name,
                    "result": t.final_result,
                    "duration": t.end_time - t.start_time,
                    "actions": len(t.actions)
                }
                for t in self.completed_tests
            ]
        }
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary by agent"""
        agent_stats = {}
        
        for test in self.completed_tests:
            for action in test.actions:
                agent_name = action.agent_name
                
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {
                        "total_actions": 0,
                        "successful_actions": 0,
                        "total_time": 0.0,
                        "errors": []
                    }
                
                agent_stats[agent_name]["total_actions"] += 1
                if action.success:
                    agent_stats[agent_name]["successful_actions"] += 1
                else:
                    if action.error_message:
                        agent_stats[agent_name]["errors"].append(action.error_message)
                
                agent_stats[agent_name]["total_time"] += action.duration
        
        # Calculate derived statistics
        for agent_name, stats in agent_stats.items():
            total = stats["total_actions"]
            if total > 0:
                stats["success_rate"] = stats["successful_actions"] / total
                stats["average_execution_time"] = stats["total_time"] / total
            else:
                stats["success_rate"] = 0.0
                stats["average_execution_time"] = 0.0
        
        return agent_stats
    
    def export_logs(self, format: str = "json") -> str:
        """Export all logs to specified format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            if format == "json":
                export_file = self.log_dir / f"qa_export_{timestamp}.json"
                export_data = {
                    "export_timestamp": timestamp,
                    "test_summary": self.get_test_summary(),
                    "completed_tests": [test.to_dict() for test in self.completed_tests],
                    "agent_performance": self.get_agent_performance_summary()
                }
                
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.info(f"Exported logs to {export_file}")
                return str(export_file)
            
            else:
                self.warning(f"Export format '{format}' not supported, using JSON")
                return self.export_logs("json")
            
        except Exception as e:
            self.error(f"Export failed: {e}")
            return ""
    
    def clear_logs(self):
        """Clear all test data"""
        self.completed_tests = []
        self.episodes = []
        self.current_test_id = None
        self.current_test_actions = []
        self.current_episode = None
        self.info("Cleared all test data")
