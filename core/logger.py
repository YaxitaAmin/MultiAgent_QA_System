# core/logger.py - CORRECTED VERSION WITH FLEXIBLE INTERFACE
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from loguru import logger as loguru_logger

@dataclass
class AgentAction:
    agent_type: str
    action_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: float
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    duration: float = 0.0  # Alias for execution_time

    def __post_init__(self):
        # Ensure duration and execution_time are synchronized
        if self.duration == 0.0 and self.execution_time > 0.0:
            self.duration = self.execution_time
        elif self.execution_time == 0.0 and self.duration > 0.0:
            self.execution_time = self.duration

@dataclass
class TestEpisode:
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
    """FIXED Comprehensive logging system for multi-agent QA testing"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup structured logging
        self.setup_logging()
        
        # Current episode tracking
        self.current_episode: Optional[TestEpisode] = None
        self.episodes: List[TestEpisode] = []
        
        print(f"[LOGGER INIT] QALogger initialized with log directory: {self.log_dir}")
        
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
    
    def start_episode(self, task_description: str) -> str:
        """Start a new test episode"""
        episode_id = f"episode_{int(time.time())}_{len(self.episodes)}"
        
        self.current_episode = TestEpisode(
            episode_id=episode_id,
            task_description=task_description,
            start_time=time.time()
        )
        
        print(f"[LOGGER] Started episode {episode_id}: {task_description}")
        loguru_logger.info(f"Started episode {episode_id}: {task_description}")
        return episode_id
    
    def end_episode(self, success: bool, final_result: str = ""):
        """End current test episode"""
        if not self.current_episode:
            loguru_logger.warning("No active episode to end")
            return
        
        self.current_episode.end_time = time.time()
        self.current_episode.success = success
        self.current_episode.final_result = final_result
        self.current_episode.total_steps = len(self.current_episode.actions)
        
        # Save episode to file
        self._save_episode(self.current_episode)
        
        # Add to episodes list
        self.episodes.append(self.current_episode)
        
        duration = self.current_episode.end_time - self.current_episode.start_time
        
        print(f"[LOGGER] Ended episode {self.current_episode.episode_id}: "
              f"Success={success}, Duration={duration:.2f}s, Steps={self.current_episode.total_steps}")
        
        loguru_logger.info(
            f"Ended episode {self.current_episode.episode_id}: "
            f"Success={success}, Duration={duration:.2f}s, Steps={self.current_episode.total_steps}"
        )
        
        self.current_episode = None
    
    def log_agent_action(self, 
                        agent_type: str = None,
                        action_type: str = None,
                        input_data: Dict[str, Any] = None,
                        output_data: Dict[str, Any] = None,
                        success: bool = True,
                        execution_time: float = 0.0,
                        error_message: Optional[str] = None,
                        # FLEXIBLE ALIASES FOR COMPATIBILITY
                        agent_name: str = None,
                        action: str = None,
                        duration: float = None,
                        details: Dict[str, Any] = None,
                        **kwargs):
        """FLEXIBLE Log an agent action with backward compatibility"""
        
        # Handle parameter aliases for backward compatibility
        final_agent_type = agent_type or agent_name or "unknown"
        final_action_type = action_type or action or "unknown"
        final_input_data = input_data or details or {}
        final_output_data = output_data or {}
        final_execution_time = execution_time or duration or 0.0
        
        # Merge any additional kwargs into input_data
        if kwargs:
            final_input_data.update(kwargs)
        
        action = AgentAction(
            agent_type=final_agent_type,
            action_type=final_action_type,
            input_data=final_input_data,
            output_data=final_output_data,
            timestamp=time.time(),
            success=success,
            execution_time=final_execution_time,
            error_message=error_message
        )
        
        if self.current_episode:
            self.current_episode.actions.append(action)
        
        print(f"[LOGGER] Agent {final_agent_type} - {final_action_type}: Success={success}, "
              f"Time={final_execution_time:.3f}s")
        
        loguru_logger.info(
            f"Agent {final_agent_type} - {final_action_type}: Success={success}, "
            f"Time={final_execution_time:.3f}s"
        )
        
        if error_message:
            print(f"[LOGGER] Agent {final_agent_type} error: {error_message}")
            loguru_logger.error(f"Agent {final_agent_type} error: {error_message}")
    
    def log_ui_interaction(self, 
                          action_type: str = None, 
                          target: str = None, 
                          result: str = None,
                          # FLEXIBLE ALIASES
                          action: str = None,
                          **kwargs):
        """FLEXIBLE Log UI interaction with backward compatibility"""
        final_action = action_type or action or "unknown"
        final_target = target or "unknown"
        final_result = result or "unknown"
        
        print(f"[LOGGER] UI Interaction: {final_action} on {final_target} -> {final_result}")
        loguru_logger.info(f"UI Interaction: {final_action} on {final_target} -> {final_result}")
    
    def log_verification_result(self, expected: str, actual: str, passed: bool):
        """Log verification result"""
        print(f"[LOGGER] Verification: Expected='{expected}', Actual='{actual}', Passed={passed}")
        loguru_logger.info(f"Verification: Expected='{expected}', Actual='{actual}', Passed={passed}")
    
    def _save_episode(self, episode: TestEpisode):
        """Save episode data to JSON file"""
        episode_file = self.log_dir / f"{episode.episode_id}.json"
        
        try:
            episode_data = asdict(episode)
            with open(episode_file, 'w') as f:
                json.dump(episode_data, f, indent=2, default=str)
                
            print(f"[LOGGER] Saved episode data to {episode_file}")
            
        except Exception as e:
            print(f"[LOGGER] ERROR: Failed to save episode {episode.episode_id}: {e}")
            loguru_logger.error(f"Failed to save episode {episode.episode_id}: {e}")
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of all episodes"""
        total_episodes = len(self.episodes)
        successful_episodes = sum(1 for ep in self.episodes if ep.success)
        
        if total_episodes == 0:
            return {
                "total_episodes": 0,
                "successful_episodes": 0,
                "success_rate": 0.0,
                "episodes": []
            }
        
        return {
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "success_rate": successful_episodes / total_episodes,
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "task": ep.task_description,
                    "success": ep.success,
                    "duration": (ep.end_time or ep.start_time) - ep.start_time,
                    "steps": ep.total_steps
                }
                for ep in self.episodes
            ]
        }
    
    def export_logs(self, format: str = "json") -> str:
        """Export all logs to specified format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            if format == "json":
                export_file = self.log_dir / f"qa_export_{timestamp}.json"
                export_data = {
                    "export_timestamp": timestamp,
                    "summary": self.get_episode_summary(),
                    "episodes": [asdict(ep) for ep in self.episodes],
                    "system_info": {
                        "total_episodes": len(self.episodes),
                        "log_directory": str(self.log_dir),
                        "export_format": format
                    }
                }
                
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                print(f"[LOGGER] Exported logs to {export_file}")
                return str(export_file)
            
            elif format == "csv":
                import pandas as pd
                export_file = self.log_dir / f"qa_export_{timestamp}.csv"
                
                # Flatten episode data for CSV
                rows = []
                for ep in self.episodes:
                    for action in ep.actions:
                        rows.append({
                            "episode_id": ep.episode_id,
                            "task_description": ep.task_description,
                            "episode_success": ep.success,
                            "agent_type": action.agent_type,
                            "action_type": action.action_type,
                            "action_success": action.success,
                            "execution_time": action.execution_time,
                            "timestamp": datetime.fromtimestamp(action.timestamp).isoformat(),
                            "error_message": action.error_message or ""
                        })
                
                if rows:
                    df = pd.DataFrame(rows)
                    df.to_csv(export_file, index=False)
                    print(f"[LOGGER] Exported logs to {export_file}")
                    return str(export_file)
                else:
                    print(f"[LOGGER] No data to export")
                    return str(self.log_dir / "no_data.csv")
            
        except Exception as e:
            print(f"[LOGGER] ERROR: Export failed: {e}")
            loguru_logger.error(f"Export failed: {e}")
            return str(self.log_dir / f"export_failed_{timestamp}.txt")
        
        return str(export_file)
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary by agent type"""
        agent_stats = {}
        
        for episode in self.episodes:
            for action in episode.actions:
                agent_type = action.agent_type
                
                if agent_type not in agent_stats:
                    agent_stats[agent_type] = {
                        "total_actions": 0,
                        "successful_actions": 0,
                        "total_time": 0.0,
                        "errors": []
                    }
                
                agent_stats[agent_type]["total_actions"] += 1
                if action.success:
                    agent_stats[agent_type]["successful_actions"] += 1
                else:
                    if action.error_message:
                        agent_stats[agent_type]["errors"].append(action.error_message)
                
                agent_stats[agent_type]["total_time"] += action.execution_time
        
        # Calculate derived statistics
        for agent_type, stats in agent_stats.items():
            total = stats["total_actions"]
            if total > 0:
                stats["success_rate"] = stats["successful_actions"] / total
                stats["average_execution_time"] = stats["total_time"] / total
            else:
                stats["success_rate"] = 0.0
                stats["average_execution_time"] = 0.0
        
        return agent_stats
    
    def clear_logs(self):
        """Clear all episode data (useful for testing)"""
        self.episodes = []
        self.current_episode = None
        print(f"[LOGGER] Cleared all episode data")
        loguru_logger.info("Cleared all episode data")
