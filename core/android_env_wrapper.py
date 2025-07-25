# core/android_env_wrapper.py
import time
import json
from typing import Dict, Any, Optional, Tuple,List
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

try:
    from android_env import AndroidEnv
    ANDROID_ENV_AVAILABLE = True
except ImportError:
    ANDROID_ENV_AVAILABLE = False
    logger.warning("AndroidEnv not available, using mock environment")

from .ui_utils import UITreeParser, UIState

class AndroidEnvWrapper:
    """Wrapper for AndroidEnv with additional utilities"""
    
    def __init__(self, task_name: str = "settings_wifi", screenshot_dir: str = "screenshots"):
        self.task_name = task_name
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(exist_ok=True)
        self.ui_parser = UITreeParser()
        self.step_count = 0
        
        if ANDROID_ENV_AVAILABLE:
            self.env = AndroidEnv(task_name=task_name)
            self.mock_mode = False
            logger.info(f"AndroidEnv initialized for task: {task_name}")
        else:
            self.env = None
            self.mock_mode = True
            logger.info("Using mock Android environment")
        
        self.current_state = None
        self.action_history = []
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation"""
        self.step_count = 0
        self.action_history = []
        
        if not self.mock_mode:
            obs = self.env.reset()
            self.current_state = self._process_observation(obs)
            return self.current_state
        else:
            # Mock initial state
            return self._create_mock_observation()
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action and return observation, reward, done, info"""
        self.step_count += 1
        self.action_history.append({
            "step": self.step_count,
            "action": action,
            "timestamp": time.time()
        })
        
        if not self.mock_mode:
            obs, reward, done, info = self.env.step(action)
            processed_obs = self._process_observation(obs)
            self.current_state = processed_obs
            return processed_obs, reward, done, info
        else:
            # Mock response
            return self._mock_step(action)
    
    def get_ui_state(self) -> UIState:
        """Get current UI state with parsed elements"""
        if self.current_state and 'ui_state' in self.current_state:
            return self.current_state['ui_state']
        return UIState(elements=[], hierarchy={})
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render current screen"""
        if not self.mock_mode and hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        else:
            # Return mock screenshot
            return np.zeros((1920, 1080, 3), dtype=np.uint8)
    
    def save_screenshot(self) -> str:
        """Save current screenshot and return path"""
        screenshot = self.render()
        if screenshot is not None:
            filename = f"step_{self.step_count:03d}_{int(time.time())}.png"
            filepath = self.screenshot_dir / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
            return str(filepath)
        return ""
    
    def _process_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw observation into structured format"""
        processed = {
            "screenshot": obs.get("screenshot"),
            "timestamp": time.time(),
            "step": self.step_count
        }
        
        # Parse UI hierarchy if available
        if "ui_hierarchy" in obs:
            ui_state = self.ui_parser.parse_ui_hierarchy(obs["ui_hierarchy"])
            processed["ui_state"] = ui_state
            processed["ui_dump"] = obs["ui_hierarchy"]
        
        return processed
    
    def _create_mock_observation(self) -> Dict[str, Any]:
        """Create mock observation for testing"""
        mock_ui_dump = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
        <hierarchy rotation="0">
            <node index="0" text="Settings" resource-id="android:id/title" class="android.widget.TextView" 
                  bounds="[100,100][300,150]" clickable="true" enabled="true" />
            <node index="1" text="Wi-Fi" resource-id="com.android.settings:id/wifi_settings" 
                  class="android.widget.TextView" bounds="[100,200][300,250]" clickable="true" enabled="true" />
            <node index="2" text="" resource-id="com.android.settings:id/wifi_switch" 
                  class="android.widget.Switch" bounds="[400,200][450,250]" 
                  clickable="true" enabled="true" checkable="true" checked="false" />
        </hierarchy>"""
        
        ui_state = self.ui_parser.parse_ui_hierarchy(mock_ui_dump)
        
        return {
            "screenshot": np.zeros((1920, 1080, 3), dtype=np.uint8),
            "ui_state": ui_state,
            "ui_dump": mock_ui_dump,
            "timestamp": time.time(),
            "step": self.step_count
        }
    
    def _mock_step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Mock step execution"""
        # Simulate state changes based on action
        if action.get("action_type") == "touch":
            # Mock successful touch
            reward = 1.0
            done = self.step_count >= 5  # End after 5 steps
        else:
            reward = 0.0
            done = False
        
        obs = self._create_mock_observation()
        info = {"success": True, "mock": True}
        
        return obs, reward, done, info
    
    def close(self):
        """Close environment"""
        if not self.mock_mode and self.env:
            self.env.close()
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get history of all actions taken"""
        return self.action_history.copy()
