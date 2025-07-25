# core/android_env_wrapper.py - FIXED VERSION FOR PROPER INTEGRATION
import time
import json
from typing import Dict, Any, Optional, Tuple, List
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
    """FIXED Wrapper for AndroidEnv with complete action interface"""
    
    def __init__(self, task_name: str = "settings_wifi", screenshot_dir: str = "screenshots"):
        self.task_name = task_name
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(exist_ok=True)
        self.ui_parser = UITreeParser()
        self.step_count = 0
        
        print(f"[ANDROID_ENV] Initializing AndroidEnvWrapper for task: {task_name}")
        
        if ANDROID_ENV_AVAILABLE:
            try:
                self.env = AndroidEnv(task_name=task_name)
                self.mock_mode = False
                logger.info(f"AndroidEnv initialized for task: {task_name}")
                print(f"[ANDROID_ENV] Real AndroidEnv initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize AndroidEnv: {e}, falling back to mock")
                print(f"[ANDROID_ENV] AndroidEnv failed, using mock: {e}")
                self.env = None
                self.mock_mode = True
        else:
            self.env = None
            self.mock_mode = True
            logger.info("Using mock Android environment")
            print(f"[ANDROID_ENV] Using mock Android environment")
        
        self.current_state = None
        self.action_history = []
        self.ui_state_cache = None
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation"""
        print(f"[ANDROID_ENV] Resetting environment (mock_mode: {self.mock_mode})")
        
        self.step_count = 0
        self.action_history = []
        
        if not self.mock_mode:
            try:
                obs = self.env.reset()
                self.current_state = self._process_observation(obs)
                print(f"[ANDROID_ENV] Real reset completed")
                return self.current_state
            except Exception as e:
                logger.error(f"AndroidEnv reset failed: {e}")
                print(f"[ANDROID_ENV] Reset failed, creating mock observation: {e}")
                return self._create_mock_observation()
        else:
            # Mock initial state
            mock_obs = self._create_mock_observation()
            self.current_state = mock_obs
            print(f"[ANDROID_ENV] Mock reset completed")
            return mock_obs
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action and return observation, reward, done, info"""
        self.step_count += 1
        self.action_history.append({
            "step": self.step_count,
            "action": action,
            "timestamp": time.time()
        })
        
        print(f"[ANDROID_ENV] Step {self.step_count}: {action.get('action_type', 'unknown')}")
        
        if not self.mock_mode:
            try:
                obs, reward, done, info = self.env.step(action)
                processed_obs = self._process_observation(obs)
                self.current_state = processed_obs
                return processed_obs, reward, done, info
            except Exception as e:
                logger.error(f"AndroidEnv step failed: {e}")
                return self._mock_step(action)
        else:
            # Mock response
            return self._mock_step(action)
    
    def get_ui_state(self) -> UIState:
        """Get current UI state with parsed elements - FIXED VERSION"""
        
        if self.current_state and 'ui_state' in self.current_state:
            ui_state = self.current_state['ui_state']
            print(f"[ANDROID_ENV] Returning cached UI state with {len(ui_state.elements)} elements")
            return ui_state
        
        # Create fresh UI state if none exists
        print(f"[ANDROID_ENV] Creating fresh UI state")
        fresh_obs = self._create_mock_observation()
        return fresh_obs['ui_state']
    
    # ✅ ADD MISSING ACTION EXECUTION METHODS
    def touch(self, x: int, y: int) -> bool:
        """Execute touch action at coordinates"""
        print(f"[ANDROID_ENV] Touch action at ({x}, {y})")
        
        action = {
            "action_type": "touch",
            "touch_point": [x, y],
            "coordinate": [x, y]  # Alternative format
        }
        
        try:
            obs, reward, done, info = self.step(action)
            success = reward > 0 or info.get("success", False)
            print(f"[ANDROID_ENV] Touch {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] Touch failed: {e}")
            return False
    
    def touch_element(self, element_id: str) -> bool:
        """Execute touch action on UI element by ID"""
        print(f"[ANDROID_ENV] Touch element: {element_id}")
        
        action = {
            "action_type": "touch",
            "element_id": element_id
        }
        
        try:
            obs, reward, done, info = self.step(action)
            success = reward > 0 or info.get("success", False)
            print(f"[ANDROID_ENV] Touch element {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] Touch element failed: {e}")
            return False
    
    def scroll(self, direction: str = "down") -> bool:
        """Execute scroll action"""
        print(f"[ANDROID_ENV] Scroll {direction}")
        
        action = {
            "action_type": "scroll",
            "direction": direction
        }
        
        try:
            obs, reward, done, info = self.step(action)
            success = reward > 0 or info.get("success", False)
            print(f"[ANDROID_ENV] Scroll {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] Scroll failed: {e}")
            return False
    
    def swipe(self, start_coords: List[int], end_coords: List[int]) -> bool:
        """Execute swipe action"""
        print(f"[ANDROID_ENV] Swipe from {start_coords} to {end_coords}")
        
        action = {
            "action_type": "swipe",
            "start_coordinate": start_coords,
            "end_coordinate": end_coords
        }
        
        try:
            obs, reward, done, info = self.step(action)
            success = reward > 0 or info.get("success", False)
            print(f"[ANDROID_ENV] Swipe {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] Swipe failed: {e}")
            return False
    
    def type_text(self, text: str) -> bool:
        """Type text into focused element"""
        print(f"[ANDROID_ENV] Type text: '{text}'")
        
        action = {
            "action_type": "type",
            "text": text
        }
        
        try:
            obs, reward, done, info = self.step(action)
            success = reward > 0 or info.get("success", False)
            print(f"[ANDROID_ENV] Type {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] Type failed: {e}")
            return False
    
    def back(self) -> bool:
        """Press back button"""
        print(f"[ANDROID_ENV] Back button")
        
        action = {
            "action_type": "back"
        }
        
        try:
            obs, reward, done, info = self.step(action)
            success = reward > 0 or info.get("success", False)
            print(f"[ANDROID_ENV] Back {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] Back failed: {e}")
            return False
    
    def home(self) -> bool:
        """Press home button"""
        print(f"[ANDROID_ENV] Home button")
        
        action = {
            "action_type": "home"
        }
        
        try:
            obs, reward, done, info = self.step(action)
            success = reward > 0 or info.get("success", False)
            print(f"[ANDROID_ENV] Home {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] Home failed: {e}")
            return False
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render current screen"""
        if not self.mock_mode and hasattr(self.env, 'render'):
            try:
                return self.env.render(mode=mode)
            except Exception as e:
                logger.error(f"Render failed: {e}")
        
        # Return mock screenshot
        return self._create_mock_screenshot()
    
    def save_screenshot(self) -> Optional[str]:
        """Save current screenshot and return path"""
        try:
            screenshot = self.render()
            if screenshot is not None:
                filename = f"step_{self.step_count:03d}_{int(time.time())}.png"
                filepath = self.screenshot_dir / filename
                cv2.imwrite(str(filepath), cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
                print(f"[ANDROID_ENV] Screenshot saved: {filepath}")
                return str(filepath)
        except Exception as e:
            print(f"[ANDROID_ENV] Screenshot save failed: {e}")
        
        return None
    
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
            self.ui_state_cache = ui_state
        elif self.ui_state_cache:
            # Use cached UI state
            processed["ui_state"] = self.ui_state_cache
        else:
            # Create basic UI state
            processed["ui_state"] = UIState(elements=[], hierarchy={})
        
        return processed
    
    def _create_mock_observation(self) -> Dict[str, Any]:
        """Create enhanced mock observation for testing"""
        
        # Create more varied mock UI based on step count
        if self.step_count == 0:
            # Initial settings screen
            mock_ui_dump = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
            <hierarchy rotation="0">
                <node index="0" text="Settings" resource-id="android:id/title" class="android.widget.TextView" 
                      bounds="[100,100][300,150]" clickable="true" enabled="true" />
                <node index="1" text="Wi-Fi" resource-id="com.android.settings:id/wifi_settings" 
                      class="android.widget.TextView" bounds="[100,200][300,250]" clickable="true" enabled="true" />
                <node index="2" text="" resource-id="com.android.settings:id/wifi_switch" 
                      class="android.widget.Switch" bounds="[400,200][450,250]" 
                      clickable="true" enabled="true" checkable="true" checked="false" />
                <node index="3" text="Bluetooth" resource-id="com.android.settings:id/bluetooth_settings" 
                      class="android.widget.TextView" bounds="[100,300][300,350]" clickable="true" enabled="true" />
                <node index="4" text="" resource-id="com.android.settings:id/bluetooth_switch" 
                      class="android.widget.Switch" bounds="[400,300][450,350]" 
                      clickable="true" enabled="true" checkable="true" checked="false" />
            </hierarchy>"""
        else:
            # Modified state after actions
            wifi_checked = "true" if self.step_count % 2 == 1 else "false"
            mock_ui_dump = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
            <hierarchy rotation="0">
                <node index="0" text="Settings" resource-id="android:id/title" class="android.widget.TextView" 
                      bounds="[100,100][300,150]" clickable="true" enabled="true" />
                <node index="1" text="Wi-Fi" resource-id="com.android.settings:id/wifi_settings" 
                      class="android.widget.TextView" bounds="[100,200][300,250]" clickable="true" enabled="true" />
                <node index="2" text="" resource-id="com.android.settings:id/wifi_switch" 
                      class="android.widget.Switch" bounds="[400,200][450,250]" 
                      clickable="true" enabled="true" checkable="true" checked="{wifi_checked}" />
                <node index="3" text="Bluetooth" resource-id="com.android.settings:id/bluetooth_settings" 
                      class="android.widget.TextView" bounds="[100,300][300,350]" clickable="true" enabled="true" />
                <node index="4" text="" resource-id="com.android.settings:id/bluetooth_switch" 
                      class="android.widget.Switch" bounds="[400,300][450,350]" 
                      clickable="true" enabled="true" checkable="true" checked="false" />
            </hierarchy>"""
        
        ui_state = self.ui_parser.parse_ui_hierarchy(mock_ui_dump)
        self.ui_state_cache = ui_state
        
        return {
            "screenshot": self._create_mock_screenshot(),
            "ui_state": ui_state,
            "ui_dump": mock_ui_dump,
            "timestamp": time.time(),
            "step": self.step_count,
            "ui_elements": [  # Add this for compatibility
                {
                    "text": elem.text,
                    "content_desc": elem.content_desc,
                    "class_name": elem.class_name,
                    "bounds": elem.bounds,
                    "clickable": elem.clickable,
                    "enabled": elem.enabled,
                    "checked": elem.checked,
                    "checkable": elem.checkable
                }
                for elem in ui_state.elements
            ]
        }
    
    def _create_mock_screenshot(self) -> np.ndarray:
        """Create a simple mock screenshot"""
        # Create a simple colored screenshot for testing
        screenshot = np.full((1920, 1080, 3), 50, dtype=np.uint8)  # Dark background
        
        # Add some colored rectangles to simulate UI elements
        screenshot[100:150, 100:300] = [70, 130, 180]  # Blue rectangle for "Settings"
        screenshot[200:250, 100:300] = [60, 179, 113]  # Green rectangle for "Wi-Fi"
        screenshot[300:350, 100:300] = [135, 206, 235]  # Light blue for "Bluetooth"
        
        return screenshot
    
    def _mock_step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Enhanced mock step execution with realistic responses"""
        action_type = action.get("action_type", "unknown")
        print(f"[ANDROID_ENV] Mock step: {action_type}")
        
        # Simulate different success rates for different actions
        success_rates = {
            "touch": 0.9,
            "scroll": 0.95,
            "type": 0.85,
            "back": 0.98,
            "home": 0.98,
            "swipe": 0.8
        }
        
        success_rate = success_rates.get(action_type, 0.7)
        import random
        is_successful = random.random() < success_rate
        
        reward = 1.0 if is_successful else 0.0
        done = self.step_count >= 10  # End after 10 steps
        
        obs = self._create_mock_observation()
        self.current_state = obs
        
        info = {
            "success": is_successful, 
            "mock": True,
            "action_type": action_type,
            "step": self.step_count
        }
        
        print(f"[ANDROID_ENV] Mock step result: {'success' if is_successful else 'failed'}")
        return obs, reward, done, info
    
    def close(self):
        """Close environment"""
        print(f"[ANDROID_ENV] Closing environment")
        if not self.mock_mode and self.env:
            try:
                self.env.close()
            except Exception as e:
                logger.error(f"Failed to close AndroidEnv: {e}")
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get history of all actions taken"""
        return self.action_history.copy()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for debugging"""
        return {
            "mock_mode": self.mock_mode,
            "task_name": self.task_name,
            "step_count": self.step_count,
            "android_env_available": ANDROID_ENV_AVAILABLE,
            "current_state_available": self.current_state is not None,
            "ui_state_cached": self.ui_state_cache is not None,
            "action_history_length": len(self.action_history)
        }
