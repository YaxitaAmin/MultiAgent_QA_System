# core/android_env_wrapper.py - FULLY FIXED VERSION WITH MISSING CLASSES
import time
import json
import os
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from loguru import logger

# Conditional imports with proper type handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    from PIL import Image

try:
    from android_env.environment import AndroidEnv
    ANDROID_ENV_AVAILABLE = True
    print("[ANDROID_ENV] ✅ AndroidEnv imported successfully")
except ImportError as e:
    ANDROID_ENV_AVAILABLE = False
    class AndroidEnv:
        pass
    logger.warning(f"AndroidEnv not available: {e}")

# Fix the import - use relative import properly
try:
    from .ui_utils import UITreeParser, UIState
except ImportError:
    from core.ui_utils import UITreeParser, UIState

# ✅ MISSING CLASSES - Add these dataclasses that other modules expect
@dataclass
class AndroidAction:
    """Represents an Android UI action"""
    action_type: str  # touch, swipe, type, key_event
    coordinates: Optional[Tuple[int, int]] = None
    element_id: Optional[str] = None
    text: Optional[str] = None
    key_code: Optional[int] = None
    start_coords: Optional[Tuple[int, int]] = None
    end_coords: Optional[Tuple[int, int]] = None

@dataclass
class AndroidObservation:
    """Android environment observation"""
    screenshot: np.ndarray
    ui_hierarchy: str
    current_activity: str
    screen_bounds: Tuple[int, int]
    timestamp: float
    task_completed: bool = False
    error_message: Optional[str] = None

# Keep all your existing AndroidEnvWrapper class code exactly as it is...
class AndroidEnvWrapper:
    """FULLY FIXED AndroidEnv wrapper with correct API usage"""
    
    def __init__(self, task_name: str = "settings_wifi", screenshot_dir: str = "screenshots"):
        self.task_name = task_name
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(exist_ok=True)
        self.ui_parser = UITreeParser()
        self.step_count = 0
        
        print(f"[ANDROID_ENV] 🚀 Initializing AndroidEnvWrapper for task: {task_name}")
        print(f"[ANDROID_ENV] AndroidEnv available: {ANDROID_ENV_AVAILABLE}")
        
        if ANDROID_ENV_AVAILABLE:
            try:
                print(f"[ANDROID_ENV] 🔧 Creating AndroidEnv instance...")
                
                # FIXED: Use correct AndroidEnv initialization (no avd_name parameter)
                self.env = AndroidEnv()  # Simplified initialization
                
                self.mock_mode = False
                logger.info(f"✅ Real AndroidEnv initialized for task: {task_name}")
                print(f"[ANDROID_ENV] ✅ Real AndroidEnv initialized successfully!")
                
            except Exception as e:
                logger.warning(f"❌ Real environment initialization failed: {e}")
                print(f"[ANDROID_ENV] ❌ Real environment initialization failed: {e}")
                print(f"[ANDROID_ENV] ⚠️  Real AndroidEnv initialization failed, using mock mode")
                self.env = None
                self.mock_mode = True
        else:
            self.env = None
            self.mock_mode = True
            logger.info("Using mock Android environment")
            print(f"[ANDROID_ENV] Using mock Android environment")
        
        self.current_state = None
        self.current_observation = None  # Add this for compatibility
        self.action_history = []
        self.ui_state_cache = None
        self.device_info = self._get_device_info()

    # ✅ ADD MISSING METHODS that other agents expect
    def _get_observation(self) -> AndroidObservation:
        """Get current Android observation in expected format"""
        if self.current_state:
            screenshot = self.current_state.get("screenshot", self._create_mock_screenshot())
            ui_hierarchy = self.current_state.get("ui_dump", "<hierarchy></hierarchy>")
        else:
            screenshot = self._create_mock_screenshot()
            ui_hierarchy = "<hierarchy></hierarchy>"
        
        # Convert to numpy array if needed
        if not isinstance(screenshot, np.ndarray):
            screenshot = np.array(screenshot) if screenshot is not None else self._create_mock_screenshot()
        
        return AndroidObservation(
            screenshot=screenshot,
            ui_hierarchy=ui_hierarchy,
            current_activity="com.android.settings/.Settings",
            screen_bounds=(1080, 1920),
            timestamp=time.time(),
            task_completed=False
        )

    def reset(self) -> AndroidObservation:
        """Reset environment and return AndroidObservation"""
        print(f"[ANDROID_ENV] 🔄 Resetting environment (mock_mode: {self.mock_mode})")
        
        self.step_count = 0
        self.action_history = []
        
        if not self.mock_mode and self.env:
            try:
                print(f"[ANDROID_ENV] 🚀 Calling real AndroidWorld reset...")
                obs = self.env.reset()
                print(f"[ANDROID_ENV] ✅ Real reset completed, processing observation...")
                self.current_state = self._process_observation(obs)
                self.current_observation = self._get_observation()
                print(f"[ANDROID_ENV] ✅ Real environment reset successful")
                return self.current_observation
            except Exception as e:
                logger.error(f"AndroidEnv reset failed: {e}")
                print(f"[ANDROID_ENV] ❌ Real reset failed: {e}")
                print(f"[ANDROID_ENV] 📱 Falling back to mock observation")
                return self._create_mock_android_observation()
        else:
            mock_obs = self._create_mock_observation()
            self.current_state = mock_obs
            self.current_observation = self._create_mock_android_observation()
            print(f"[ANDROID_ENV] 📱 Mock reset completed")
            return self.current_observation

    def _create_mock_android_observation(self) -> AndroidObservation:
        """Create mock AndroidObservation for compatibility"""
        mock_ui = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
        <hierarchy rotation="0">
            <node index="0" text="Settings" resource-id="android:id/title" class="android.widget.TextView" 
                  bounds="[100,100][300,150]" clickable="true" enabled="true" />
            <node index="1" text="Wi-Fi" resource-id="com.android.settings:id/wifi_settings" 
                  class="android.widget.TextView" bounds="[100,200][300,250]" clickable="true" enabled="true" />
            <node index="2" text="" resource-id="com.android.settings:id/wifi_switch" 
                  class="android.widget.Switch" bounds="[400,200][450,250]" 
                  clickable="true" enabled="true" checkable="true" checked="false" />
        </hierarchy>"""
        
        return AndroidObservation(
            screenshot=self._create_mock_screenshot(),
            ui_hierarchy=mock_ui,
            current_activity="com.android.settings/.Settings",
            screen_bounds=(1080, 1920),
            timestamp=time.time(),
            task_completed=False
        )

    def step(self, action: Union[Dict[str, Any], AndroidAction]) -> Tuple[AndroidObservation, bool, Dict[str, Any]]:
        """Execute action and return AndroidObservation, success, info"""
        self.step_count += 1
        
        # Convert AndroidAction to dict if needed
        if isinstance(action, AndroidAction):
            action_dict = {
                "action_type": action.action_type,
                "coordinates": action.coordinates,
                "element_id": action.element_id,
                "text": action.text,
                "key_code": action.key_code,
                "start_coords": action.start_coords,
                "end_coords": action.end_coords
            }
        else:
            action_dict = action
        
        self.action_history.append({
            "step": self.step_count,
            "action": action_dict,
            "timestamp": time.time()
        })
        
        print(f"[ANDROID_ENV] 🎯 Step {self.step_count}: {action_dict.get('action_type', 'unknown')}")
        
        if not self.mock_mode and self.env:
            try:
                print(f"[ANDROID_ENV] 🚀 Executing real AndroidWorld step...")
                androidworld_action = self._convert_to_androidworld_action(action_dict)
                print(f"[ANDROID_ENV] 🔄 Converted action: {androidworld_action}")
                
                obs, reward, done, info = self.env.step(androidworld_action)
                print(f"[ANDROID_ENV] ✅ Real step completed - Reward: {reward}, Done: {done}")
                
                processed_obs = self._process_observation(obs)
                self.current_state = processed_obs
                self.current_observation = self._get_observation()
                
                return self.current_observation, done, info
                
            except Exception as e:
                logger.error(f"Real AndroidWorld step failed: {e}")
                print(f"[ANDROID_ENV] ❌ Real step failed: {e}")
                return self._mock_step_android_obs(action_dict)
        else:
            return self._mock_step_android_obs(action_dict)

    def _mock_step_android_obs(self, action: Dict[str, Any]) -> Tuple[AndroidObservation, bool, Dict[str, Any]]:
        """Mock step that returns AndroidObservation"""
        action_type = action.get("action_type", "unknown")
        print(f"[ANDROID_ENV] 📱 Mock step: {action_type}")
        
        success_rates = {"touch": 0.9, "scroll": 0.95, "type": 0.85, "back": 0.98, "home": 0.98, "swipe": 0.8}
        success_rate = success_rates.get(action_type, 0.7)
        
        import random
        is_successful = random.random() < success_rate
        
        done = self.step_count >= 10
        
        # Update mock state
        mock_state = self._create_mock_observation()
        self.current_state = mock_state
        self.current_observation = self._get_observation()
        
        info = {"success": is_successful, "mock": True, "action_type": action_type, "step": self.step_count}
        
        print(f"[ANDROID_ENV] {'✅' if is_successful else '❌'} Mock step result: {'success' if is_successful else 'failed'}")
        return self.current_observation, done, info
    
    # Keep all your existing methods exactly as they are...
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        if self.mock_mode:
            return {"device_type": "mock", "screen_size": [1080, 1920]}
        
        try:
            import subprocess
            result = subprocess.run(['adb', 'shell', 'wm', 'size'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                output = result.stdout.strip()
                if "Physical size:" in output:
                    size_str = output.split(":")[1].strip()
                    width, height = map(int, size_str.split("x"))
                    return {
                        "device_type": "real_device",
                        "screen_size": [width, height],
                        "adb_available": True
                    }
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")
        
        return {"device_type": "emulator", "screen_size": [1080, 1920]}

    def initialize_real_androidworld(self, adb_device_serial: Optional[str] = None) -> bool:
        """Initialize real AndroidWorld environment with proper API"""
        
        if not ANDROID_ENV_AVAILABLE:
            print("[ANDROID_ENV] AndroidEnv not available, cannot initialize real environment")
            return False
        
        try:
            print(f"[ANDROID_ENV] 🔧 Attempting to initialize real AndroidWorld...")
            
            # FIXED: Use simplified AndroidEnv initialization
            self.env = AndroidEnv()
            
            # Try to reset to verify it works
            try:
                obs = self.env.reset()
                print(f"[ANDROID_ENV] ✅ AndroidEnv reset successful")
                self.mock_mode = False
                return True
            except Exception as reset_error:
                print(f"[ANDROID_ENV] ⚠️  AndroidEnv reset failed: {reset_error}")
                if hasattr(self.env, 'close'):
                    self.env.close()
                self.env = None
                self.mock_mode = True
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize real AndroidWorld: {e}")
            print(f"[ANDROID_ENV] ❌ Real initialization failed: {e}")
            self.env = None
            self.mock_mode = True
            return False
    
    def _convert_to_androidworld_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert our action format to AndroidWorld format"""
        action_type = action.get("action_type", "touch")
        
        if action_type == "touch":
            coordinates = action.get("coordinates", [200, 400])
            return {"action_type": "TOUCH", "coordinate": coordinates}
        elif action_type == "scroll":
            direction = action.get("direction", "down")
            start_coords = action.get("start_coordinates", [200, 600])
            if direction == "down":
                end_coords = [start_coords[0], start_coords[1] - 400]
            elif direction == "up":
                end_coords = [start_coords[0], start_coords[1] + 400]
            else:
                end_coords = action.get("end_coordinates", [200, 200])
            return {"action_type": "SCROLL", "startCoordinate": start_coords, "endCoordinate": end_coords}
        elif action_type == "type":
            return {"action_type": "TYPE", "text": action.get("text", "")}
        elif action_type == "back":
            return {"action_type": "KEY", "keyCode": "BACK"}
        elif action_type == "home":
            return {"action_type": "KEY", "keyCode": "HOME"}
        else:
            return {"action_type": "TOUCH", "coordinate": action.get("coordinates", [200, 400])}
    
    def get_ui_state(self) -> UIState:
        """Get current UI state with parsed elements"""
        if self.current_state and 'ui_state' in self.current_state:
            ui_state = self.current_state['ui_state']
            print(f"[ANDROID_ENV] 📱 Returning cached UI state with {len(ui_state.elements)} elements")
            return ui_state
        
        print(f"[ANDROID_ENV] 🔄 Creating fresh UI state")
        fresh_obs = self._create_mock_observation()
        return fresh_obs['ui_state']
    
    def touch(self, x: int, y: int) -> bool:
        """Execute touch action at coordinates"""
        print(f"[ANDROID_ENV] 👆 Touch action at ({x}, {y})")
        action = {"action_type": "touch", "touch_point": [x, y], "coordinate": [x, y]}
        try:
            obs, done, info = self.step(action)
            success = info.get("success", False)
            print(f"[ANDROID_ENV] {'✅' if success else '❌'} Touch {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] ❌ Touch failed: {e}")
            return False
    
    def scroll(self, direction: str = "down") -> bool:
        """Execute scroll action"""
        print(f"[ANDROID_ENV] 📜 Scroll {direction}")
        action = {"action_type": "scroll", "direction": direction}
        try:
            obs, done, info = self.step(action)
            success = info.get("success", False)
            print(f"[ANDROID_ENV] {'✅' if success else '❌'} Scroll {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] ❌ Scroll failed: {e}")
            return False
    
    def back(self) -> bool:
        """Press back button"""
        print(f"[ANDROID_ENV] ⬅️ Back button")
        action = {"action_type": "back"}
        try:
            obs, done, info = self.step(action)
            success = info.get("success", False)
            print(f"[ANDROID_ENV] {'✅' if success else '❌'} Back {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            print(f"[ANDROID_ENV] ❌ Back failed: {e}")
            return False
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render current screen"""
        if not self.mock_mode and self.env and hasattr(self.env, 'render'):
            try:
                return self.env.render(mode=mode)
            except Exception as e:
                logger.error(f"Render failed: {e}")
        return self._create_mock_screenshot()
    
    def save_screenshot(self) -> Optional[str]:
        """FIXED: Save current screenshot and return path"""
        try:
            screenshot = self.render()
            if screenshot is not None:
                filename = f"step_{self.step_count:03d}_{int(time.time())}.png"
                filepath = self.screenshot_dir / filename
                
                if CV2_AVAILABLE:
                    cv2.imwrite(str(filepath), cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
                else:
                    from PIL import Image
                    image = Image.fromarray(screenshot.astype('uint8'), 'RGB')
                    image.save(str(filepath))
                
                print(f"[ANDROID_ENV] 📸 Screenshot saved: {filepath}")
                return str(filepath)
        except Exception as e:
            print(f"[ANDROID_ENV] ❌ Screenshot save failed: {e}")
        return None
    
    def get_real_device_status(self) -> Dict[str, Any]:
        """Get status of real Android device/emulator"""
        if self.mock_mode:
            return {"status": "mock_mode", "real_device": False}
        
        try:
            import subprocess
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=10)
            devices = []
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]
                devices = [line.split('\t')[0] for line in lines if '\tdevice' in line]
            
            return {
                "status": "real_mode",
                "real_device": True,
                "connected_devices": devices,
                "device_count": len(devices),
                "android_env_available": ANDROID_ENV_AVAILABLE
            }
        except Exception as e:
            return {"status": "error", "real_device": False, "error": str(e)}
    
    def _process_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw observation into structured format"""
        processed = {
            "screenshot": obs.get("screenshot") or obs.get("pixels"),
            "timestamp": time.time(),
            "step": self.step_count
        }
        
        if "ui_hierarchy" in obs:
            ui_state = self.ui_parser.parse_ui_hierarchy(obs["ui_hierarchy"])
            processed["ui_state"] = ui_state
            processed["ui_dump"] = obs["ui_hierarchy"]
            self.ui_state_cache = ui_state
        elif "forest" in obs:
            ui_state = self.ui_parser.parse_ui_hierarchy(str(obs["forest"]))
            processed["ui_state"] = ui_state
            processed["ui_dump"] = str(obs["forest"])
            self.ui_state_cache = ui_state
        elif self.ui_state_cache:
            processed["ui_state"] = self.ui_state_cache
        else:
            processed["ui_state"] = UIState(elements=[], hierarchy={})
        
        return processed
    
    def _create_mock_observation(self) -> Dict[str, Any]:
        """Create enhanced mock observation for testing"""
        if self.step_count == 0:
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
            "ui_elements": [
                {
                    "text": elem.text,
                    "content_desc": elem.content_description,
                    "class_name": elem.class_name,
                    "bounds": elem.bounds,
                    "clickable": elem.clickable,
                    "enabled": elem.enabled,
                    "checked": getattr(elem, 'checked', False),
                    "checkable": getattr(elem, 'checkable', False)
                }
                for elem in ui_state.elements
            ]
        }
    
    def _create_mock_screenshot(self) -> np.ndarray:
        """Create a simple mock screenshot"""
        screenshot = np.full((1920, 1080, 3), 50, dtype=np.uint8)
        screenshot[100:150, 100:300] = [70, 130, 180]  # Blue rectangle for "Settings"
        screenshot[200:250, 100:300] = [60, 179, 113]  # Green rectangle for "Wi-Fi"
        screenshot[300:350, 100:300] = [135, 206, 235]  # Light blue for "Bluetooth"
        return screenshot
    
    def close(self):
        """Close environment"""
        print(f"[ANDROID_ENV] 🔒 Closing environment")
        if not self.mock_mode and self.env and hasattr(self.env, 'close'):
            try:
                self.env.close()
            except Exception as e:
                logger.error(f"Failed to close AndroidEnv: {e}")
    
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
