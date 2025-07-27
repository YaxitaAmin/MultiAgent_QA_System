"""
Enhanced Virtual Device Environment with Pixel 5 Coordinate Mapping
Handles real Android device interaction with device-specific UI coordinates
"""

import time
from typing import Dict, Any, List, Tuple, Optional
from .adb_manager import ADBManager
from .observation_wrapper import ObservationWrapper

class VirtualDeviceEnv:
    """Enhanced virtual device environment with Pixel 5 optimization and fixed ADB handling"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.adb = ADBManager()
        self.last_observation = None
        
        # Pixel 5 specific properties
        self.screen_width = 1080
        self.screen_height = 2340
        self.density = 440
        self.device_type = "pixel_5"
        
        print(f"🔧 VirtualDeviceEnv initialized for Pixel 5: {device_id}")
        
    def reset(self) -> ObservationWrapper:
        """Reset the virtual device to home screen with proper error handling"""
        try:
            # Go to home screen
            self.adb.press_key(self.device_id, "KEYCODE_HOME")
            time.sleep(1)
            
            # Get current state
            observation = self.get_observation()
            self.last_observation = observation
            
            print(f"🔄 Device reset completed: {self.device_id}")
            return observation
            
        except Exception as e:
            print(f"❌ Reset failed: {e}")
            return self._create_fallback_observation()
    
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with Pixel 5 specific coordinate mapping and fixed error handling"""
        
        print(f"🔧 VirtualDeviceEnv executing: {action}")
        
        action_type = action.get("action", "unknown")
        
        try:
            if action_type == "touch":
                coordinates = action.get("coordinate", [540, 1170])
                result = self._execute_touch(coordinates)
                
            elif action_type == "swipe":
                start_coord = action.get("startCoordinate", [540, 1500])
                end_coord = action.get("endCoordinate", [540, 600])
                result = self._execute_swipe(start_coord, end_coord)
                
            elif action_type == "type":
                text = action.get("text", "")
                result = self._execute_type(text)
                
            elif action_type == "key":
                key = action.get("key", "KEYCODE_HOME")
                result = self._execute_key(key)
                
            elif action_type == "wait":
                duration = action.get("duration", 1)
                result = self._execute_wait(duration)
                
            else:
                print(f"⚠️ Unknown action type: {action_type}")
                result = self._create_success_result()
            
            # Add current observation
            result["current_observation"] = self.get_observation()
            return result
            
        except Exception as e:
            print(f"❌ Action execution failed: {e}")
            return self._create_error_result(str(e))
    
    def _execute_touch(self, coordinates: List[int]) -> Dict[str, Any]:
        """Execute touch action with Pixel 5 coordinate validation"""
        
        x, y = coordinates
        
        # Validate coordinates for Pixel 5 screen
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        
        print(f"🎯 Virtual device executing: touch")
        print(f"📱 Touch at ({x}, {y})")
        
        # Execute touch via ADB with proper error handling
        success = self.adb.tap(self.device_id, x, y)
        
        if success:
            time.sleep(0.5)  # Wait for UI response
        
        return {
            "success": success,
            "action": "touch",
            "coordinates": [x, y],
            "device_id": self.device_id,
            "pixel_5_validated": True,
            "timestamp": time.time()
        }
    
    def _execute_swipe(self, start_coord: List[int], end_coord: List[int]) -> Dict[str, Any]:
        """Execute swipe action with Pixel 5 optimization"""
        
        x1, y1 = start_coord
        x2, y2 = end_coord
        
        # Validate coordinates
        x1 = max(0, min(x1, self.screen_width - 1))
        y1 = max(0, min(y1, self.screen_height - 1))
        x2 = max(0, min(x2, self.screen_width - 1))
        y2 = max(0, min(y2, self.screen_height - 1))
        
        print(f"🎯 Virtual device executing: swipe")
        print(f"📱 Swipe from ({x1}, {y1}) to ({x2}, {y2})")
        
        # Execute swipe via ADB with appropriate duration
        duration = 500  # milliseconds
        success = self.adb.swipe(self.device_id, x1, y1, x2, y2, duration)
        
        if success:
            time.sleep(1)  # Wait for swipe animation
        
        return {
            "success": success,
            "action": "swipe",
            "start_coordinates": [x1, y1],
            "end_coordinates": [x2, y2],
            "device_id": self.device_id,
            "pixel_5_validated": True,
            "timestamp": time.time()
        }
    
    def _execute_type(self, text: str) -> Dict[str, Any]:
        """Execute text input action"""
        
        print(f"🎯 Virtual device executing: type")
        print(f"⌨️ Typing: '{text}'")
        
        success = False
        
        if text:
            success = self.adb.type_text(self.device_id, text)
            if success:
                time.sleep(0.5)
        else:
            print("⚠️ No text to type")
            success = True  # Consider empty text as success
        
        return {
            "success": success,
            "action": "type",
            "text": text,
            "device_id": self.device_id,
            "timestamp": time.time()
        }
    
    def _execute_key(self, key: str) -> Dict[str, Any]:
        """Execute hardware key press"""
        
        print(f"🎯 Virtual device executing: key")
        print(f"🔑 Key: {key}")
        
        success = self.adb.press_key(self.device_id, key)
        
        if success:
            time.sleep(0.5)
        
        return {
            "success": success,
            "action": "key",
            "key": key,
            "device_id": self.device_id,
            "timestamp": time.time()
        }
    
    def _execute_wait(self, duration: float) -> Dict[str, Any]:
        """Execute wait action"""
        
        print(f"🎯 Virtual device executing: wait")
        print(f"⏳ Duration: {duration}s")
        
        time.sleep(duration)
        
        return {
            "success": True,
            "action": "wait",
            "duration": duration,
            "device_id": self.device_id,
            "timestamp": time.time()
        }
    
    def get_observation(self) -> ObservationWrapper:
        """Get current device observation with proper error handling"""
        
        try:
            # Capture screenshot
            screenshot = self.adb.get_screenshot(self.device_id)
            
            # Get UI hierarchy
            ui_hierarchy = self.adb.get_ui_hierarchy(self.device_id)
            
            # Get current activity
            current_activity = self.adb.get_current_activity(self.device_id)
            
            # Create observation
            observation_data = {
                "screenshot": screenshot,
                "ui_hierarchy": ui_hierarchy,
                "current_activity": current_activity,
                "screen_bounds": [self.screen_width, self.screen_height],
                "timestamp": time.time(),
                "device_id": self.device_id,
                "device_type": self.device_type,
                "screen_specs": f"{self.screen_width}x{self.screen_height}_density{self.density}"
            }
            
            observation = ObservationWrapper(observation_data)
            self.last_observation = observation
            
            return observation
            
        except Exception as e:
            print(f"❌ Failed to get observation: {e}")
            return self._create_fallback_observation()
    
    def _create_success_result(self) -> Dict[str, Any]:
        """Create a success result"""
        return {
            "success": True,
            "device_id": self.device_id,
            "timestamp": time.time()
        }
    
    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create an error result"""
        return {
            "success": False,
            "error": error,
            "device_id": self.device_id,
            "timestamp": time.time(),
            "current_observation": self.get_observation()
        }
    
    def _create_fallback_observation(self) -> ObservationWrapper:
        """Create fallback observation when real observation fails"""
        
        fallback_data = {
            "screenshot": b"",  # Empty screenshot
            "ui_hierarchy": "",
            "current_activity": "unknown",
            "screen_bounds": [self.screen_width, self.screen_height],
            "timestamp": time.time(),
            "device_id": self.device_id,
            "device_type": self.device_type,
            "fallback": True
        }
        
        return ObservationWrapper(fallback_data)
    
    def get_pixel5_coordinates(self, ui_element: str) -> List[int]:
        """Get Pixel 5 specific coordinates for common UI elements"""
        
        pixel5_coordinates = {
            # App icons (assuming 5x6 grid)
            "settings": [270, 1200],      # Settings app icon
            "calculator": [540, 1400],    # Calculator app icon  
            "phone": [810, 1200],         # Phone app icon
            "camera": [270, 1000],        # Camera app icon
            "gallery": [540, 1000],       # Gallery app icon
            
            # Settings screen elements
            "wifi_option": [150, 400],     # WiFi option in settings
            "wifi_toggle": [1000, 400],    # WiFi toggle switch
            "bluetooth_option": [150, 500], # Bluetooth option
            "display_option": [150, 600],   # Display option
            "network_option": [150, 300],   # Network & internet option
            
            # Quick settings (notification panel pulled down twice)
            "wifi_qs": [200, 300],         # WiFi quick setting
            "airplane_qs": [400, 300],     # Airplane mode quick setting
            "bluetooth_qs": [600, 300],    # Bluetooth quick setting
            "flashlight_qs": [800, 300],   # Flashlight quick setting
            "location_qs": [200, 400],     # Location quick setting
            "rotation_qs": [400, 400],     # Auto-rotate quick setting
            
            # Common areas
            "center": [540, 1170],         # True center of screen
            "top_center": [540, 200],      # Top center for notifications
            "bottom_center": [540, 2100],  # Bottom center for navigation
            
            # Navigation gestures (Android 10+)
            "back_gesture": [100, 1170],   # Left edge for back gesture
            "home_gesture": [540, 2280],   # Bottom center for home gesture
            "recent_gesture": [980, 2250], # Right bottom for recent apps gesture
            
            # Notification panel
            "notification_pull": [540, 50], # Top center to pull notifications
            "quick_settings_pull": [540, 100], # Slightly lower for quick settings
            
            # Common UI patterns
            "menu_button": [980, 100],     # Top right menu (3 dots)
            "search_bar": [540, 150],      # Search bar area
            "fab_button": [900, 1800],     # Floating action button area
        }
        
        return pixel5_coordinates.get(ui_element, [540, 1170])  # Default to center
    
    def perform_smart_action(self, action_name: str, **kwargs) -> Dict[str, Any]:
        """Perform smart actions using predefined sequences"""
        
        smart_actions = {
            "open_settings": self._smart_open_settings,
            "toggle_wifi": self._smart_toggle_wifi,
            "toggle_airplane_mode": self._smart_toggle_airplane_mode,
            "open_calculator": self._smart_open_calculator,
            "go_home": self._smart_go_home,
            "open_notification_panel": self._smart_open_notification_panel,
            "close_notification_panel": self._smart_close_notification_panel,
        }
        
        action_func = smart_actions.get(action_name)
        if action_func:
            return action_func(**kwargs)
        else:
            return {"success": False, "error": f"Unknown smart action: {action_name}"}
    
    def _smart_open_settings(self, **kwargs) -> Dict[str, Any]:
        """Smart action to open settings app"""
        
        print("🔧 Smart action: Opening settings")
        
        # Method 1: Direct tap on settings icon
        settings_coords = self.get_pixel5_coordinates("settings")
        touch_action = {
            "action": "touch",
            "coordinate": settings_coords
        }
        
        result = self.execute_action(touch_action)
        
        if result.get("success"):
            time.sleep(2)
            # Verify settings opened by checking for settings text
            observation = self.get_observation()
            settings_elements = observation.find_elements_by_text("settings")
            
            if settings_elements:
                result["verified"] = True
                print("✅ Settings app opened successfully")
            else:
                result["verified"] = False
                print("⚠️ Settings app may not have opened")
        
        return result
    
    def _smart_toggle_wifi(self, **kwargs) -> Dict[str, Any]:
        """Smart action to toggle WiFi"""
        
        print("🔧 Smart action: Toggling WiFi")
        
        # First open settings
        settings_result = self._smart_open_settings()
        if not settings_result.get("success"):
            return settings_result
        
        time.sleep(1)
        
        # Tap on WiFi option
        wifi_coords = self.get_pixel5_coordinates("wifi_option")
        wifi_action = {
            "action": "touch",
            "coordinate": wifi_coords
        }
        
        wifi_result = self.execute_action(wifi_action)
        
        if wifi_result.get("success"):
            time.sleep(2)
            
            # Tap on WiFi toggle
            toggle_coords = self.get_pixel5_coordinates("wifi_toggle")
            toggle_action = {
                "action": "touch",
                "coordinate": toggle_coords
            }
            
            toggle_result = self.execute_action(toggle_action)
            return toggle_result
        
        return wifi_result
    
    def _smart_toggle_airplane_mode(self, **kwargs) -> Dict[str, Any]:
        """Smart action to toggle airplane mode"""
        
        print("🔧 Smart action: Toggling airplane mode")
        
        # Pull down notification panel
        notification_result = self._smart_open_notification_panel()
        if not notification_result.get("success"):
            return notification_result
        
        time.sleep(1)
        
        # Pull down again for quick settings
        quick_settings_action = {
            "action": "swipe",
            "startCoordinate": [540, 100],
            "endCoordinate": [540, 800]
        }
        
        qs_result = self.execute_action(quick_settings_action)
        
        if qs_result.get("success"):
            time.sleep(1)
            
            # Tap airplane mode toggle
            airplane_coords = self.get_pixel5_coordinates("airplane_qs")
            airplane_action = {
                "action": "touch",
                "coordinate": airplane_coords
            }
            
            return self.execute_action(airplane_action)
        
        return qs_result
    
    def _smart_open_calculator(self, **kwargs) -> Dict[str, Any]:
        """Smart action to open calculator app"""
        
        print("🔧 Smart action: Opening calculator")
        
        calc_coords = self.get_pixel5_coordinates("calculator")
        calc_action = {
            "action": "touch",
            "coordinate": calc_coords
        }
        
        return self.execute_action(calc_action)
    
    def _smart_go_home(self, **kwargs) -> Dict[str, Any]:
        """Smart action to go to home screen"""
        
        print("🔧 Smart action: Going home")
        
        home_action = {
            "action": "key",
            "key": "KEYCODE_HOME"
        }
        
        return self.execute_action(home_action)
    
    def _smart_open_notification_panel(self, **kwargs) -> Dict[str, Any]:
        """Smart action to open notification panel"""
        
        print("🔧 Smart action: Opening notification panel")
        
        notification_action = {
            "action": "swipe",
            "startCoordinate": [540, 50],
            "endCoordinate": [540, 800]
        }
        
        return self.execute_action(notification_action)
    
    def _smart_close_notification_panel(self, **kwargs) -> Dict[str, Any]:
        """Smart action to close notification panel"""
        
        print("🔧 Smart action: Closing notification panel")
        
        close_action = {
            "action": "swipe",
            "startCoordinate": [540, 800],
            "endCoordinate": [540, 50]
        }
        
        return self.execute_action(close_action)
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get comprehensive device status"""
        
        try:
            device_info = self.adb.get_device_info(self.device_id)
            
            status = {
                "device_id": self.device_id,
                "connected": self.device_id in self.adb.get_connected_devices(),
                "device_info": device_info,
                "screen_config": {
                    "width": self.screen_width,
                    "height": self.screen_height,
                    "density": self.density,
                    "type": self.device_type
                },
                "last_observation_time": self.last_observation.timestamp if self.last_observation else None,
                "capabilities": {
                    "touch": True,
                    "swipe": True,
                    "type": True,
                    "keys": True,
                    "screenshot": True,
                    "ui_hierarchy": True
                }
            }
            
            return status
            
        except Exception as e:
            return {
                "device_id": self.device_id,
                "connected": False,
                "error": str(e)
            }
    
    def close(self):
        """Clean up resources"""
        print(f"🔧 VirtualDeviceEnv closed for {self.device_id}")
        
        # Perform any cleanup if needed
        self.last_observation = None
