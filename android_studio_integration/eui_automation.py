# Real UI interaction via ADB
"""
Enhanced UI Automation for Virtual Devices
Provides intelligent UI interaction methods
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import re


class UIElement:
    """Represents a UI element from hierarchy"""
    
    def __init__(self, element: ET.Element):
        self.element = element
        self.text = element.get('text', '')
        self.content_desc = element.get('content-desc', '')
        self.resource_id = element.get('resource-id', '')
        self.class_name = element.get('class', '')
        self.package = element.get('package', '')
        self.clickable = element.get('clickable', 'false').lower() == 'true'
        self.enabled = element.get('enabled', 'false').lower() == 'true'
        self.bounds = self._parse_bounds(element.get('bounds', ''))
        
    def _parse_bounds(self, bounds_str: str) -> Tuple[int, int, int, int]:
        """Parse bounds string like '[0,0][100,50]' to (x1,y1,x2,y2)"""
        try:
            pattern = r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]'
            match = re.match(pattern, bounds_str)
            if match:
                return tuple(map(int, match.groups()))
        except:
            pass
        return (0, 0, 0, 0)
    
    def get_center(self) -> Tuple[int, int]:
        """Get center coordinates of element"""
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def contains_text(self, text: str) -> bool:
        """Check if element contains specific text"""
        text_lower = text.lower()
        return (text_lower in self.text.lower() or 
                text_lower in self.content_desc.lower())


class UIAutomation:
    """Advanced UI automation for virtual devices"""
    
    def __init__(self, adb_manager):
        self.adb = adb_manager
    
    def parse_ui_hierarchy(self, ui_xml: str) -> List[UIElement]:
        """Parse UI hierarchy XML into UIElement objects"""
        if not ui_xml:
            return []
        
        try:
            root = ET.fromstring(ui_xml)
            elements = []
            
            def extract_elements(node):
                elements.append(UIElement(node))
                for child in node:
                    extract_elements(child)
            
            extract_elements(root)
            return elements
            
        except Exception as e:
            print(f"Error parsing UI hierarchy: {e}")
            return []
    
    def find_element_by_text(self, device_id: str, text: str) -> Optional[UIElement]:
        """Find UI element containing specific text"""
        ui_xml = self.adb.get_ui_hierarchy(device_id)
        elements = self.parse_ui_hierarchy(ui_xml)
        
        for element in elements:
            if element.contains_text(text) and element.clickable:
                return element
        
        return None
    
    def find_element_by_id(self, device_id: str, resource_id: str) -> Optional[UIElement]:
        """Find UI element by resource ID"""
        ui_xml = self.adb.get_ui_hierarchy(device_id)
        elements = self.parse_ui_hierarchy(ui_xml)
        
        for element in elements:
            if resource_id in element.resource_id and element.clickable:
                return element
        
        return None
    
    def tap_element_by_text(self, device_id: str, text: str) -> bool:
        """Tap element containing specific text"""
        element = self.find_element_by_text(device_id, text)
        if element:
            x, y = element.get_center()
            return self.adb.tap(device_id, x, y)
        return False
    
    def tap_element_by_id(self, device_id: str, resource_id: str) -> bool:
        """Tap element by resource ID"""
        element = self.find_element_by_id(device_id, resource_id)
        if element:
            x, y = element.get_center()
            return self.adb.tap(device_id, x, y)
        return False
    
    def scroll(self, device_id: str, direction: str = "down") -> bool:
        """Intelligent scrolling based on screen content"""
        # Get screen bounds (standard Android resolution)
        screen_width, screen_height = 1080, 1920
        
        # Calculate scroll coordinates
        center_x = screen_width // 2
        start_y = int(screen_height * 0.8)
        end_y = int(screen_height * 0.2)
        
        if direction.lower() == "up":
            start_y, end_y = end_y, start_y
        elif direction.lower() == "left":
            return self.adb.swipe(device_id, int(screen_width * 0.8), center_x // 2, 
                                int(screen_width * 0.2), center_x // 2)
        elif direction.lower() == "right":
            return self.adb.swipe(device_id, int(screen_width * 0.2), center_x // 2,
                                int(screen_width * 0.8), center_x // 2)
        
        return self.adb.swipe(device_id, center_x, start_y, center_x, end_y)
    
    def get_clickable_elements(self, device_id: str) -> List[UIElement]:
        """Get all clickable elements on current screen"""
        ui_xml = self.adb.get_ui_hierarchy(device_id)
        elements = self.parse_ui_hierarchy(ui_xml)
        
        return [elem for elem in elements if elem.clickable and elem.enabled]
    
    def get_text_elements(self, device_id: str) -> List[str]:
        """Get all text content from current screen"""
        ui_xml = self.adb.get_ui_hierarchy(device_id)
        elements = self.parse_ui_hierarchy(ui_xml)
        
        texts = []
        for element in elements:
            if element.text:
                texts.append(element.text)
            if element.content_desc:
                texts.append(element.content_desc)
        
        return [text for text in texts if text.strip()]
"""
Enhanced UI Automation for Android devices
Provides high-level UI interaction methods
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from .adb_manager import ADBManager
from .observation_wrapper import ObservationWrapper
from .screen_capture import ScreenCapture

class EUIAutomation:
    """Enhanced UI Automation with intelligent interaction methods"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.adb = ADBManager()
        self.screen_capture = ScreenCapture(device_id)
        self.pixel5_config = {
            'width': 1080,
            'height': 2340,
            'safe_area': {'top': 100, 'bottom': 2200, 'left': 50, 'right': 1030}
        }
    
    def smart_tap(self, target: str, observation: ObservationWrapper = None) -> bool:
        """Smart tap that finds target by text, description, or coordinates"""
        
        if not observation:
            # Get current observation if not provided
            from .virtual_device_env import VirtualDeviceEnv
            env = VirtualDeviceEnv(self.device_id)
            observation = env.get_observation()
        
        # Try different matching strategies
        coordinates = None
        
        # Strategy 1: Find by text
        elements = observation.find_elements_by_text(target)
        if elements:
            element = elements[0]
            coordinates = element['center']
            print(f"🎯 Found element by text: {target} at {coordinates}")
        
        # Strategy 2: Find by content description
        if not coordinates:
            elements = [e for e in observation.ui_elements 
                       if target.lower() in e.get('content_desc', '').lower()]
            if elements:
                element = elements[0]
                coordinates = element['center']
                print(f"🎯 Found element by description: {target} at {coordinates}")
        
        # Strategy 3: Use predefined Pixel 5 coordinates
        if not coordinates:
            coordinates = observation.get_pixel5_optimized_coordinates(target.lower())
            print(f"🎯 Using Pixel 5 coordinates for {target}: {coordinates}")
        
        # Strategy 4: Parse as coordinates if format matches
        if not coordinates and isinstance(target, str):
            try:
                if ',' in target:
                    x, y = map(int, target.split(','))
                    coordinates = [x, y]
                    print(f"🎯 Using parsed coordinates: {coordinates}")
            except:
                pass
        
        # Execute tap
        if coordinates:
            return self.tap_at(coordinates[0], coordinates[1])
        else:
            print(f"❌ Could not find target: {target}")
            return False
    
    def tap_at(self, x: int, y: int) -> bool:
        """Tap at specific coordinates with validation"""
        
        # Validate coordinates are within screen bounds
        x = max(0, min(x, self.pixel5_config['width']))
        y = max(0, min(y, self.pixel5_config['height']))
        
        print(f"👆 Tapping at ({x}, {y})")
        return self.adb.tap(self.device_id, x, y)
    
    def smart_swipe(self, direction: str, distance: str = "medium") -> bool:
        """Smart swipe with direction and distance"""
        
        # Define swipe distances
        distances = {
            'short': 200,
            'medium': 400,
            'long': 600
        }
        
        swipe_distance = distances.get(distance, 400)
        center_x = self.pixel5_config['width'] // 2
        center_y = self.pixel5_config['height'] // 2
        
        # Calculate swipe coordinates based on direction
        swipe_coords = {
            'up': (center_x, center_y + swipe_distance//2, center_x, center_y - swipe_distance//2),
            'down': (center_x, center_y - swipe_distance//2, center_x, center_y + swipe_distance//2),
            'left': (center_x + swipe_distance//2, center_y, center_x - swipe_distance//2, center_y),
            'right': (center_x - swipe_distance//2, center_y, center_x + swipe_distance//2, center_y)
        }
        
        if direction not in swipe_coords:
            print(f"❌ Invalid swipe direction: {direction}")
            return False
        
        x1, y1, x2, y2 = swipe_coords[direction]
        print(f"👆 Swiping {direction} from ({x1}, {y1}) to ({x2}, {y2})")
        
        return self.adb.swipe(self.device_id, x1, y1, x2, y2)
    
    def scroll_to_find_text(self, target_text: str, max_scrolls: int = 5) -> bool:
        """Scroll to find specific text on screen"""
        
        from .virtual_device_env import VirtualDeviceEnv
        env = VirtualDeviceEnv(self.device_id)
        
        for scroll_attempt in range(max_scrolls):
            # Get current observation
            observation = env.get_observation()
            
            # Check if text is visible
            elements = observation.find_elements_by_text(target_text)
            if elements:
                print(f"✅ Found text '{target_text}' after {scroll_attempt} scrolls")
                return True
            
            # Scroll down to find more content
            if not self.smart_swipe('up', 'medium'):  # Swipe up to scroll down
                break
            
            time.sleep(1)  # Wait for scroll animation
        
        print(f"❌ Text '{target_text}' not found after {max_scrolls} scrolls")
        return False
    
    def wait_for_element(self, target: str, timeout: int = 10) -> bool:
        """Wait for element to appear on screen"""
        
        from .virtual_device_env import VirtualDeviceEnv
        env = VirtualDeviceEnv(self.device_id)
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            observation = env.get_observation()
            
            # Check if element exists
            elements = observation.find_elements_by_text(target)
            if elements:
                print(f"✅ Element '{target}' appeared after {time.time() - start_time:.1f}s")
                return True
            
            time.sleep(0.5)
        
        print(f"❌ Element '{target}' did not appear within {timeout}s")
        return False
    
    def wait_for_screen_change(self, timeout: int = 5, threshold: float = 5.0) -> bool:
        """Wait for screen to change significantly"""
        
        # Capture initial screenshot
        initial_image = self.screen_capture.capture_screenshot()
        if not initial_image:
            return False
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            time.sleep(0.5)
            
            # Capture current screenshot
            current_image = self.screen_capture.capture_screenshot()
            if not current_image:
                continue
            
            # Check if screen has changed
            if self.screen_capture.detect_ui_changes(initial_image, current_image, threshold):
                print(f"✅ Screen changed after {time.time() - start_time:.1f}s")
                return True
        
        print(f"❌ Screen did not change within {timeout}s")
        return False
    
    def smart_type_text(self, text: str, clear_first: bool = True) -> bool:
        """Smart text input with field clearing"""
        
        if clear_first:
            # Try to clear existing text
            self.adb.press_key(self.device_id, "KEYCODE_CTRL_A")
            time.sleep(0.2)
        
        success = self.adb.type_text(self.device_id, text)
        
        if success:
            print(f"⌨️ Typed text: '{text}'")
        else:
            print(f"❌ Failed to type text: '{text}'")
        
        return success
    
    def navigate_to_settings(self) -> bool:
        """Navigate to device settings"""
        
        # Method 1: Try to tap settings app
        if self.smart_tap("settings"):
            return self.wait_for_element("Settings", timeout=5)
        
        # Method 2: Use quick settings
        if self.pull_down_notification_panel():
            time.sleep(1)
            if self.smart_tap("settings"):
                return self.wait_for_element("Settings", timeout=5)
        
        # Method 3: Use intent
        result = self.adb.execute_command(
            self.device_id,
            ["shell", "am", "start", "-a", "android.settings.SETTINGS"]
        )
        
        return not result.startswith("Command")
    
    def pull_down_notification_panel(self) -> bool:
        """Pull down notification panel"""
        
        # Swipe from top of screen
        return self.adb.swipe(
            self.device_id,
            self.pixel5_config['width'] // 2,
            50,
            self.pixel5_config['width'] // 2,
            self.pixel5_config['height'] // 2
        )
    
    def go_home(self) -> bool:
        """Go to home screen"""
        
        return self.adb.press_key(self.device_id, "KEYCODE_HOME")
    
    def go_back(self) -> bool:
        """Go back"""
        
        return self.adb.press_key(self.device_id, "KEYCODE_BACK")
    
    def open_recent_apps(self) -> bool:
        """Open recent apps"""
        
        return self.adb.press_key(self.device_id, "KEYCODE_APP_SWITCH")
    
    def toggle_wifi(self) -> bool:
        """Toggle WiFi on/off"""
        
        # Navigate to WiFi settings
        if not self.navigate_to_settings():
            return False
        
        # Find and tap WiFi option
        if not self.smart_tap("wifi"):
            return False
        
        time.sleep(2)
        
        # Find and tap WiFi toggle
        return self.smart_tap("wifi_toggle")
    
    def toggle_airplane_mode(self) -> bool:
        """Toggle airplane mode on/off"""
        
        # Pull down notification panel
        if not self.pull_down_notification_panel():
            return False
        
        time.sleep(1)
        
        # Pull down again for quick settings
        if not self.pull_down_notification_panel():
            return False
        
        time.sleep(1)
        
        # Tap airplane mode toggle
        return self.smart_tap("airplane_mode")
    
    def launch_app(self, app_name: str) -> bool:
        """Launch app by name"""
        
        # Go to home screen first
        self.go_home()
        time.sleep(1)
        
        # Try to find and tap app icon
        if self.smart_tap(app_name.lower()):
            return self.wait_for_screen_change(timeout=10)
        
        # Fallback: Use app drawer
        # Swipe up to open app drawer
        self.smart_swipe('up', 'long')
        time.sleep(2)
        
        # Try to find app in drawer
        if self.scroll_to_find_text(app_name):
            return self.smart_tap(app_name)
        
        return False
    
    def capture_ui_state(self) -> Dict[str, Any]:
        """Capture comprehensive UI state"""
        
        from .virtual_device_env import VirtualDeviceEnv
        env = VirtualDeviceEnv(self.device_id)
        observation = env.get_observation()
        
        # Capture screenshot
        screenshot = self.screen_capture.capture_screenshot()
        
        state = {
            'timestamp': time.time(),
            'observation': observation.to_dict(),
            'screenshot_captured': screenshot is not None,
            'screen_analysis': observation.analyze_screen_content(),
            'clickable_elements': len(observation.get_clickable_elements()),
            'current_activity': observation.current_activity
        }
        
        return state
    
    def perform_gesture_sequence(self, gestures: List[Dict[str, Any]]) -> bool:
        """Perform sequence of gestures"""
        
        success = True
        
        for i, gesture in enumerate(gestures):
            gesture_type = gesture.get('type', 'tap')
            
            print(f"🎯 Performing gesture {i+1}/{len(gestures)}: {gesture_type}")
            
            if gesture_type == 'tap':
                target = gesture.get('target', '')
                if not self.smart_tap(target):
                    success = False
            
            elif gesture_type == 'swipe':
                direction = gesture.get('direction', 'up')
                distance = gesture.get('distance', 'medium')
                if not self.smart_swipe(direction, distance):
                    success = False
            
            elif gesture_type == 'type':
                text = gesture.get('text', '')
                if not self.smart_type_text(text):
                    success = False
            
            elif gesture_type == 'wait':
                duration = gesture.get('duration', 1)
                time.sleep(duration)
            
            elif gesture_type == 'key':
                key = gesture.get('key', 'KEYCODE_HOME')
                if not self.adb.press_key(self.device_id, key):
                    success = False
            
            # Wait between gestures
            delay = gesture.get('delay', 1)
            if delay > 0:
                time.sleep(delay)
        
        return success
    
    def validate_ui_state(self, expected_elements: List[str]) -> Dict[str, bool]:
        """Validate that expected UI elements are present"""
        
        from .virtual_device_env import VirtualDeviceEnv
        env = VirtualDeviceEnv(self.device_id)
        observation = env.get_observation()
        
        validation_results = {}
        
        for element in expected_elements:
            found_elements = observation.find_elements_by_text(element)
            validation_results[element] = len(found_elements) > 0
            
            if validation_results[element]:
                print(f"✅ Found expected element: {element}")
            else:
                print(f"❌ Missing expected element: {element}")
        
        return validation_results
