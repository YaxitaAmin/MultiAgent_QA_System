"""
Observation Wrapper for Android Device State
Provides structured interface for device observations
"""

import time
import base64
from typing import Dict, Any, Optional, List, Union
import json

class ObservationWrapper:
    """Enhanced observation wrapper with Pixel 5 optimizations"""
    
    def __init__(self, observation_data: Dict[str, Any]):
        self.raw_data = observation_data
        self.timestamp = observation_data.get('timestamp', time.time())
        self._parse_observation_data()
    
    def _parse_observation_data(self):
        """Parse and structure observation data"""
        
        # Screenshot data
        self.screenshot = self.raw_data.get('screenshot', b'')
        self.screenshot_size = len(self.screenshot) if self.screenshot else 0
        
        # UI hierarchy
        self.ui_hierarchy = self.raw_data.get('ui_hierarchy', '')
        self.ui_elements = self._parse_ui_hierarchy()
        
        # Device information
        self.device_info = self.raw_data.get('device_info', {})
        self.device_id = self.device_info.get('device_id', 'unknown')
        self.device_type = self.device_info.get('type', 'unknown')
        
        # Screen information
        self.screen_bounds = self.raw_data.get('screen_bounds', [1080, 2340])
        self.screen_width = self.screen_bounds[0] if len(self.screen_bounds) >= 2 else 1080
        self.screen_height = self.screen_bounds[1] if len(self.screen_bounds) >= 2 else 2340
        
        # Activity information
        self.current_activity = self.raw_data.get('current_activity', 'unknown')
        
        # Task information
        self.task_name = self.raw_data.get('task_name', 'unknown')
        
        # Additional metadata
        self.metadata = {
            'pixel_5_optimized': self._is_pixel5_device(),
            'screen_specs': f"{self.screen_width}x{self.screen_height}",
            'ui_elements_count': len(self.ui_elements),
            'has_screenshot': self.screenshot_size > 0,
            'observation_timestamp': self.timestamp
        }
    
    def _parse_ui_hierarchy(self) -> List[Dict[str, Any]]:
        """Parse UI hierarchy XML into structured elements"""
        
        elements = []
        
        if not self.ui_hierarchy:
            return elements
        
        try:
            # Simple XML parsing for UI elements
            import re
            
            # Extract node elements with bounds and attributes
            node_pattern = r'<node[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"[^>]*>'
            matches = re.finditer(node_pattern, self.ui_hierarchy)
            
            for match in matches:
                x1, y1, x2, y2 = map(int, match.groups())
                
                element = {
                    'bounds': [[x1, y1], [x2, y2]],
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1)
                }
                
                # Extract additional attributes
                node_text = match.group(0)
                
                # Text content
                text_match = re.search(r'text="([^"]*)"', node_text)
                element['text'] = text_match.group(1) if text_match else ''
                
                # Content description
                desc_match = re.search(r'content-desc="([^"]*)"', node_text)
                element['content_desc'] = desc_match.group(1) if desc_match else ''
                
                # Class name
                class_match = re.search(r'class="([^"]*)"', node_text)
                element['class'] = class_match.group(1) if class_match else ''
                
                # Clickable
                clickable_match = re.search(r'clickable="([^"]*)"', node_text)
                element['clickable'] = clickable_match.group(1) == 'true' if clickable_match else False
                
                # Enabled
                enabled_match = re.search(r'enabled="([^"]*)"', node_text)
                element['enabled'] = enabled_match.group(1) == 'true' if enabled_match else True
                
                elements.append(element)
        
        except Exception as e:
            print(f"⚠️ UI hierarchy parsing error: {e}")
        
        return elements
    
    def _is_pixel5_device(self) -> bool:
        """Check if this is a Pixel 5 device"""
        
        # Check screen dimensions
        if self.screen_width == 1080 and self.screen_height == 2340:
            return True
        
        # Check device info
        device_info = str(self.device_info).lower()
        return any(identifier in device_info for identifier in ['pixel', 'pixel_5'])
    
    def get_clickable_elements(self) -> List[Dict[str, Any]]:
        """Get all clickable UI elements"""
        
        return [element for element in self.ui_elements if element.get('clickable', False)]
    
    def find_elements_by_text(self, text: str, partial_match: bool = True) -> List[Dict[str, Any]]:
        """Find UI elements containing specific text"""
        
        matching_elements = []
        search_text = text.lower()
        
        for element in self.ui_elements:
            element_text = element.get('text', '').lower()
            content_desc = element.get('content_desc', '').lower()
            
            if partial_match:
                if search_text in element_text or search_text in content_desc:
                    matching_elements.append(element)
            else:
                if search_text == element_text or search_text == content_desc:
                    matching_elements.append(element)
        
        return matching_elements
    
    def find_elements_by_class(self, class_name: str) -> List[Dict[str, Any]]:
        """Find UI elements by class name"""
        
        return [element for element in self.ui_elements 
                if class_name.lower() in element.get('class', '').lower()]
    
    def get_largest_clickable_element(self) -> Optional[Dict[str, Any]]:
        """Get the largest clickable element (often the main content)"""
        
        clickable_elements = self.get_clickable_elements()
        
        if not clickable_elements:
            return None
        
        return max(clickable_elements, key=lambda e: e.get('area', 0))
    
    def get_pixel5_optimized_coordinates(self, element_type: str) -> List[int]:
        """Get Pixel 5 optimized coordinates for common elements"""
        
        coordinates_map = {
            # Settings navigation
            "settings_app": [270, 1200],
            "wifi_setting": [150, 400],
            "wifi_toggle": [1000, 400],
            
            # Quick settings
            "notification_pull": [540, 50],
            "quick_settings": [540, 300],
            "airplane_mode": [400, 300],
            
            # Navigation
            "home_center": [540, 1170],
            "back_gesture": [100, 1170],
            "menu_button": [540, 2200],
            
            # Common app locations
            "calculator": [540, 1400],
            "phone": [810, 1200],
            "camera": [270, 1000]
        }
        
        return coordinates_map.get(element_type, [540, 1170])
    
    def get_safe_touch_area(self) -> Dict[str, int]:
        """Get safe touch area avoiding system UI"""
        
        # For Pixel 5, avoid top status bar and bottom navigation
        return {
            'top': 100,
            'bottom': self.screen_height - 200,
            'left': 50,
            'right': self.screen_width - 50
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary"""
        
        return {
            'timestamp': self.timestamp,
            'device_id': self.device_id,
            'device_type': self.device_type,
            'screen_bounds': self.screen_bounds,
            'current_activity': self.current_activity,
            'task_name': self.task_name,
            'ui_elements_count': len(self.ui_elements),
            'clickable_elements_count': len(self.get_clickable_elements()),
            'screenshot_size': self.screenshot_size,
            'metadata': self.metadata
        }
    
    def get_screenshot_base64(self) -> str:
        """Get screenshot as base64 string"""
        
        if self.screenshot:
            return base64.b64encode(self.screenshot).decode('utf-8')
        return ''
    
    def save_screenshot(self, filepath: str) -> bool:
        """Save screenshot to file"""
        
        try:
            if self.screenshot:
                with open(filepath, 'wb') as f:
                    f.write(self.screenshot)
                print(f"📸 Screenshot saved to {filepath}")
                return True
            else:
                print("⚠️ No screenshot data to save")
                return False
        except Exception as e:
            print(f"❌ Screenshot save error: {e}")
            return False
    
    def analyze_screen_content(self) -> Dict[str, Any]:
        """Analyze screen content for testing insights"""
        
        analysis = {
            'interactive_elements': len(self.get_clickable_elements()),
            'text_elements': len([e for e in self.ui_elements if e.get('text')]),
            'buttons': len(self.find_elements_by_class('Button')),
            'text_views': len(self.find_elements_by_class('TextView')),
            'edit_texts': len(self.find_elements_by_class('EditText')),
            'scroll_views': len(self.find_elements_by_class('ScrollView')),
            'complexity_score': self._calculate_complexity_score()
        }
        
        # Detect common UI patterns
        analysis['ui_patterns'] = self._detect_ui_patterns()
        
        return analysis
    
    def _calculate_complexity_score(self) -> float:
        """Calculate UI complexity score"""
        
        if not self.ui_elements:
            return 0.0
        
        # Base score from element count
        element_score = min(len(self.ui_elements) / 50.0, 1.0)
        
        # Interactive elements bonus
        interactive_score = min(len(self.get_clickable_elements()) / 20.0, 1.0)
        
        # Text content bonus
        text_elements = [e for e in self.ui_elements if e.get('text')]
        text_score = min(len(text_elements) / 30.0, 1.0)
        
        return (element_score + interactive_score + text_score) / 3.0
    
    def _detect_ui_patterns(self) -> List[str]:
        """Detect common UI patterns"""
        
        patterns = []
        
        # Check for common patterns
        if self.find_elements_by_text('settings', partial_match=True):
            patterns.append('settings_screen')
        
        if self.find_elements_by_text('wifi', partial_match=True):
            patterns.append('wifi_settings')
        
        if self.find_elements_by_class('Button'):
            patterns.append('has_buttons')
        
        if self.find_elements_by_class('EditText'):
            patterns.append('has_input_fields')
        
        if len(self.get_clickable_elements()) > 10:
            patterns.append('complex_interface')
        
        return patterns
    
    def __str__(self) -> str:
        """String representation of observation"""
        
        return (f"ObservationWrapper(device={self.device_id}, "
                f"screen={self.screen_width}x{self.screen_height}, "
                f"elements={len(self.ui_elements)}, "
                f"activity={self.current_activity})")
    
    def __repr__(self) -> str:
        return self.__str__()
