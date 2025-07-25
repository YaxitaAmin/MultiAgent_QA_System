# core/ui_utils.py
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2
import numpy as np
from loguru import logger
import time

@dataclass
class UIElement:
    element_id: str
    bounds: Tuple[int, int, int, int]  # (left, top, right, bottom)
    text: str
    content_desc: str
    class_name: str
    clickable: bool
    enabled: bool
    focused: bool
    scrollable: bool
    checkable: bool
    checked: bool

@dataclass
class UIState:
    elements: List[UIElement]
    hierarchy: Dict[str, Any]
    screenshot_path: Optional[str] = None
    timestamp: float = 0.0

class UITreeParser:
    """Parse Android UI hierarchy and extract actionable elements"""
    
    def __init__(self):
        self.element_counter = 0
    
    def parse_ui_hierarchy(self, ui_dump: str) -> UIState:
        """Parse UI hierarchy from XML dump"""
        try:
            root = ET.fromstring(ui_dump)
            elements = []
            hierarchy = self._build_hierarchy(root)
            
            self._extract_elements(root, elements)
            
            return UIState(
                elements=elements,
                hierarchy=hierarchy,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"UI parsing error: {e}")
            return UIState(elements=[], hierarchy={})
    
    def _build_hierarchy(self, node: ET.Element, depth: int = 0) -> Dict[str, Any]:
        """Build hierarchical representation of UI"""
        hierarchy = {
            "tag": node.tag,
            "attrib": node.attrib,
            "depth": depth,
            "children": []
        }
        
        for child in node:
            hierarchy["children"].append(self._build_hierarchy(child, depth + 1))
        
        return hierarchy
    
    def _extract_elements(self, node: ET.Element, elements: List[UIElement]):
        """Extract actionable UI elements"""
        bounds_str = node.get('bounds', '')
        if bounds_str and self._is_actionable(node):
            bounds = self._parse_bounds(bounds_str)
            if bounds:
                element = UIElement(
                    element_id=f"element_{self.element_counter}",
                    bounds=bounds,
                    text=node.get('text', ''),
                    content_desc=node.get('content-desc', ''),
                    class_name=node.get('class', ''),
                    clickable=node.get('clickable', 'false').lower() == 'true',
                    enabled=node.get('enabled', 'false').lower() == 'true',
                    focused=node.get('focused', 'false').lower() == 'true',
                    scrollable=node.get('scrollable', 'false').lower() == 'true',
                    checkable=node.get('checkable', 'false').lower() == 'true',
                    checked=node.get('checked', 'false').lower() == 'true'
                )
                elements.append(element)
                self.element_counter += 1
        
        # Recursively process children
        for child in node:
            self._extract_elements(child, elements)
    
    def _is_actionable(self, node: ET.Element) -> bool:
        """Check if element is actionable"""
        return (
            node.get('clickable', 'false').lower() == 'true' or
            node.get('scrollable', 'false').lower() == 'true' or
            node.get('checkable', 'false').lower() == 'true' or
            node.get('focusable', 'false').lower() == 'true' or
            'Button' in node.get('class', '') or
            'Switch' in node.get('class', '') or
            'EditText' in node.get('class', '')
        )
    
    def _parse_bounds(self, bounds_str: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse bounds string '[left,top][right,bottom]' to tuple"""
        try:
            # Remove brackets and split
            bounds_str = bounds_str.replace('[', '').replace(']', ',')
            coords = [int(x) for x in bounds_str.split(',') if x]
            if len(coords) >= 4:
                return (coords[0], coords[1], coords[2], coords[3])
        except Exception as e:
            logger.warning(f"Bounds parsing error: {e}")
        return None

class UIElementMatcher:
    """Match UI elements based on various criteria"""
    
    def __init__(self):
        pass
    
    def find_by_text(self, elements: List[UIElement], text: str, partial: bool = True) -> List[UIElement]:
        """Find elements by text content"""
        if partial:
            return [e for e in elements if text.lower() in e.text.lower()]
        else:
            return [e for e in elements if e.text.lower() == text.lower()]
    
    def find_by_content_desc(self, elements: List[UIElement], desc: str, partial: bool = True) -> List[UIElement]:
        """Find elements by content description"""
        if partial:
            return [e for e in elements if desc.lower() in e.content_desc.lower()]
        else:
            return [e for e in elements if e.content_desc.lower() == desc.lower()]
    
    def find_by_class(self, elements: List[UIElement], class_name: str) -> List[UIElement]:
        """Find elements by class name"""
        return [e for e in elements if class_name in e.class_name]
    
    def find_clickable(self, elements: List[UIElement]) -> List[UIElement]:
        """Find all clickable elements"""
        return [e for e in elements if e.clickable and e.enabled]
    
    def find_switches(self, elements: List[UIElement]) -> List[UIElement]:
        """Find toggle switches"""
        return [e for e in elements if 'Switch' in e.class_name or e.checkable]
    
    def find_wifi_toggle(self, elements: List[UIElement]) -> Optional[UIElement]:
        """Find Wi-Fi toggle switch specifically"""
        # Look for WiFi/Wi-Fi related elements
        wifi_keywords = ['wifi', 'wi-fi', 'wireless']
        
        for element in elements:
            element_text = (element.text + " " + element.content_desc).lower()
            if any(keyword in element_text for keyword in wifi_keywords):
                if element.checkable or 'Switch' in element.class_name:
                    return element
        
        return None
    
    def get_element_center(self, element: UIElement) -> Tuple[int, int]:
        """Get center coordinates of element"""
        left, top, right, bottom = element.bounds
        return ((left + right) // 2, (top + bottom) // 2)

class ActionGrounder:
    """Ground high-level actions to specific UI interactions"""
    
    def __init__(self):
        self.matcher = UIElementMatcher()
    
    def ground_action(self, action_desc: str, ui_state: UIState) -> Optional[Dict[str, Any]]:
        """Ground action description to specific UI action"""
        action_desc_lower = action_desc.lower()
        
        if 'open settings' in action_desc_lower:
            return self._ground_open_settings(ui_state)
        elif 'wifi' in action_desc_lower and 'toggle' in action_desc_lower:
            return self._ground_wifi_toggle(ui_state)
        elif 'navigate' in action_desc_lower and 'wifi' in action_desc_lower:
            return self._ground_navigate_wifi(ui_state)
        elif 'click' in action_desc_lower or 'tap' in action_desc_lower:
            return self._ground_click_action(action_desc, ui_state)
        
        return None
    
    def _ground_open_settings(self, ui_state: UIState) -> Optional[Dict[str, Any]]:
        """Ground 'open settings' action"""
        settings_elements = self.matcher.find_by_text(ui_state.elements, 'settings')
        if settings_elements:
            element = settings_elements[0]
            x, y = self.matcher.get_element_center(element)
            return {
                "action_type": "touch",
                "element_id": element.element_id,
                "coordinates": [x, y],
                "target_text": element.text
            }
        return None
    
    def _ground_wifi_toggle(self, ui_state: UIState) -> Optional[Dict[str, Any]]:
        """Ground Wi-Fi toggle action"""
        wifi_element = self.matcher.find_wifi_toggle(ui_state.elements)
        if wifi_element:
            x, y = self.matcher.get_element_center(wifi_element)
            return {
                "action_type": "touch",
                "element_id": wifi_element.element_id,
                "coordinates": [x, y],
                "target_text": wifi_element.text,
                "current_state": wifi_element.checked
            }
        return None
    
    def _ground_navigate_wifi(self, ui_state: UIState) -> Optional[Dict[str, Any]]:
        """Ground navigate to Wi-Fi settings"""
        wifi_elements = self.matcher.find_by_text(ui_state.elements, 'wifi')
        if not wifi_elements:
            wifi_elements = self.matcher.find_by_content_desc(ui_state.elements, 'wifi')
        
        if wifi_elements:
            element = wifi_elements[0]
            x, y = self.matcher.get_element_center(element)
            return {
                "action_type": "touch",
                "element_id": element.element_id,
                "coordinates": [x, y],
                "target_text": element.text
            }
        return None
    
    def _ground_click_action(self, action_desc: str, ui_state: UIState) -> Optional[Dict[str, Any]]:
        """Ground generic click action"""
        # Extract target from action description
        words = action_desc.lower().split()
        if 'click' in words:
            click_idx = words.index('click')
            if click_idx + 1 < len(words):
                target = words[click_idx + 1]
                elements = self.matcher.find_by_text(ui_state.elements, target)
                if elements:
                    element = elements[0]
                    x, y = self.matcher.get_element_center(element)
                    return {
                        "action_type": "touch",
                        "element_id": element.element_id,
                        "coordinates": [x, y],
                        "target_text": element.text
                    }
        return None
