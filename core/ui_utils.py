"""
UI utilities for Android UI parsing and interaction
"""

import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np
from loguru import logger
import time

from core.logger import QALogger

class ElementType(Enum):
    """UI element types"""
    BUTTON = "button"
    TEXT = "text"
    EDIT_TEXT = "edit_text"
    IMAGE = "image"
    LIST = "list"
    TOGGLE = "toggle"
    CHECKBOX = "checkbox"
    MENU = "menu"
    UNKNOWN = "unknown"

@dataclass
class UIElement:
    """Represents a UI element"""
    element_id: str
    element_type: ElementType
    text: Optional[str]
    resource_id: Optional[str]
    class_name: Optional[str]
    bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    clickable: bool
    enabled: bool
    visible: bool
    content_description: Optional[str] = None
    package: Optional[str] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates of element"""
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """Get area of element"""
        x1, y1, x2, y2 = self.bounds
        return (x2 - x1) * (y2 - y1)

@dataclass
class ActionResult:
    """Result of a UI action"""
    success: bool
    action_type: str
    target_element: Optional[UIElement]
    error_message: Optional[str] = None
    before_state: Optional[str] = None
    after_state: Optional[str] = None
    execution_time: float = 0.0

@dataclass
class UIState:
    """Complete UI state representation"""
    elements: List[UIElement]
    hierarchy: Dict[str, Any]
    screenshot_path: Optional[str] = None
    timestamp: float = 0.0

class UIParser:
    """Parse Android UI hierarchy and extract actionable elements"""
    
    def __init__(self):
        self.logger = QALogger("UIParser")
        self.element_counter = 0
    
    def parse_ui_hierarchy(self, ui_xml: str) -> List[UIElement]:
        """Parse UI hierarchy XML and return list of UI elements"""
        try:
            if not ui_xml or ui_xml.strip() == "<hierarchy></hierarchy>":
                self.logger.warning("Empty or invalid UI hierarchy")
                return []
            
            root = ET.fromstring(ui_xml)
            elements = []
            
            # Recursively parse all nodes
            self._parse_node(root, elements)
            
            self.logger.debug(f"Parsed {len(elements)} UI elements")
            return elements
            
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse UI XML: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error parsing UI: {e}")
            return []
    
    def _parse_node(self, node: ET.Element, elements: List[UIElement], parent_package: str = "") -> None:
        """Recursively parse XML node and extract UI elements"""
        try:
            # Extract element attributes
            resource_id = node.get('resource-id', '')
            class_name = node.get('class', '')
            text = node.get('text', '')
            content_desc = node.get('content-desc', '')
            bounds_str = node.get('bounds', '')
            clickable = node.get('clickable', 'false').lower() == 'true'
            enabled = node.get('enabled', 'true').lower() == 'true'
            visible = node.get('visible-to-user', 'true').lower() == 'true'
            package = node.get('package', parent_package)
            
            # Parse bounds
            bounds = self._parse_bounds(bounds_str)
            if not bounds:
                bounds = (0, 0, 0, 0)
            
            # Determine element type
            element_type = self._determine_element_type(class_name, resource_id, text)
            
            # Generate unique element ID
            element_id = self._generate_element_id(resource_id, text, content_desc, class_name)
            
            # Create UI element if it's meaningful
            if self._is_meaningful_element(class_name, text, content_desc, clickable, bounds):
                ui_element = UIElement(
                    element_id=element_id,
                    element_type=element_type,
                    text=text if text else None,
                    resource_id=resource_id if resource_id else None,
                    class_name=class_name,
                    bounds=bounds,
                    clickable=clickable,
                    enabled=enabled,
                    visible=visible,
                    content_description=content_desc if content_desc else None,
                    package=package if package else None
                )
                elements.append(ui_element)
            
            # Recursively parse child nodes
            for child in node:
                self._parse_node(child, elements, package)
                
        except Exception as e:
            self.logger.error(f"Error parsing node: {e}")
    
    def _parse_bounds(self, bounds_str: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse bounds string like '[0,0][100,50]' into coordinates"""
        try:
            if not bounds_str:
                return None
            
            # Remove brackets and split
            coords = bounds_str.replace('[', '').replace(']', ',').split(',')
            if len(coords) >= 4:
                x1, y1, x2, y2 = map(int, coords[:4])
                return (x1, y1, x2, y2)
            
            return None
        except (ValueError, IndexError):
            return None
    
    def _determine_element_type(self, class_name: str, resource_id: str, text: str) -> ElementType:
        """Determine element type based on class name and other attributes"""
        class_name_lower = class_name.lower()
        resource_id_lower = resource_id.lower()
        text_lower = text.lower()
        
        # Button-like elements
        if any(keyword in class_name_lower for keyword in ['button', 'imagebutton']):
            return ElementType.BUTTON
        
        # Text input elements
        if any(keyword in class_name_lower for keyword in ['edittext', 'textinput']):
            return ElementType.EDIT_TEXT
        
        # Text display elements
        if any(keyword in class_name_lower for keyword in ['textview', 'text']):
            return ElementType.TEXT
        
        # Image elements
        if any(keyword in class_name_lower for keyword in ['imageview', 'image']):
            return ElementType.IMAGE
        
        # List elements
        if any(keyword in class_name_lower for keyword in ['listview', 'recyclerview', 'list']):
            return ElementType.LIST
        
        # Toggle/Switch elements
        if any(keyword in class_name_lower for keyword in ['switch', 'toggle', 'togglebutton']):
            return ElementType.TOGGLE
        
        # Checkbox elements
        if any(keyword in class_name_lower for keyword in ['checkbox', 'checkable']):
            return ElementType.CHECKBOX
        
        # Menu elements
        if any(keyword in class_name_lower for keyword in ['menu', 'popup']):
            return ElementType.MENU
        
        # Check resource ID and text for additional hints
        if any(keyword in resource_id_lower for keyword in ['button', 'btn']):
            return ElementType.BUTTON
        
        if any(keyword in resource_id_lower for keyword in ['toggle', 'switch']):
            return ElementType.TOGGLE
        
        if any(keyword in text_lower for keyword in ['on', 'off', 'enable', 'disable']) and len(text) < 10:
            return ElementType.TOGGLE
        
        return ElementType.UNKNOWN
    
    def _generate_element_id(self, resource_id: str, text: str, content_desc: str, class_name: str) -> str:
        """Generate unique element identifier"""
        # Prefer resource ID if available
        if resource_id:
            return resource_id
        
        # Use text if meaningful
        if text and len(text.strip()) > 0 and len(text) < 50:
            # Clean text for ID
            clean_text = re.sub(r'[^\w\s-]', '', text.strip())
            clean_text = re.sub(r'\s+', '_', clean_text)
            return f"text_{clean_text.lower()}"
        
        # Use content description
        if content_desc and len(content_desc.strip()) > 0:
            clean_desc = re.sub(r'[^\w\s-]', '', content_desc.strip())
            clean_desc = re.sub(r'\s+', '_', clean_desc)
            return f"desc_{clean_desc.lower()}"
        
        # Fallback to class name with timestamp
        class_simple = class_name.split('.')[-1] if '.' in class_name else class_name
        timestamp = int(time.time() * 1000) % 10000  # Last 4 digits for uniqueness
        return f"{class_simple.lower()}_{timestamp}"
    
    def _is_meaningful_element(self, class_name: str, text: str, content_desc: str, 
                              clickable: bool, bounds: Tuple[int, int, int, int]) -> bool:
        """Determine if element is meaningful for QA testing"""
        # Skip decorative elements
        if any(keyword in class_name.lower() for keyword in ['decoration', 'divider', 'space']):
            return False
        
        # Include clickable elements
        if clickable:
            return True
        
        # Include elements with meaningful text
        if text and len(text.strip()) > 0:
            return True
        
        # Include elements with content description
        if content_desc and len(content_desc.strip()) > 0:
            return True
        
        # Include input fields
        if 'edit' in class_name.lower():
            return True
        
        # Check if element has reasonable size
        x1, y1, x2, y2 = bounds
        area = (x2 - x1) * (y2 - y1)
        return area > 100  # Minimum area threshold
    
    def find_elements_by_text(self, elements: List[UIElement], text: str, exact_match: bool = False) -> List[UIElement]:
        """Find elements by text content"""
        results = []
        text_lower = text.lower()
        
        for element in elements:
            if element.text:
                element_text_lower = element.text.lower()
                if exact_match:
                    if element_text_lower == text_lower:
                        results.append(element)
                else:
                    if text_lower in element_text_lower:
                        results.append(element)
        
        return results
    
    def find_elements_by_resource_id(self, elements: List[UIElement], resource_id: str) -> List[UIElement]:
        """Find elements by resource ID"""
        return [element for element in elements if element.resource_id == resource_id]
    
    def find_clickable_elements(self, elements: List[UIElement]) -> List[UIElement]:
        """Find all clickable elements"""
        return [element for element in elements if element.clickable and element.enabled and element.visible]
    
    def find_text_input_elements(self, elements: List[UIElement]) -> List[UIElement]:
        """Find text input elements"""
        return [element for element in elements if element.element_type == ElementType.EDIT_TEXT]
    
    def find_toggle_elements(self, elements: List[UIElement]) -> List[UIElement]:
        """Find toggle/switch elements"""
        return [element for element in elements if element.element_type == ElementType.TOGGLE]
    
    def analyze_screen_state(self, elements: List[UIElement]) -> Dict[str, Any]:
        """Analyze current screen state"""
        total_elements = len(elements)
        clickable_elements = len(self.find_clickable_elements(elements))
        input_elements = len(self.find_text_input_elements(elements))
        toggle_elements = len(self.find_toggle_elements(elements))
        
        # Extract key UI components
        buttons = [e for e in elements if e.element_type == ElementType.BUTTON]
        texts = [e for e in elements if e.element_type == ElementType.TEXT and e.text]
        
        # Screen title detection
        title_candidates = [e for e in texts if e.bounds[1] < 200]  # Top of screen
        screen_title = title_candidates[0].text if title_candidates else "Unknown Screen"
        
        return {
            "screen_title": screen_title,
            "total_elements": total_elements,
            "clickable_elements": clickable_elements,
            "input_elements": input_elements,
            "toggle_elements": toggle_elements,
            "buttons": [{"text": b.text, "id": b.element_id} for b in buttons[:5]],
            "key_texts": [t.text for t in texts[:10] if t.text and len(t.text) < 50],
            "interaction_opportunities": clickable_elements + input_elements
        }

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
                    element_type=ElementType.UNKNOWN,
                    text=node.get('text', ''),
                    resource_id=node.get('resource-id', ''),
                    class_name=node.get('class', ''),
                    bounds=bounds,
                    clickable=node.get('clickable', 'false').lower() == 'true',
                    enabled=node.get('enabled', 'false').lower() == 'true',
                    visible=node.get('visible-to-user', 'true').lower() == 'true',
                    content_description=node.get('content-desc', ''),
                    package=node.get('package', '')
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
            return [e for e in elements if e.text and text.lower() in e.text.lower()]
        else:
            return [e for e in elements if e.text and e.text.lower() == text.lower()]
    
    def find_by_content_desc(self, elements: List[UIElement], desc: str, partial: bool = True) -> List[UIElement]:
        """Find elements by content description"""
        if partial:
            return [e for e in elements if e.content_description and desc.lower() in e.content_description.lower()]
        else:
            return [e for e in elements if e.content_description and e.content_description.lower() == desc.lower()]
    
    def find_by_class(self, elements: List[UIElement], class_name: str) -> List[UIElement]:
        """Find elements by class name"""
        return [e for e in elements if e.class_name and class_name in e.class_name]
    
    def find_clickable(self, elements: List[UIElement]) -> List[UIElement]:
        """Find all clickable elements"""
        return [e for e in elements if e.clickable and e.enabled]
    
    def find_switches(self, elements: List[UIElement]) -> List[UIElement]:
        """Find toggle switches"""
        return [e for e in elements if e.element_type == ElementType.TOGGLE or (e.class_name and 'Switch' in e.class_name)]
    
    def find_wifi_toggle(self, elements: List[UIElement]) -> Optional[UIElement]:
        """Find Wi-Fi toggle switch specifically"""
        # Look for WiFi/Wi-Fi related elements
        wifi_keywords = ['wifi', 'wi-fi', 'wireless']
        
        for element in elements:
            element_text = f"{element.text or ''} {element.content_description or ''}".lower()
            if any(keyword in element_text for keyword in wifi_keywords):
                if element.element_type == ElementType.TOGGLE or (element.class_name and 'Switch' in element.class_name):
                    return element
        
        return None
    
    def get_element_center(self, element: UIElement) -> Tuple[int, int]:
        """Get center coordinates of element"""
        return element.center

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
                "current_state": getattr(wifi_element, 'checked', False)
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
