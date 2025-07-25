"""
Android World Integration for QualGent Multi-Agent QA System
Real AndroidEnv integration with comprehensive UI processing
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import cv2
from PIL import Image
import xml.etree.ElementTree as ET

# Import android_world
sys.path.append('android_world')
try:
    from android_world.env import AndroidEnv
    from android_world.task_evals import task_eval_registry
    from android_world.utils import ui_utils
    from android_world.utils.ui_utils import get_element_bounds, extract_ui_elements
    ANDROID_WORLD_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Android World not available: {e}")
    ANDROID_WORLD_AVAILABLE = False

class AndroidWorldEnvironment:
    """
    Production Android World environment integration
    Implements comprehensive UI interaction and analysis
    """
    
    def __init__(self, task_name: str = "settings_wifi", device_id: Optional[str] = None,
                 headless: bool = True, record_video: bool = True):
        self.task_name = task_name
        self.device_id = device_id
        self.headless = headless
        self.record_video = record_video
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Android environment
        if ANDROID_WORLD_AVAILABLE:
            self.env = AndroidEnv(
                task_name=task_name,
                device_id=device_id,
                headless=headless,
                record_screen=record_video
            )
            self.is_real_env = True
        else:
            self.env = self._create_mock_env()
            self.is_real_env = False
        
        # Visual trace recording
        self.visual_traces = []
        self.ui_history = []
        self.action_history = []
        
        # UI processing components
        self.ui_processor = UIProcessor()
        
        self.logger.info(f"AndroidWorld environment initialized: {task_name} (real: {self.is_real_env})")
    
    def reset(self) -> Dict:
        """Reset environment and return initial observation"""
        try:
            obs = self.env.reset()
            
            # Record initial state
            self._record_visual_frame(obs)
            self._record_ui_state(obs)
            
            return self._process_observation(obs)
            
        except Exception as e:
            self.logger.error(f"Failed to reset environment: {e}")
            return self._create_fallback_observation()
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Execute action and return comprehensive results"""
        try:
            # Log action
            self.action_history.append({
                "timestamp": time.time(),
                "action": action
            })
            
            # Execute in environment
            obs, reward, done, info = self.env.step(action)
            
            # Record post-action state
            self._record_visual_frame(obs)
            self._record_ui_state(obs)
            
            # Process and enhance observation
            processed_obs = self._process_observation(obs)
            enhanced_info = self._enhance_info(info, action)
            
            return processed_obs, reward, done, enhanced_info
            
        except Exception as e:
            self.logger.error(f"Failed to execute action: {e}")
            return self._create_fallback_observation(), -1.0, True, {"error": str(e)}
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render current state with comprehensive visual capture"""
        try:
            if hasattr(self.env, 'render'):
                frame = self.env.render(mode=mode)
                if frame is not None:
                    # Record frame for visual traces
                    self._record_visual_frame({"screenshot": frame})
                    return frame
            
            # Fallback: capture screenshot
            return self._capture_screenshot()
            
        except Exception as e:
            self.logger.error(f"Failed to render: {e}")
            return None
    
    def get_ui_hierarchy(self) -> str:
        """Get current UI hierarchy with comprehensive parsing"""
        try:
            if hasattr(self.env, 'get_ui_dump'):
                ui_dump = self.env.get_ui_dump()
                return ui_dump
            elif hasattr(self.env, 'current_observation'):
                obs = self.env.current_observation
                return obs.get('ui_hierarchy', '')
            
            return self._get_mock_ui_hierarchy()
            
        except Exception as e:
            self.logger.error(f"Failed to get UI hierarchy: {e}")
            return ""
    
    def get_ui_elements(self) -> List[Dict]:
        """Extract interactive UI elements with full details"""
        try:
            ui_hierarchy = self.get_ui_hierarchy()
            return self.ui_processor.extract_elements(ui_hierarchy)
        except Exception as e:
            self.logger.error(f"Failed to extract UI elements: {e}")
            return []
    
    def find_element_by_text(self, text: str, exact: bool = False) -> Optional[Dict]:
        """Find UI element by text with fuzzy matching"""
        elements = self.get_ui_elements()
        for element in elements:
            element_text = element.get('text', '') + ' ' + element.get('content_desc', '')
            if exact and text == element_text.strip():
                return element
            elif not exact and text.lower() in element_text.lower():
                return element
        return None
    
    def find_element_by_id(self, element_id: str) -> Optional[Dict]:
        """Find UI element by resource ID"""
        elements = self.get_ui_elements()
        for element in elements:
            if element.get('resource_id', '') == element_id:
                return element
        return None
    
    def wait_for_ui_stable(self, timeout: float = 5.0, similarity_threshold: float = 0.95) -> bool:
        """Wait for UI to stabilize using visual comparison"""
        start_time = time.time()
        previous_frame = self.render()
        
        while time.time() - start_time < timeout:
            time.sleep(0.5)
            current_frame = self.render()
            
            if previous_frame is not None and current_frame is not None:
                similarity = self._compute_frame_similarity(previous_frame, current_frame)
                if similarity >= similarity_threshold:
                    return True
            
            previous_frame = current_frame
        
        return False
    
    def detect_modal_dialog(self) -> Optional[Dict]:
        """Detect modal dialogs and popups"""
        try:
            ui_hierarchy = self.get_ui_hierarchy()
            return self.ui_processor.detect_modal(ui_hierarchy)
        except Exception as e:
            self.logger.error(f"Failed to detect modal: {e}")
            return None
    
    def get_visual_trace(self) -> List[np.ndarray]:
        """Get recorded visual trace frames"""
        return [trace['frame'] for trace in self.visual_traces if 'frame' in trace]
    
    def get_ui_trace(self) -> List[Dict]:
        """Get recorded UI state trace"""
        return self.ui_history.copy()
    
    def get_action_trace(self) -> List[Dict]:
        """Get recorded action trace"""
        return self.action_history.copy()
    
    def export_episode_data(self, episode_id: str) -> Dict:
        """Export complete episode data for analysis"""
        return {
            "episode_id": episode_id,
            "task_name": self.task_name,
            "visual_traces": len(self.visual_traces),
            "ui_traces": len(self.ui_history),
            "actions": len(self.action_history),
            "is_real_env": self.is_real_env,
            "timestamp": time.time()
        }
    
    def _process_observation(self, obs: Dict) -> Dict:
        """Process raw observation into structured format"""
        processed = {
            "screenshot": obs.get("screenshot"),
            "ui_hierarchy": obs.get("ui_hierarchy", ""),
            "task_info": obs.get("task_info", {}),
            "timestamp": time.time(),
            "step_count": len(self.action_history)
        }
        
        # Add UI element analysis
        if processed["ui_hierarchy"]:
            processed["ui_elements"] = self.ui_processor.extract_elements(processed["ui_hierarchy"])
            processed["interactive_elements"] = [
                elem for elem in processed["ui_elements"] 
                if elem.get("clickable", False)
            ]
        
        return processed
    
    def _record_visual_frame(self, obs: Dict):
        """Record visual frame for trace analysis"""
        if "screenshot" in obs and obs["screenshot"] is not None:
            self.visual_traces.append({
                "timestamp": time.time(),
                "frame": obs["screenshot"],
                "step": len(self.action_history)
            })
    
    def _record_ui_state(self, obs: Dict):
        """Record UI state for trace analysis"""
        ui_state = {
            "timestamp": time.time(),
            "ui_hierarchy": obs.get("ui_hierarchy", ""),
            "elements_count": len(self.ui_processor.extract_elements(obs.get("ui_hierarchy", ""))),
            "step": len(self.action_history)
        }
        self.ui_history.append(ui_state)
    
    def _enhance_info(self, info: Dict, action: Dict) -> Dict:
        """Enhance info with additional context"""
        enhanced = info.copy()
        enhanced.update({
            "action_executed": action,
            "ui_elements_detected": len(self.get_ui_elements()),
            "visual_trace_length": len(self.visual_traces),
            "environment_type": "real" if self.is_real_env else "mock"
        })
        return enhanced
    
    def _compute_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute similarity between two frames"""
        try:
            # Convert to grayscale for comparison
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if len(frame1.shape) == 3 else frame1
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if len(frame2.shape) == 3 else frame2
            
            # Compute structural similarity
            from skimage.metrics import structural_similarity as ssim
            similarity = ssim(gray1, gray2)
            return similarity
        except Exception:
            # Fallback: simple MSE-based similarity
            mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
            max_mse = 255.0 ** 2
            return 1.0 - (mse / max_mse)
    
    def _capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot fallback"""
        try:
            # Try to get screenshot from environment
            if hasattr(self.env, 'get_screenshot'):
                return self.env.get_screenshot()
            return None
        except Exception:
            return None
    
    def _create_mock_env(self):
        """Create mock environment when Android World not available"""
        from core.android_env_wrapper import MockAndroidEnv
        return MockAndroidEnv(self.task_name)
    
    def _create_fallback_observation(self) -> Dict:
        """Create fallback observation when errors occur"""
        return {
            "screenshot": None,
            "ui_hierarchy": "",
            "task_info": {"status": "error"},
            "timestamp": time.time(),
            "step_count": len(self.action_history),
            "ui_elements": [],
            "interactive_elements": []
        }
    
    def _get_mock_ui_hierarchy(self) -> str:
        """Get mock UI hierarchy"""
        return """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node text="Settings" class="android.widget.TextView" bounds="[100,200][300,250]" clickable="true"/>
  <node text="WiFi" class="android.widget.TextView" bounds="[100,300][500,350]" clickable="true"/>
</hierarchy>"""
    
    def close(self):
        """Close environment and cleanup resources"""
        try:
            if hasattr(self.env, 'close'):
                self.env.close()
            self.logger.info("Android environment closed")
        except Exception as e:
            self.logger.error(f"Error closing environment: {e}")

class UIProcessor:
    """Advanced UI processing for Android environments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_elements(self, ui_hierarchy: str) -> List[Dict]:
        """Extract UI elements from hierarchy XML"""
        if not ui_hierarchy:
            return []
        
        try:
            root = ET.fromstring(ui_hierarchy)
            elements = []
            self._parse_node(root, elements, "")
            return elements
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse UI hierarchy: {e}")
            return []
    
    def _parse_node(self, node: ET.Element, elements: List[Dict], path: str):
        """Recursively parse UI hierarchy nodes"""
        element = {
            "path": f"{path}/{node.tag}",
            "tag": node.tag,
            "text": node.get("text", ""),
            "content_desc": node.get("content-desc", ""),
            "class": node.get("class", ""),
            "resource_id": node.get("resource-id", ""),
            "bounds": self._parse_bounds(node.get("bounds", "[0,0][0,0]")),
            "clickable": node.get("clickable", "false").lower() == "true",
            "enabled": node.get("enabled", "true").lower() == "true",
            "focused": node.get("focused", "false").lower() == "true",
            "selected": node.get("selected", "false").lower() == "true"
        }
        
        # Calculate center point
        bounds = element["bounds"]
        element["center"] = ((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2)
        
        elements.append(element)
        
        # Process child nodes
        for i, child in enumerate(node):
            self._parse_node(child, elements, f"{element['path']}[{i}]")
    
    def _parse_bounds(self, bounds_str: str) -> Tuple[int, int, int, int]:
        """Parse bounds string like '[0,0][100,50]' to (0,0,100,50)"""
        try:
            import re
            pattern = r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]'
            match = re.match(pattern, bounds_str)
            if match:
                return tuple(map(int, match.groups()))
        except Exception:
            pass
        return (0, 0, 0, 0)
    
    def detect_modal(self, ui_hierarchy: str) -> Optional[Dict]:
        """Detect modal dialogs and popups"""
        elements = self.extract_elements(ui_hierarchy)
        
        modal_indicators = ["dialog", "popup", "modal", "alert"]
        action_words = ["ok", "cancel", "dismiss", "close", "yes", "no", "allow", "deny"]
        
        modal_elements = []
        action_elements = []
        
        for element in elements:
            # Check for modal container elements
            class_name = element.get("class", "").lower()
            if any(indicator in class_name for indicator in modal_indicators):
                modal_elements.append(element)
            
            # Check for action buttons
            text = (element.get("text", "") + " " + element.get("content_desc", "")).lower()
            if any(action in text for action in action_words) and element.get("clickable"):
                action_elements.append(element)
        
        if modal_elements or action_elements:
            return {
                "modal_detected": True,
                "modal_elements": modal_elements,
                "action_elements": action_elements,
                "suggested_actions": [elem["path"] for elem in action_elements],
                "modal_type": "dialog" if modal_elements else "system_popup"
            }
        
        return None
