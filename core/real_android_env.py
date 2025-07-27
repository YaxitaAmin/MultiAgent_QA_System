# core/real_android_env.py
"""
Real Android Environment Integration for Agent-S QA System
Connects to Android Studio AVD via ADB
"""

import subprocess
import time
import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from pathlib import Path

@dataclass
class AndroidDeviceInfo:
    device_id: str
    device_name: str
    android_version: str
    screen_size: Tuple[int, int]
    is_emulator: bool

@dataclass 
class RealAndroidObservation:
    screenshot: np.ndarray
    ui_hierarchy: str
    current_activity: str
    screen_bounds: Tuple[int, int]
    timestamp: float
    device_info: AndroidDeviceInfo

class RealAndroidEnvWrapper:
    """
    Real Android Environment for Agent-S Integration
    Connects to Android Studio AVD through ADB
    """
    
    def __init__(self, device_id: Optional[str] = None, adb_path: Optional[str] = None):
        self.device_id = device_id or self._get_default_device()
        self.adb_path = adb_path or self._find_adb_path()
        self.screen_size = (1080, 1920)  # Will be detected automatically
        self.device_info = None
        self._last_screenshot = None
        
        # Initialize connection
        if not self._verify_device_connection():
            raise Exception(f"Cannot connect to Android device: {self.device_id}")
        
        self._initialize_device_info()
        self.logger = self._setup_logger()
        
    def _find_adb_path(self) -> str:
        """Find ADB path automatically"""
        possible_paths = [
            # Android Studio default locations
            os.path.expanduser("~/Android/Sdk/platform-tools/adb"),  # Linux/Mac
            os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),  # Mac
            "C:\\Android\\platform-tools\\adb.exe",  # Windows
            # System PATH
            "adb"
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return path
            except:
                continue
        
        raise Exception("ADB not found. Please install Android SDK or set ADB_PATH environment variable")
    
    def _get_default_device(self) -> str:
        """Get the first available Android device/emulator"""
        try:
            result = subprocess.run([self.adb_path, "devices"], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if '\tdevice' in line:
                    return line.split('\t')[0]
            
            raise Exception("No Android devices found")
        except Exception as e:
            return "emulator-5554"  # Default emulator
    
    def _verify_device_connection(self) -> bool:
        """Verify device is connected and responsive"""
        try:
            cmd = [self.adb_path, "-s", self.device_id, "shell", "echo", "test"]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _initialize_device_info(self):
        """Get device information"""
        try:
            # Get device properties
            props = self._get_device_properties()
            
            # Get screen size
            screen_size = self._get_screen_size()
            
            self.device_info = AndroidDeviceInfo(
                device_id=self.device_id,
                device_name=props.get("ro.product.model", "Unknown"),
                android_version=props.get("ro.build.version.release", "Unknown"),
                screen_size=screen_size,
                is_emulator="emulator" in self.device_id
            )
            
            self.screen_size = screen_size
            
        except Exception as e:
            print(f"Warning: Could not get device info: {e}")
            self.device_info = AndroidDeviceInfo(
                device_id=self.device_id,
                device_name="Unknown",
                android_version="Unknown", 
                screen_size=(1080, 1920),
                is_emulator=True
            )
    
    def _get_device_properties(self) -> Dict[str, str]:
        """Get device system properties"""
        try:
            cmd = [self.adb_path, "-s", self.device_id, "shell", "getprop"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            props = {}
            for line in result.stdout.split('\n'):
                if ']: [' in line:
                    key = line.split(']: [')[0].strip('[')
                    value = line.split(']: [')[1].strip(']')
                    props[key] = value
            
            return props
        except:
            return {}
    
    def _get_screen_size(self) -> Tuple[int, int]:
        """Get device screen resolution"""
        try:
            cmd = [self.adb_path, "-s", self.device_id, "shell", "wm", "size"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse output like "Physical size: 1080x1920"
            size_line = result.stdout.strip()
            if 'x' in size_line:
                size_part = size_line.split(': ')[-1]
                width, height = map(int, size_part.split('x'))
                return (width, height)
                
        except:
            pass
        
        return (1080, 1920)  # Default
    
    def reset(self) -> RealAndroidObservation:
        """Reset to home screen and return initial observation"""
        try:
            # Go to home screen
            self._adb_shell(["input", "keyevent", "KEYCODE_HOME"])
            time.sleep(2)
            
            # Wait for UI to stabilize
            self._wait_for_ui_stable()
            
            return self.get_observation()
            
        except Exception as e:
            print(f"Error during reset: {e}")
            raise
    
    def get_observation(self) -> RealAndroidObservation:
        """Get current UI state observation"""
        screenshot = self._capture_screenshot()
        ui_hierarchy = self._get_ui_hierarchy()
        current_activity = self._get_current_activity()
        
        return RealAndroidObservation(
            screenshot=screenshot,
            ui_hierarchy=ui_hierarchy,
            current_activity=current_activity,
            screen_bounds=self.screen_size,
            timestamp=time.time(),
            device_info=self.device_info
        )
    
    def _capture_screenshot(self) -> np.ndarray:
        """Capture device screenshot"""
        try:
            # Capture screenshot to device
            cmd = [self.adb_path, "-s", self.device_id, "shell", "screencap", "/sdcard/screenshot.png"]
            subprocess.run(cmd, check=True, timeout=10)
            
            # Pull screenshot to local
            temp_path = "temp_screenshot.png"
            cmd = [self.adb_path, "-s", self.device_id, "pull", "/sdcard/screenshot.png", temp_path]
            subprocess.run(cmd, check=True, timeout=10)
            
            # Load and return as numpy array
            screenshot = cv2.imread(temp_path)
            if screenshot is not None:
                self._last_screenshot = screenshot
                # Cleanup
                try:
                    os.remove(temp_path)
                except:
                    pass
                return screenshot
            else:
                # Return last screenshot if capture failed
                return self._last_screenshot if self._last_screenshot is not None else np.zeros((1920, 1080, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"Screenshot capture failed: {e}")
            # Return black image if capture fails
            return np.zeros((1920, 1080, 3), dtype=np.uint8)
    
    def _get_ui_hierarchy(self) -> str:
        """Get UI hierarchy dump"""
        try:
            # Dump UI hierarchy
            cmd = [self.adb_path, "-s", self.device_id, "shell", "uiautomator", "dump", "/sdcard/ui_dump.xml"]
            subprocess.run(cmd, check=True, timeout=15)
            
            # Pull UI dump
            temp_path = "temp_ui_dump.xml"
            cmd = [self.adb_path, "-s", self.device_id, "pull", "/sdcard/ui_dump.xml", temp_path]
            subprocess.run(cmd, check=True, timeout=10)
            
            # Read UI hierarchy
            with open(temp_path, 'r', encoding='utf-8') as f:
                ui_hierarchy = f.read()
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
                
            return ui_hierarchy
            
        except Exception as e:
            print(f"UI hierarchy capture failed: {e}")
            return "<hierarchy></hierarchy>"
    
    def _get_current_activity(self) -> str:
        """Get current foreground activity"""
        try:
            cmd = [self.adb_path, "-s", self.device_id, "shell", "dumpsys", "activity", "activities"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Parse current activity from dumpsys output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'mResumedActivity' in line or 'mCurrentFocus' in line:
                    # Extract activity name
                    if '{' in line and '}' in line:
                        activity_part = line.split('{')[1].split('}')[0]
                        if '/' in activity_part:
                            return activity_part.split('/')[-1]
            
            return "unknown_activity"
            
        except:
            return "unknown_activity"
    
    def _wait_for_ui_stable(self, timeout: int = 5):
        """Wait for UI to stabilize"""
        time.sleep(1)  # Basic wait - could be enhanced with UI change detection
    
    def _adb_shell(self, command: list, timeout: int = 30) -> subprocess.CompletedProcess:
        """Execute ADB shell command"""
        full_cmd = [self.adb_path, "-s", self.device_id, "shell"] + command
        return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    
    # Action Methods
    def touch(self, x: int, y: int) -> bool:
        """Perform touch action at coordinates"""
        try:
            result = self._adb_shell(["input", "tap", str(x), str(y)])
            time.sleep(0.5)  # Wait for action to complete
            return result.returncode == 0
        except:
            return False
    
    def scroll(self, direction: str, start_x: int = None, start_y: int = None) -> bool:
        """Perform scroll action"""
        try:
            # Default scroll coordinates (center of screen)
            if start_x is None:
                start_x = self.screen_size[0] // 2
            if start_y is None:
                start_y = self.screen_size[1] // 2
            
            # Calculate end coordinates based on direction
            if direction.lower() == "up":
                end_y = start_y - 500
                end_x = start_x
            elif direction.lower() == "down":
                end_y = start_y + 500
                end_x = start_x
            elif direction.lower() == "left":
                end_x = start_x - 500
                end_y = start_y
            elif direction.lower() == "right":
                end_x = start_x + 500
                end_y = start_y
            else:
                return False
            
            # Perform swipe
            result = self._adb_shell([
                "input", "swipe", 
                str(start_x), str(start_y), 
                str(end_x), str(end_y), 
                "300"  # Duration in ms
            ])
            
            time.sleep(0.8)  # Wait for scroll to complete
            return result.returncode == 0
            
        except:
            return False
    
    def type_text(self, text: str) -> bool:
        """Type text input"""
        try:
            # Escape special characters
            escaped_text = text.replace(" ", "%s").replace("'", "\\'")
            result = self._adb_shell(["input", "text", escaped_text])
            time.sleep(0.3)
            return result.returncode == 0
        except:
            return False
    
    def back(self) -> bool:
        """Press back button"""
        try:
            result = self._adb_shell(["input", "keyevent", "KEYCODE_BACK"])
            time.sleep(0.5)
            return result.returncode == 0
        except:
            return False
    
    def home(self) -> bool:
        """Press home button"""
        try:
            result = self._adb_shell(["input", "keyevent", "KEYCODE_HOME"])
            time.sleep(0.5)
            return result.returncode == 0
        except:
            return False
    
    def open_app(self, package_name: str, activity_name: str = None) -> bool:
        """Open Android app"""
        try:
            if activity_name:
                # Open specific activity
                result = self._adb_shell([
                    "am", "start", "-n", f"{package_name}/{activity_name}"
                ])
            else:
                # Open main activity
                result = self._adb_shell([
                    "monkey", "-p", package_name, "-c", 
                    "android.intent.category.LAUNCHER", "1"
                ])
            
            time.sleep(2)  # Wait for app to launch
            return result.returncode == 0
            
        except:
            return False
    
    def close(self):
        """Cleanup resources"""
        try:
            # Clean up temporary files on device
            self._adb_shell(["rm", "/sdcard/screenshot.png"])
            self._adb_shell(["rm", "/sdcard/ui_dump.xml"])
        except:
            pass
    
    def _setup_logger(self):
        """Setup logging for real Android environment"""
        import logging
        logger = logging.getLogger(f"RealAndroidEnv_{self.device_id}")
        logger.setLevel(logging.INFO)
        return logger
