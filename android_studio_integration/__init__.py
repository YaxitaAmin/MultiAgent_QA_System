"""
Android Studio Integration Package for Pixel 5 Virtual Device
Enhanced for QA System with proper virtual device management
"""

from .adb_manager import ADBManager
from .device_discovery import DeviceDiscovery
from .virtual_device_env import VirtualDeviceEnv
from .observation_wrapper import ObservationWrapper
from .screen_capture import ScreenCapture
from .emulator_controller import EmulatorController
from .integration_bridge import IntegrationBridge
from .eui_automation import EUIAutomation

__version__ = "1.0.0"
__author__ = "QA System Team"

# Package-level configuration
PIXEL_5_CONFIG = {
    "screen_width": 1080,
    "screen_height": 2340,
    "density": 440,
    "device_type": "pixel_5",
    "supported_android_versions": ["11", "12", "13", "14"]
}

def get_pixel5_coordinates(element_type: str) -> list:
    """Get Pixel 5 optimized coordinates for UI elements"""
    
    coordinates_map = {
        # App icons (5x6 grid layout)
        "settings": [270, 1200],
        "calculator": [540, 1400],
        "phone": [810, 1200],
        "camera": [270, 1000],
        "gallery": [540, 1000],
        
        # Settings screen
        "wifi_option": [150, 400],
        "wifi_toggle": [1000, 400],
        "bluetooth_option": [150, 500],
        "display_option": [150, 600],
        
        # Quick settings
        "wifi_qs": [200, 300],
        "airplane_qs": [400, 300],
        "bluetooth_qs": [600, 300],
        "flashlight_qs": [800, 300],
        
        # Common areas
        "center": [540, 1170],
        "top_center": [540, 200],
        "bottom_center": [540, 2100],
        
        # Navigation
        "back_gesture": [100, 2250],
        "home_gesture": [540, 2280],
        "recent_gesture": [980, 2250],
    }
    
    return coordinates_map.get(element_type, [540, 1170])

__all__ = [
    'ADBManager',
    'DeviceDiscovery', 
    'VirtualDeviceEnv',
    'ObservationWrapper',
    'ScreenCapture',
    'EmulatorController',
    'IntegrationBridge',
    'EUIAutomation',
    'PIXEL_5_CONFIG',
    'get_pixel5_coordinates'
]
