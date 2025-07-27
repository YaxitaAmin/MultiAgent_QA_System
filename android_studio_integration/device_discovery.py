"""
Device Discovery for Android Studio Integration
Handles detection and setup of virtual and physical devices
"""

import subprocess
import time
import re
from typing import List, Dict, Optional
from .adb_manager import ADBManager

class DeviceDiscovery:
    """Enhanced device discovery with Pixel 5 virtual device support"""
    
    def __init__(self):
        self.adb = ADBManager()
        self.preferred_devices = [
            "emulator-5554",
            "emulator-5556", 
            "emulator-5558"
        ]
        self.pixel5_identifiers = [
            "Pixel_5",
            "pixel_5", 
            "Pixel 5"
        ]
    
    def discover_devices(self) -> List[Dict[str, str]]:
        """Discover all available devices with detailed information"""
        
        devices = []
        connected_device_ids = self.adb.get_connected_devices()
        
        print(f"🔍 Discovered {len(connected_device_ids)} connected devices")
        
        for device_id in connected_device_ids:
            device_info = self._get_device_details(device_id)
            if device_info:
                devices.append(device_info)
                print(f"📱 Device: {device_id} - {device_info.get('model', 'Unknown')}")
        
        return devices
    
    def _get_device_details(self, device_id: str) -> Optional[Dict[str, str]]:
        """Get detailed information for a specific device"""
        
        try:
            info = self.adb.get_device_info(device_id)
            
            # Determine device type
            device_type = "physical"
            if "emulator" in device_id:
                device_type = "emulator"
            
            # Check if it's a Pixel 5
            is_pixel5 = any(identifier in info.get("model", "").lower() 
                           for identifier in ["pixel", "pixel_5"])
            
            return {
                "device_id": device_id,
                "type": device_type,
                "model": info.get("model", "Unknown"),
                "android_version": info.get("android_version", "Unknown"),
                "screen_resolution": info.get("screen_resolution", "Unknown"),
                "density": info.get("density", "Unknown"),
                "is_pixel5": is_pixel5,
                "status": "online"
            }
            
        except Exception as e:
            print(f"❌ Failed to get device details for {device_id}: {e}")
            return None
    
    def find_pixel5_devices(self) -> List[Dict[str, str]]:
        """Find specifically Pixel 5 devices (virtual or physical)"""
        
        all_devices = self.discover_devices()
        pixel5_devices = []
        
        for device in all_devices:
            # Check model name and identifiers
            model = device.get("model", "").lower()
            device_id = device.get("device_id", "").lower()
            
            is_pixel5 = (
                "pixel" in model or
                any(identifier.lower() in device_id for identifier in self.pixel5_identifiers) or
                device.get("is_pixel5", False)
            )
            
            if is_pixel5:
                pixel5_devices.append(device)
                print(f"📱 Found Pixel 5: {device['device_id']}")
        
        return pixel5_devices
    
    def setup_recommended_device(self) -> Optional[str]:
        """Setup and return the best available device for testing"""
        
        print("🔍 Setting up recommended device...")
        
        # First, try to find Pixel 5 devices
        pixel5_devices = self.find_pixel5_devices()
        if pixel5_devices:
            device = pixel5_devices[0]
            device_id = device["device_id"]
            print(f"✅ Using Pixel 5 device: {device_id}")
            
            # Verify device is ready
            if self._verify_device_ready(device_id):
                return device_id
        
        # Fallback to any connected device
        all_devices = self.discover_devices()
        if all_devices:
            device = all_devices[0]
            device_id = device["device_id"]
            print(f"✅ Using fallback device: {device_id}")
            
            if self._verify_device_ready(device_id):
                return device_id
        
        # Try to start an emulator if none available
        return self._try_start_emulator()
    
    def _verify_device_ready(self, device_id: str) -> bool:
        """Verify device is ready for testing"""
        
        try:
            # Wait for device to be ready
            if not self.adb.wait_for_device(device_id, timeout=30):
                print(f"⚠️ Device {device_id} not ready")
                return False
            
            # Check if device is responsive
            result = self.adb.execute_command(device_id, ["shell", "echo", "test"])
            if result != "test":
                print(f"⚠️ Device {device_id} not responsive")
                return False
            
            # Check screen state
            screen_state = self.adb.execute_command(device_id, ["shell", "dumpsys", "power", "|", "grep", "mScreenOn"])
            if "mScreenOn=false" in screen_state:
                print(f"📱 Waking up device {device_id}")
                self.adb.press_key(device_id, "KEYCODE_WAKEUP")
                time.sleep(2)
            
            print(f"✅ Device {device_id} verified and ready")
            return True
            
        except Exception as e:
            print(f"❌ Device verification failed: {e}")
            return False
    
    def _try_start_emulator(self) -> Optional[str]:
        """Try to start a Pixel 5 emulator"""
        
        print("🚀 Attempting to start Pixel 5 emulator...")
        
        # List available AVDs
        avds = self._list_available_avds()
        pixel5_avd = None
        
        # Look for Pixel 5 AVD
        for avd in avds:
            if any(identifier.lower() in avd.lower() for identifier in self.pixel5_identifiers):
                pixel5_avd = avd
                break
        
        if pixel5_avd:
            return self._start_emulator(pixel5_avd)
        else:
            print("⚠️ No Pixel 5 AVD found")
            return None
    
    def _list_available_avds(self) -> List[str]:
        """List available Android Virtual Devices"""
        
        try:
            result = subprocess.run(
                ["emulator", "-list-avds"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                avds = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                print(f"📋 Available AVDs: {avds}")
                return avds
            else:
                print("⚠️ Failed to list AVDs")
                return []
                
        except Exception as e:
            print(f"❌ AVD list error: {e}")
            return []
    
    def _start_emulator(self, avd_name: str) -> Optional[str]:
        """Start specific emulator AVD"""
        
        try:
            print(f"🚀 Starting emulator: {avd_name}")
            
            # Start emulator in background
            subprocess.Popen(
                ["emulator", "-avd", avd_name, "-no-snapshot"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for emulator to start
            for attempt in range(60):  # Wait up to 60 seconds
                time.sleep(1)
                devices = self.adb.get_connected_devices()
                
                for device_id in devices:
                    if "emulator" in device_id and self._verify_device_ready(device_id):
                        print(f"✅ Emulator started: {device_id}")
                        return device_id
                
                if attempt % 10 == 0:
                    print(f"⏳ Waiting for emulator... ({attempt}s)")
            
            print("⚠️ Emulator start timeout")
            return None
            
        except Exception as e:
            print(f"❌ Emulator start error: {e}")
            return None
    
    def get_device_capabilities(self, device_id: str) -> Dict[str, bool]:
        """Get device testing capabilities"""
        
        capabilities = {
            "touch": False,
            "multitouch": False,
            "keyboard": False,
            "camera": False,
            "gps": False,
            "sensors": False
        }
        
        try:
            # Check input methods
            input_methods = self.adb.execute_command(device_id, ["shell", "getevent", "-t"])
            if "touch" in input_methods.lower():
                capabilities["touch"] = True
            
            # Check features
            features = self.adb.execute_command(device_id, ["shell", "pm", "list", "features"])
            if "android.hardware.touchscreen.multitouch" in features:
                capabilities["multitouch"] = True
            if "android.hardware.camera" in features:
                capabilities["camera"] = True
            if "android.hardware.location.gps" in features:
                capabilities["gps"] = True
            if "android.hardware.sensor" in features:
                capabilities["sensors"] = True
            
            capabilities["keyboard"] = True  # Assume keyboard input available
            
        except Exception as e:
            print(f"❌ Capabilities check error: {e}")
        
        return capabilities
    
    def optimize_device_for_testing(self, device_id: str) -> bool:
        """Optimize device settings for automated testing"""
        
        try:
            print(f"⚙️ Optimizing device {device_id} for testing...")
            
            optimizations = [
                # Disable animations
                ["shell", "settings", "put", "global", "window_animation_scale", "0.0"],
                ["shell", "settings", "put", "global", "transition_animation_scale", "0.0"],
                ["shell", "settings", "put", "global", "animator_duration_scale", "0.0"],
                
                # Keep screen on during testing
                ["shell", "settings", "put", "system", "screen_off_timeout", "1800000"],
                
                # Disable lock screen
                ["shell", "settings", "put", "secure", "lockscreen.disabled", "1"],
            ]
            
            for command in optimizations:
                result = self.adb.execute_command(device_id, command)
                if result.startswith("Command"):
                    print(f"⚠️ Optimization failed: {' '.join(command)}")
            
            print(f"✅ Device {device_id} optimized for testing")
            return True
            
        except Exception as e:
            print(f"❌ Device optimization error: {e}")
            return False
