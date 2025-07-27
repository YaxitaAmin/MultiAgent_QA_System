"""
Fixed ADB Manager with proper string handling for Pixel 5
Handles all ADB operations with proper command construction
"""

import subprocess
import time
import json
from typing import List, Optional, Union

class ADBManager:
    """Fixed ADB Manager with proper command handling and Pixel 5 support"""
    
    def __init__(self):
        self.adb_path = "adb"
        self.default_timeout = 30
        
    def execute_command(self, device_id: str, command: Union[str, List[str]]) -> str:
        """Execute ADB command with fixed string handling"""
        
        try:
            # ✅ CRITICAL FIX: Proper command construction
            if isinstance(command, list):
                command_parts = [str(part) for part in command]
            else:
                command_parts = str(command).split()
            
            # Build full command with proper string conversion
            full_command = [self.adb_path, "-s", str(device_id)] + command_parts
            
            print(f"🔧 ADB Command: {' '.join(full_command)}")
            
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=self.default_timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                print(f"⚠️ ADB command failed: {error_msg}")
                return f"Command failed: {error_msg}"  # ✅ FIX: Consistent error format
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            print(f"⏰ ADB command timed out after {self.default_timeout}s")
            return "Command timed out"
        except Exception as e:
            print(f"❌ ADB command error: {e}")
            return f"Command error: {str(e)}"  # ✅ FIX: Consistent error format
    
    def get_screenshot(self, device_id: str) -> bytes:
        """Get device screenshot with proper error handling"""
        
        try:
            # Use exec-out for direct binary capture
            full_command = [self.adb_path, "-s", str(device_id), "exec-out", "screencap", "-p"]
            
            result = subprocess.run(
                full_command,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                print(f"📸 Screenshot captured: {len(result.stdout)} bytes")
                return result.stdout
            else:
                print(f"⚠️ Screenshot capture failed: {result.stderr if result.stderr else 'Unknown error'}")
                return b""
                
        except Exception as e:
            print(f"❌ Screenshot error: {e}")
            return b""
    
    def get_ui_hierarchy(self, device_id: str) -> str:
        """Get UI hierarchy with proper XML handling"""
        
        try:
            # Dump UI hierarchy to stdout
            command = ["shell", "uiautomator", "dump", "/dev/stdout"]
            result = self.execute_command(device_id, command)
            
            if result and not result.startswith("Command"):
                return result
            else:
                print(f"⚠️ UI hierarchy dump failed")
                return ""
                
        except Exception as e:
            print(f"❌ UI hierarchy error: {e}")
            return ""
    
    def get_current_activity(self, device_id: str) -> str:
        """Get current foreground activity with fixed pipe handling"""
        
        try:
            # ✅ FIX: Avoid pipe in command, use alternative approach
            command = ["shell", "dumpsys", "window", "displays"]
            result = self.execute_command(device_id, command)
            
            if result and "mCurrentFocus" in result:
                # Extract activity name from focus info
                lines = result.split('\n')
                for line in lines:
                    if "mCurrentFocus" in line:
                        parts = line.split()
                        for part in parts:
                            if "/" in part and "}" not in part and "." in part:
                                return part.strip()
            
            # Alternative method if first fails
            command2 = ["shell", "dumpsys", "activity", "recents"]
            result2 = self.execute_command(device_id, command2)
            
            if result2 and "Recent" in result2:
                lines = result2.split('\n')
                for line in lines:
                    if "intent=" in line and "cmp=" in line:
                        # Extract component name
                        start = line.find("cmp=") + 4
                        end = line.find(" ", start)
                        if end == -1:
                            end = line.find("}", start)
                        if start > 3 and end > start:
                            return line[start:end].strip()
                        
            return "unknown"
            
        except Exception as e:
            print(f"❌ Current activity error: {e}")
            return "unknown"
    
    def get_device_info(self, device_id: str) -> dict:
        """Get comprehensive device information"""
        
        info = {
            "device_id": device_id,
            "model": "unknown",
            "android_version": "unknown",
            "screen_resolution": "unknown",
            "density": "unknown"
        }
        
        try:
            # Get device model
            model = self.execute_command(device_id, ["shell", "getprop", "ro.product.model"])
            if model and not model.startswith("Command"):
                info["model"] = model
            
            # Get Android version
            version = self.execute_command(device_id, ["shell", "getprop", "ro.build.version.release"])
            if version and not version.startswith("Command"):
                info["android_version"] = version
            
            # Get screen resolution
            wm_size = self.execute_command(device_id, ["shell", "wm", "size"])
            if wm_size and "Physical size:" in wm_size:
                resolution = wm_size.split("Physical size:")[-1].strip()
                info["screen_resolution"] = resolution
            
            # Get screen density
            density = self.execute_command(device_id, ["shell", "wm", "density"])
            if density and "Physical density:" in density:
                density_value = density.split("Physical density:")[-1].strip()
                info["density"] = density_value
                
        except Exception as e:
            print(f"❌ Device info error: {e}")
        
        return info
    
    def tap(self, device_id: str, x: int, y: int) -> bool:
        """Execute tap at coordinates"""
        
        try:
            command = ["shell", "input", "tap", str(x), str(y)]
            result = self.execute_command(device_id, command)
            return not result.startswith("Command")
        except Exception as e:
            print(f"❌ Tap error: {e}")
            return False
    
    def swipe(self, device_id: str, x1: int, y1: int, x2: int, y2: int, duration: int = 500) -> bool:
        """Execute swipe gesture"""
        
        try:
            command = ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration)]
            result = self.execute_command(device_id, command)
            return not result.startswith("Command")
        except Exception as e:
            print(f"❌ Swipe error: {e}")
            return False
    
    def type_text(self, device_id: str, text: str) -> bool:
        """Type text with proper escaping"""
        
        try:
            # Escape special characters for shell
            escaped_text = text.replace(' ', '%s').replace("'", "\\'").replace('"', '\\"')
            command = ["shell", "input", "text", escaped_text]
            result = self.execute_command(device_id, command)
            return not result.startswith("Command")
        except Exception as e:
            print(f"❌ Type text error: {e}")
            return False
    
    def press_key(self, device_id: str, key_code: str) -> bool:
        """Press hardware key"""
        
        try:
            command = ["shell", "input", "keyevent", key_code]
            result = self.execute_command(device_id, command)
            return not result.startswith("Command")
        except Exception as e:
            print(f"❌ Key press error: {e}")
            return False
    
    def get_connected_devices(self) -> List[str]:
        """Get list of connected devices"""
        
        try:
            result = subprocess.run(
                [self.adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                devices = []
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip() and '\tdevice' in line:
                        device_id = line.split('\t')[0].strip()
                        devices.append(device_id)
                return devices
            
            return []
            
        except Exception as e:
            print(f"❌ Get devices error: {e}")
            return []
    
    def wait_for_device(self, device_id: str, timeout: int = 30) -> bool:
        """Wait for device to be ready"""
        
        try:
            command = [self.adb_path, "-s", device_id, "wait-for-device"]
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=timeout
            )
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Wait for device error: {e}")
            return False
    
    # ✅ NEW: Additional utility methods for better functionality
    
    def is_device_connected(self, device_id: str) -> bool:
        """Check if specific device is connected"""
        
        connected_devices = self.get_connected_devices()
        return device_id in connected_devices
    
    def get_device_state(self, device_id: str) -> str:
        """Get device state (device, offline, unauthorized, etc.)"""
        
        try:
            result = subprocess.run(
                [self.adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip() and device_id in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            return parts[1].strip()
                            
            return "unknown"
            
        except Exception as e:
            print(f"❌ Get device state error: {e}")
            return "error"
    
    def clear_logcat(self, device_id: str) -> bool:
        """Clear device logcat"""
        
        try:
            command = ["shell", "logcat", "-c"]
            result = self.execute_command(device_id, command)
            return not result.startswith("Command")
        except Exception as e:
            print(f"❌ Clear logcat error: {e}")
            return False
    
    def get_logcat(self, device_id: str, lines: int = 100) -> str:
        """Get device logcat"""
        
        try:
            command = ["logcat", "-d", "-t", str(lines)]
            result = self.execute_command(device_id, command)
            return result if not result.startswith("Command") else ""
        except Exception as e:
            print(f"❌ Get logcat error: {e}")
            return ""
    
    def install_apk(self, device_id: str, apk_path: str) -> bool:
        """Install APK on device"""
        
        try:
            command = [self.adb_path, "-s", device_id, "install", "-r", apk_path]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60  # APK installation can take longer
            )
            
            return result.returncode == 0 and "Success" in result.stdout
            
        except Exception as e:
            print(f"❌ APK install error: {e}")
            return False
    
    def uninstall_package(self, device_id: str, package_name: str) -> bool:
        """Uninstall package from device"""
        
        try:
            command = ["shell", "pm", "uninstall", package_name]
            result = self.execute_command(device_id, command)
            return "Success" in result
        except Exception as e:
            print(f"❌ Package uninstall error: {e}")
            return False
    
    def start_activity(self, device_id: str, activity_name: str) -> bool:
        """Start specific activity"""
        
        try:
            command = ["shell", "am", "start", "-n", activity_name]
            result = self.execute_command(device_id, command)
            return not result.startswith("Command")
        except Exception as e:
            print(f"❌ Start activity error: {e}")
            return False
    
    def force_stop_app(self, device_id: str, package_name: str) -> bool:
        """Force stop application"""
        
        try:
            command = ["shell", "am", "force-stop", package_name]
            result = self.execute_command(device_id, command)
            return not result.startswith("Command")
        except Exception as e:
            print(f"❌ Force stop error: {e}")
            return False
