"""
Integration Bridge for Android Studio and QA System
Provides high-level interface for device interaction
"""

import time
from typing import Dict, List, Optional, Any
from .adb_manager import ADBManager
from .device_discovery import DeviceDiscovery
from .emulator_controller import EmulatorController
from .virtual_device_env import VirtualDeviceEnv
from .observation_wrapper import ObservationWrapper

class IntegrationBridge:
    """High-level bridge for Android device integration"""
    
    def __init__(self, preferred_device_id: str = None):
        self.adb = ADBManager()
        self.discovery = DeviceDiscovery()
        self.emulator_controller = EmulatorController()
        self.preferred_device_id = preferred_device_id
        self.active_device = None
        self.virtual_env = None
    
    def initialize_testing_environment(self) -> bool:
        """Initialize complete testing environment"""
        
        try:
            print("🚀 Initializing Android testing environment...")
            
            # Step 1: Discover or setup device
            device_id = self._setup_device()
            if not device_id:
                return False
            
            # Step 2: Create virtual device environment
            self.virtual_env = VirtualDeviceEnv(device_id)
            
            # Step 3: Verify environment
            if not self._verify_environment():
                return False
            
            print(f"✅ Testing environment initialized with device: {device_id}")
            return True
            
        except Exception as e:
            print(f"❌ Environment initialization failed: {e}")
            return False
    
    def _setup_device(self) -> Optional[str]:
        """Setup the best available device for testing"""
        
        # Use preferred device if specified and available
        if self.preferred_device_id:
            devices = self.adb.get_connected_devices()
            if self.preferred_device_id in devices:
                if self.discovery._verify_device_ready(self.preferred_device_id):
                    self.active_device = self.preferred_device_id
                    return self.preferred_device_id
        
        # Use device discovery to find best device
        device_id = self.discovery.setup_recommended_device()
        if device_id:
            self.active_device = device_id
            return device_id
        
        # Try to start an emulator as last resort
        return self._start_fallback_emulator()
    
    def _start_fallback_emulator(self) -> Optional[str]:
        """Start emulator as fallback option"""
        
        print("🚀 Starting fallback emulator...")
        device_id = self.emulator_controller.start_emulator()
        
        if device_id:
            self.active_device = device_id
            return device_id
        
        return None
    
    def _verify_environment(self) -> bool:
        """Verify testing environment is ready"""
        
        if not self.active_device or not self.virtual_env:
            return False
        
        try:
            # Test basic device communication
            test_result = self.adb.execute_command(self.active_device, ["shell", "echo", "test"])
            if test_result != "test":
                print("❌ Device communication test failed")
                return False
            
            # Test virtual environment
            observation = self.virtual_env.get_observation()
            if not observation or not observation.screenshot:
                print("❌ Virtual environment test failed")
                return False
            
            # Optimize device for testing
            self.discovery.optimize_device_for_testing(self.active_device)
            
            return True
            
        except Exception as e:
            print(f"❌ Environment verification failed: {e}")
            return False
    
    def execute_test_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test action through virtual environment"""
        
        if not self.virtual_env:
            raise RuntimeError("Virtual environment not initialized")
        
        try:
            result = self.virtual_env.execute_action(action)
            return result
        except Exception as e:
            print(f"❌ Action execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_current_state(self) -> ObservationWrapper:
        """Get current device state"""
        
        if not self.virtual_env:
            raise RuntimeError("Virtual environment not initialized")
        
        return self.virtual_env.get_observation()
    
    def reset_device_state(self) -> ObservationWrapper:
        """Reset device to initial state"""
        
        if not self.virtual_env:
            raise RuntimeError("Virtual environment not initialized")
        
        try:
            # Go to home screen
            self.adb.press_key(self.active_device, "KEYCODE_HOME")
            time.sleep(1)
            
            # Clear recent apps
            self.adb.press_key(self.active_device, "KEYCODE_APP_SWITCH")
            time.sleep(0.5)
            self.adb.execute_command(self.active_device, ["shell", "input", "swipe", "540", "1170", "540", "100"])
            time.sleep(0.5)
            self.adb.press_key(self.active_device, "KEYCODE_HOME")
            
            # Get observation
            return self.virtual_env.get_observation()
            
        except Exception as e:
            print(f"❌ Device reset failed: {e}")
            return self.virtual_env.get_observation()
    
    def run_test_sequence(self, test_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run sequence of test actions"""
        
        results = []
        
        for i, step in enumerate(test_steps):
            try:
                print(f"🎯 Executing step {i+1}/{len(test_steps)}: {step.get('description', 'Unknown')}")
                
                result = self.execute_test_action(step)
                result["step_number"] = i + 1
                result["step_description"] = step.get('description', 'Unknown')
                
                results.append(result)
                
                # Add delay between steps if specified
                delay = step.get('delay', 1)
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                error_result = {
                    "step_number": i + 1,
                    "step_description": step.get('description', 'Unknown'),
                    "success": False,
                    "error": str(e)
                }
                results.append(error_result)
                print(f"❌ Step {i+1} failed: {e}")
        
        return results
    
    def capture_device_logs(self) -> Dict[str, str]:
        """Capture device logs for debugging"""
        
        logs = {}
        
        if not self.active_device:
            return logs
        
        try:
            # System log
            logcat = self.adb.execute_command(self.active_device, ["logcat", "-d", "-t", "100"])
            logs["logcat"] = logcat
            
            # System info
            system_info = {
                "device_info": self.adb.get_device_info(self.active_device),
                "memory_info": self.adb.execute_command(self.active_device, ["shell", "dumpsys", "meminfo"]),
                "battery_info": self.adb.execute_command(self.active_device, ["shell", "dumpsys", "battery"])
            }
            
            logs["system_info"] = str(system_info)
            
        except Exception as e:
            logs["error"] = f"Failed to capture logs: {e}"
        
        return logs
    
    def cleanup(self):
        """Cleanup resources and stop emulators if needed"""
        
        try:
            if self.virtual_env:
                self.virtual_env.close()
            
            # Stop emulators we started
            if self.active_device and "emulator" in self.active_device:
                self.emulator_controller.stop_emulator(self.active_device)
            
            print("✅ Integration bridge cleanup completed")
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        
        status = {
            "active_device": self.active_device,
            "virtual_env_ready": self.virtual_env is not None,
            "device_connected": False,
            "emulator_managed": False
        }
        
        if self.active_device:
            # Check device connection
            devices = self.adb.get_connected_devices()
            status["device_connected"] = self.active_device in devices
            
            # Check if it's a managed emulator
            emulator_status = self.emulator_controller.get_emulator_status(self.active_device)
            status["emulator_managed"] = emulator_status.get("status") == "running"
            
            # Get device capabilities
            status["device_capabilities"] = self.discovery.get_device_capabilities(self.active_device)
        
        return status
