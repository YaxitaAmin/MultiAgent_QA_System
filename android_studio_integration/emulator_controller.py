"""
Emulator Controller for Android Studio Integration
Handles emulator lifecycle and configuration management
"""

import subprocess
import time
import os
import signal
from typing import Dict, List, Optional, Tuple
from .adb_manager import ADBManager

class EmulatorController:
    """Enhanced emulator controller with Pixel 5 support"""
    
    def __init__(self):
        self.adb = ADBManager()
        self.running_emulators = {}
        self.pixel5_config = {
            "avd_name": "Pixel_5_API_33",
            "api_level": "33",
            "target": "android-33",
            "abi": "x86_64",
            "device": "pixel_5",
            "ram": "2048",
            "heap": "256",
            "data_partition": "6G"
        }
    
    def create_pixel5_avd(self, avd_name: str = None) -> bool:
        """Create a new Pixel 5 AVD for testing"""
        
        if not avd_name:
            avd_name = self.pixel5_config["avd_name"]
        
        try:
            print(f"🔧 Creating Pixel 5 AVD: {avd_name}")
            
            # Check if AVD already exists
            if self._avd_exists(avd_name):
                print(f"✅ AVD {avd_name} already exists")
                return True
            
            # Create AVD command
            create_cmd = [
                "avdmanager", "create", "avd",
                "-n", avd_name,
                "-k", f"system-images;android-{self.pixel5_config['api_level']};google_apis;{self.pixel5_config['abi']}",
                "-d", self.pixel5_config["device"],
                "--force"
            ]
            
            # Execute AVD creation
            process = subprocess.Popen(
                create_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Provide default responses to prompts
            stdout, stderr = process.communicate(input="\n")
            
            if process.returncode == 0:
                print(f"✅ AVD {avd_name} created successfully")
                
                # Configure AVD
                self._configure_avd(avd_name)
                return True
            else:
                print(f"❌ AVD creation failed: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ AVD creation error: {e}")
            return False
    
    def _avd_exists(self, avd_name: str) -> bool:
        """Check if AVD exists"""
        
        try:
            result = subprocess.run(
                ["avdmanager", "list", "avd", "-c"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                avds = result.stdout.strip().split('\n')
                return avd_name in avds
            
            return False
            
        except Exception as e:
            print(f"❌ AVD check error: {e}")
            return False
    
    def _configure_avd(self, avd_name: str):
        """Configure AVD with optimal settings for testing"""
        
        try:
            # Find AVD config file
            avd_dir = os.path.expanduser(f"~/.android/avd/{avd_name}.avd")
            config_file = os.path.join(avd_dir, "config.ini")
            
            if not os.path.exists(config_file):
                print(f"⚠️ AVD config file not found: {config_file}")
                return
            
            # Read current config
            with open(config_file, 'r') as f:
                config_lines = f.readlines()
            
            # Update configuration
            new_config = []
            updated_keys = set()
            
            for line in config_lines:
                if '=' in line:
                    key = line.split('=')[0].strip()
                    
                    # Override specific settings
                    if key == "hw.ramSize":
                        new_config.append(f"hw.ramSize={self.pixel5_config['ram']}\n")
                        updated_keys.add(key)
                    elif key == "vm.heapSize":
                        new_config.append(f"vm.heapSize={self.pixel5_config['heap']}\n")
                        updated_keys.add(key)
                    elif key == "disk.dataPartition.size":
                        new_config.append(f"disk.dataPartition.size={self.pixel5_config['data_partition']}\n")
                        updated_keys.add(key)
                    elif key == "hw.gpu.enabled":
                        new_config.append("hw.gpu.enabled=yes\n")
                        updated_keys.add(key)
                    elif key == "hw.gpu.mode":
                        new_config.append("hw.gpu.mode=auto\n")
                        updated_keys.add(key)
                    else:
                        new_config.append(line)
                else:
                    new_config.append(line)
            
            # Add missing configurations
            required_configs = {
                "hw.ramSize": self.pixel5_config['ram'],
                "vm.heapSize": self.pixel5_config['heap'],
                "disk.dataPartition.size": self.pixel5_config['data_partition'],
                "hw.gpu.enabled": "yes",
                "hw.gpu.mode": "auto",
                "hw.keyboard": "yes",
                "hw.accelerometer": "yes",
                "hw.gps": "yes"
            }
            
            for key, value in required_configs.items():
                if key not in updated_keys:
                    new_config.append(f"{key}={value}\n")
            
            # Write updated config
            with open(config_file, 'w') as f:
                f.writelines(new_config)
            
            print(f"✅ AVD {avd_name} configured for testing")
            
        except Exception as e:
            print(f"❌ AVD configuration error: {e}")
    
    def start_emulator(self, avd_name: str = None, **kwargs) -> Optional[str]:
        """Start emulator with optimal settings"""
        
        if not avd_name:
            avd_name = self.pixel5_config["avd_name"]
        
        # Check if AVD exists, create if not
        if not self._avd_exists(avd_name):
            if not self.create_pixel5_avd(avd_name):
                return None
        
        try:
            print(f"🚀 Starting emulator: {avd_name}")
            
            # Build emulator command with optimal flags
            emulator_cmd = [
                "emulator",
                "-avd", avd_name,
                "-no-snapshot",
                "-no-snapshot-save",
                "-no-boot-anim",
                "-memory", self.pixel5_config['ram'],
                "-cores", "2",
                "-gpu", "auto"
            ]
            
            # Add custom parameters
            if kwargs.get("headless", False):
                emulator_cmd.extend(["-no-window", "-no-audio"])
            
            if kwargs.get("wipe_data", False):
                emulator_cmd.append("-wipe-data")
            
            # Start emulator process
            process = subprocess.Popen(
                emulator_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait for emulator to start
            device_id = self._wait_for_emulator_boot(process)
            
            if device_id:
                self.running_emulators[device_id] = {
                    "avd_name": avd_name,
                    "process": process,
                    "start_time": time.time()
                }
                
                # Optimize for testing
                self._optimize_emulator(device_id)
                
                print(f"✅ Emulator started: {device_id}")
                return device_id
            else:
                print("❌ Emulator failed to start")
                process.terminate()
                return None
                
        except Exception as e:
            print(f"❌ Emulator start error: {e}")
            return None
    
    def _wait_for_emulator_boot(self, process: subprocess.Popen, timeout: int = 120) -> Optional[str]:
        """Wait for emulator to fully boot"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            if process.poll() is not None:
                print("❌ Emulator process terminated")
                return None
            
            # Check for connected devices
            devices = self.adb.get_connected_devices()
            
            for device_id in devices:
                if "emulator" in device_id:
                    # Check if device is fully booted
                    if self._is_device_booted(device_id):
                        return device_id
            
            time.sleep(2)
            
            # Progress indicator
            elapsed = int(time.time() - start_time)
            if elapsed % 15 == 0:
                print(f"⏳ Waiting for emulator boot... ({elapsed}s)")
        
        print(f"⚠️ Emulator boot timeout after {timeout}s")
        return None
    
    def _is_device_booted(self, device_id: str) -> bool:
        """Check if device has fully booted"""
        
        try:
            # Check boot completion
            boot_completed = self.adb.execute_command(
                device_id, 
                ["shell", "getprop", "sys.boot_completed"]
            )
            
            if boot_completed.strip() != "1":
                return False
            
            # Check if package manager is ready
            pm_ready = self.adb.execute_command(
                device_id,
                ["shell", "pm", "list", "packages"]
            )
            
            return not pm_ready.startswith("Command")
            
        except Exception:
            return False
    
    def _optimize_emulator(self, device_id: str):
        """Apply testing optimizations to running emulator"""
        
        try:
            print(f"⚙️ Optimizing emulator {device_id}")
            
            optimizations = [
                # Disable animations
                ["shell", "settings", "put", "global", "window_animation_scale", "0.0"],
                ["shell", "settings", "put", "global", "transition_animation_scale", "0.0"],
                ["shell", "settings", "put", "global", "animator_duration_scale", "0.0"],
                
                # Performance settings
                ["shell", "settings", "put", "system", "screen_off_timeout", "1800000"],
                ["shell", "settings", "put", "secure", "lockscreen.disabled", "1"],
                
                # Wake device
                ["shell", "input", "keyevent", "KEYCODE_WAKEUP"],
                ["shell", "input", "keyevent", "KEYCODE_MENU"],
                ["shell", "input", "keyevent", "KEYCODE_HOME"]
            ]
            
            for command in optimizations:
                self.adb.execute_command(device_id, command)
                time.sleep(0.5)
            
            print(f"✅ Emulator {device_id} optimized")
            
        except Exception as e:
            print(f"❌ Emulator optimization error: {e}")
    
    def stop_emulator(self, device_id: str) -> bool:
        """Stop running emulator"""
        
        try:
            if device_id in self.running_emulators:
                emulator_info = self.running_emulators[device_id]
                process = emulator_info["process"]
                
                print(f"🛑 Stopping emulator: {device_id}")
                
                # Try graceful shutdown first
                self.adb.execute_command(device_id, ["emu", "kill"])
                time.sleep(5)
                
                # Force kill if still running
                if process.poll() is None:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    else:
                        process.terminate()
                    
                    time.sleep(3)
                    
                    if process.poll() is None:
                        if os.name != 'nt':
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        else:
                            process.kill()
                
                del self.running_emulators[device_id]
                print(f"✅ Emulator {device_id} stopped")
                return True
            else:
                print(f"⚠️ Emulator {device_id} not managed by this controller")
                return False
                
        except Exception as e:
            print(f"❌ Emulator stop error: {e}")
            return False
    
    def stop_all_emulators(self) -> bool:
        """Stop all running emulators"""
        
        success = True
        running_devices = list(self.running_emulators.keys())
        
        for device_id in running_devices:
            if not self.stop_emulator(device_id):
                success = False
        
        return success
    
    def get_emulator_status(self, device_id: str) -> Dict[str, any]:
        """Get emulator status and performance info"""
        
        if device_id not in self.running_emulators:
            return {"status": "not_managed"}
        
        emulator_info = self.running_emulators[device_id]
        process = emulator_info["process"]
        
        status = {
            "status": "running" if process.poll() is None else "stopped",
            "avd_name": emulator_info["avd_name"],
            "uptime": time.time() - emulator_info["start_time"],
            "device_ready": self._is_device_booted(device_id)
        }
        
        return status
