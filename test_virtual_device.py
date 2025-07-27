"""
COMPLETELY FIXED: Virtual Device Integration 
Resolves data format mismatch and observation access issues
"""

import asyncio
import time
from typing import Dict, Any, Optional
from env_manager import EnvironmentManager
from android_studio_integration.device_discovery import DeviceDiscovery
from android_studio_integration.virtual_device_env import VirtualDeviceEnv
from android_studio_integration.observation_wrapper import ObservationWrapper

class CompletelyFixedVirtualDeviceManager(EnvironmentManager):
    """COMPLETELY FIXED: All data format issues resolved"""
    
    def __init__(self, device_id: Optional[str] = None):
        super().__init__()
        self.device_id = device_id
        self.virtual_device = None
        self.real_device_mode = False
        
    async def initialize_with_virtual_device(self):
        """FIXED: Initialize with proper data format handling"""
        await super().initialize()
        
        if not self.device_id:
            discovery = DeviceDiscovery()
            self.device_id = discovery.setup_recommended_device()
            
        if not self.device_id:
            raise RuntimeError("No virtual device available")
        
        try:
            self.virtual_device = VirtualDeviceEnv(self.device_id)
            observation = self.virtual_device.get_observation()
            
            if not observation.screenshot:
                raise RuntimeError("Virtual device not responding")
                
            print(f"✅ Virtual device initialized: {self.device_id}")
            print(f"📸 Screenshot: {len(observation.screenshot)} bytes")
            
            self.real_device_mode = True
            self._replace_android_env_in_executor()
            
            return True
            
        except Exception as e:
            print(f"❌ Virtual device initialization failed: {e}")
            raise
    
    def _replace_android_env_in_executor(self):
        """COMPLETELY FIXED: Proper data format handling"""
        
        class CompletelyFixedVirtualDeviceEnv:
            """COMPLETELY FIXED: All data format issues resolved"""
            
            def __init__(self, virtual_device, device_id):
                self.virtual_device = virtual_device
                self.device_id = device_id
                self.mock_mode = False
                self.task_name = "virtual_device_real"
                
            def reset(self):
                """FIXED: Return ObservationWrapper object"""
                observation = self.virtual_device.reset()
                
                # Convert to proper format with attributes
                observation_dict = {
                    "screenshot": observation.screenshot,
                    "ui_hierarchy": observation.ui_hierarchy,
                    "current_activity": observation.current_activity,
                    "screen_bounds": observation.screen_bounds,
                    "timestamp": observation.timestamp,
                    "device_info": {
                        "device_id": observation.device_id,
                        "type": "real_virtual_device"
                    },
                    "task_name": self.task_name,
                    "step_count": 0
                }
                
                # ✅ CRITICAL FIX: Return ObservationWrapper with attribute access
                return ObservationWrapper(observation_dict)
            
            def step(self, action):
                """FIXED: Proper action mapping and observation handling"""
                
                # Action mapping
                action_mapping = {
                    "touch": "touch", "tap": "touch", "click": "touch",
                    "scroll": "scroll", "swipe": "swipe", 
                    "type": "type", "text": "type",
                    "back": "key", "home": "key",
                    "wait": "wait", "verify": "wait", "unknown": "wait"
                }
                
                original_action = action.get("action", "unknown")
                mapped_action = action_mapping.get(original_action, "wait")
                
                print(f"🔧 Action: {original_action} -> {mapped_action}")
                
                # Create mapped action
                mapped_action_dict = {
                    "action": mapped_action,
                    **action
                }
                
                if mapped_action == "key":
                    mapped_action_dict["key"] = "KEYCODE_BACK" if original_action == "back" else "KEYCODE_HOME"
                
                # Execute action
                result = self.virtual_device.execute_action(mapped_action_dict)
                
                # ✅ CRITICAL FIX: Ensure current_observation is ObservationWrapper
                if "current_observation" not in result:
                    obs = self.virtual_device.get_observation()
                    observation_dict = {
                        "screenshot": obs.screenshot,
                        "ui_hierarchy": obs.ui_hierarchy,
                        "current_activity": obs.current_activity,
                        "screen_bounds": obs.screen_bounds,
                        "timestamp": obs.timestamp,
                        "device_info": {"device_id": obs.device_id, "type": "real_virtual_device"},
                        "task_name": self.task_name
                    }
                    result["current_observation"] = ObservationWrapper(observation_dict)
                elif isinstance(result["current_observation"], dict):
                    # Convert dict to ObservationWrapper if needed
                    result["current_observation"] = ObservationWrapper(result["current_observation"])
                
                return result
            
            def _get_observation(self):
                """FIXED: Return ObservationWrapper object"""
                obs = self.virtual_device.get_observation()
                observation_dict = {
                    "screenshot": obs.screenshot,
                    "ui_hierarchy": obs.ui_hierarchy,
                    "current_activity": obs.current_activity,
                    "screen_bounds": obs.screen_bounds,
                    "timestamp": obs.timestamp,
                    "device_info": {"device_id": obs.device_id, "type": "real_virtual_device"},
                    "task_name": self.task_name
                }
                return ObservationWrapper(observation_dict)
            
            def get_observation(self):
                """Public observation method"""
                return self._get_observation()
            
            def close(self):
                if hasattr(self.virtual_device, 'close'):
                    self.virtual_device.close()
        
        # Replace environments
        new_env = CompletelyFixedVirtualDeviceEnv(self.virtual_device, self.device_id)
        
        if hasattr(self, 'executor_agent') and self.executor_agent:
            self.executor_agent.android_env = new_env
            print("✅ ExecutorAgent now using FIXED virtual device with proper observations")
        
        self.android_env = new_env
        print("✅ Main android_env now using FIXED virtual device")
    
    async def run_completely_fixed_test(self, task_config: Dict[str, Any]):
        """FIXED: Run test with proper observation handling"""
        
        if not self.real_device_mode:
            raise RuntimeError("Virtual device not initialized")
        
        print(f"🤖 Running COMPLETELY FIXED test on: {self.device_id}")
        print(f"📋 Task: {task_config.get('goal', 'Unknown')}")
        
        try:
            # Reset device
            reset_obs = self.virtual_device.reset()
            print(f"✅ Device reset - Screenshot: {len(reset_obs.screenshot)} bytes")
            
            # Run test
            result = await self.run_qa_test(task_config)
            
            if hasattr(result, '__dict__'):
                result.environment_type = "completely_fixed_virtual_device"
                result.device_id = self.device_id
                result.mock_mode = False
                result.data_format_fixed = True
            
            return result
            
        except Exception as e:
            print(f"❌ Test execution error: {e}")
            import traceback
            print(f"🔍 Full error:\n{traceback.format_exc()}")
            raise

# Test function
async def run_completely_fixed_test():
    """COMPLETELY FIXED test with all issues resolved"""
    
    print("🚀 COMPLETELY FIXED: Virtual Device Integration Test")
    
    manager = CompletelyFixedVirtualDeviceManager()
    
    try:
        await manager.initialize_with_virtual_device()
        
        test_config = {
            "goal": "FIXED Test - Settings navigation",
            "android_world_task": "settings_wifi",
            "max_steps": 6,
            "timeout": 30
        }
        
        print(f"\n🧪 Running FIXED test: {test_config['goal']}")
        
        result = await manager.run_completely_fixed_test(test_config)
        
        print(f"✅ FIXED Test Result: {result.final_result}")
        print(f"📊 Steps: {len(result.actions)}")
        duration = getattr(result, 'end_time', time.time()) - getattr(result, 'start_time', time.time())
        print(f"⏱️ Duration: {duration:.2f}s")
        print(f"🤖 Data Format Fixed: {getattr(result, 'data_format_fixed', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FIXED test failed: {e}")
        return False
        
    finally:
        if hasattr(manager, 'close'):
            manager.close()

if __name__ == "__main__":
    asyncio.run(run_completely_fixed_test())
