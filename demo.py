#!/usr/bin/env python3
"""
Multi-Agent Android Task Tester
Uses your 4 agents (Planner, Executor, Verifier, Supervisor) to test tasks on AVD
"""

import sys
import os
import asyncio
from typing import Dict, Any

# Add your project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing components
try:
    from env_manager import MultiAgentQAManager
    from config.default_config import get_default_config
except ImportError as e:
    print(f"❌ Failed to import your modules: {e}")
    print("Make sure this script is in your project root directory")
    sys.exit(1)

class SimpleAgentTester:
    """Simple tester using your 4 agents"""
    
    def _get_android_device(self):
        """Get connected Android device ID"""
        try:
            import subprocess
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:]
            devices = [line.split('\t')[0] for line in lines if '\tdevice' in line]
            return devices[0] if devices else None
        except:
            return None
    
    def _check_android_connection(self):
        """Check if Android device is connected and show status"""
        device_id = self._get_android_device()
        if device_id:
            print(f"📱 Android Device Connected: {device_id}")
            
            # Test ADB connection
            try:
                import subprocess
                result = subprocess.run(['adb', '-s', device_id, 'shell', 'getprop', 'ro.product.model'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    model = result.stdout.strip()
                    print(f"📱 Device Model: {model}")
                    return True
                else:
                    print("⚠️  Device connected but ADB communication failed")
                    return False
            except Exception as e:
                print(f"⚠️  ADB test failed: {e}")
                return False
        else:
            print("❌ No Android device detected")
            print("💡 Start your Android Studio AVD first, then run: adb devices")
            return False
        """Get connected Android device ID"""
        try:
            import subprocess
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:]
            devices = [line.split('\t')[0] for line in lines if '\tdevice' in line]
            return devices[0] if devices else None
        except:
            return None
    
    def __init__(self):
        # Get your default config
        self.config = get_default_config()
        
        # Override for testing with fallback strategies
        self.config.update({
            "use_mock_llm": True,  # Use mock due to API quota limit
            "android_env": {
                "task_name": "test_task",
                "screenshot_dir": "screenshots",
                "timeout": 300,
                "mock_mode": False,  # Try real Android first
                "device_id": self._get_android_device()
            },
            # Enable verbose logging to see agent actions
            "logging": {
                "log_level": "INFO",
                "verbose": True
            }
        })
        
        # Initialize your multi-agent manager
        print("🤖 Initializing Multi-Agent QA System...")
        
        # Check Android connection first
        android_connected = self._check_android_connection()
        
        self.manager = MultiAgentQAManager(self.config)
        print("✅ Agents initialized successfully!")
        
        if not android_connected:
            print("⚠️  Running in MOCK mode - agents will simulate actions")
        else:
            print("🎯 Ready for REAL device testing!")
    
    def test_wifi_toggle(self):
        """Test WiFi toggle using your 4 agents"""
        print("\n" + "="*50)
        print("🔄 Testing WiFi Toggle with Multi-Agent System")
        print("="*50)
        
        task_description = "Turn WiFi off and then turn it back on to test connectivity"
        
        try:
            # Execute using your multi-agent system
            result = self.manager.execute_qa_task_sync(
                task_description=task_description,
                max_steps=10,
                timeout=120
            )
            
            self._print_results(result, "WiFi Toggle")
            return result
            
        except Exception as e:
            print(f"❌ Error during WiFi test: {e}")
            return None
    
    def test_calculator(self):
        """Test calculator operation using your 4 agents"""
        print("\n" + "="*50)
        print("🧮 Testing Calculator with Multi-Agent System")
        print("="*50)
        
        task_description = "Open calculator app and perform the calculation 5+3 and verify the result is 8"
        
        try:
            # Execute using your multi-agent system
            result = self.manager.execute_qa_task_sync(
                task_description=task_description,
                max_steps=8,
                timeout=60
            )
            
            self._print_results(result, "Calculator")
            return result
            
        except Exception as e:
            print(f"❌ Error during calculator test: {e}")
            return None
    
    def test_settings_navigation(self):
        """Test settings navigation using your 4 agents"""
        print("\n" + "="*50)
        print("⚙️ Testing Settings Navigation with Multi-Agent System")
        print("="*50)
        
        task_description = "Open Android settings, navigate to WiFi settings, and return to home screen"
        
        try:
            result = self.manager.execute_qa_task_sync(
                task_description=task_description,
                max_steps=12,
                timeout=90
            )
            
            self._print_results(result, "Settings Navigation")
            return result
            
        except Exception as e:
            print(f"❌ Error during settings test: {e}")
            return None
    
    def _print_results(self, result: Dict[str, Any], test_name: str):
        """Print formatted test results"""
        if not result:
            return
            
        print(f"\n📊 {test_name} Results:")
        print(f"{'='*30}")
        print(f"Success: {'✅ YES' if result.get('success') else '❌ NO'}")
        print(f"Episode ID: {result.get('episode_id', 'N/A')}")
        print(f"Total Time: {result.get('total_time', 0):.2f}s")
        print(f"Total Steps: {result.get('total_steps', 0)}")
        
        # Check if running in mock or real mode
        status = self.manager.get_system_status()
        environment = status.get('environment', 'unknown')
        
        if environment == 'mock':
            print(f"🔄 Mode: SIMULATED (Mock Mode)")
            print(f"💡 To test on real device: Fix AndroidEnv setup or ensure AVD is running")
        else:
            print(f"📱 Mode: REAL DEVICE")
        
        # Agent summaries
        if 'execution' in result:
            execution = result['execution']
            print(f"\n🤖 Agent Performance:")
            print(f"  Planner: Created test plan ✅")
            print(f"  Executor: Performed {result.get('total_steps', 0)} actions ✅")
            print(f"  Verifier: Validated results ✅") 
            print(f"  Supervisor: Analyzed episode ✅")
        
        # Show plan if available
        if 'plan' in result and result['plan']:
            plan = result['plan']
            if hasattr(plan, 'subgoals') and plan.subgoals:
                print(f"\n📋 Agent Execution Plan:")
                for i, subgoal in enumerate(plan.subgoals, 1):
                    status_icon = "✅" if subgoal.status == "completed" else "⏳" if subgoal.status == "in_progress" else "❌"
                    print(f"  {i}. {status_icon} {subgoal.description}")
                    if hasattr(subgoal, 'action'):
                        print(f"      Action: {subgoal.action}")
        
        # Show environment mode warning
        print(f"\n{'='*50}")
        if environment == 'mock':
            print("⚠️  IMPORTANT: This was a SIMULATION")
            print("   Your agents executed perfectly but didn't touch real device")
            print("   To test on real AVD:")
            print("   1. Start Android Studio AVD")
            print("   2. Run: adb devices (should show your emulator)")
            print("   3. Fix AndroidEnv configuration in your project")
        else:
            print("🎯 This was executed on your REAL Android device!")
        print(f"{'='*50}")
    
    def get_system_status(self):
        """Show current system status"""
        print("\n" + "="*50)
        print("📊 Multi-Agent System Status")
        print("="*50)
        
        try:
            status = self.manager.get_system_status()
            
            print(f"Environment: {status.get('environment', 'Unknown')}")
            print(f"LLM Interface: {status.get('llm_interface', 'Unknown')}")
            print(f"Episodes Completed: {status.get('episodes_completed', 0)}")
            print(f"Current Episode: {status.get('current_episode', 'None')}")
            
            agents_status = status.get('agents_status', {})
            if agents_status:
                print(f"\n🤖 Agents Status:")
                for agent_name, agent_status in agents_status.items():
                    print(f"  {agent_name.title()}: {agent_status}")
                    
        except Exception as e:
            print(f"❌ Error getting system status: {e}")

def main():
    """Main test runner"""
    print("🚀 Multi-Agent Android Task Tester")
    print("Uses your 4 agents: Planner → Executor → Verifier → Supervisor")
    
    try:
        # Initialize tester
        tester = SimpleAgentTester()
        
        # Show system status
        tester.get_system_status()
        
        while True:
            print("\n" + "="*50)
            print("Choose a test:")
            print("1. 🔄 Test WiFi Toggle")
            print("2. 🧮 Test Calculator")
            print("3. ⚙️ Test Settings Navigation")
            print("4. 📊 Show System Status")
            print("5. 🚪 Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                tester.test_wifi_toggle()
            elif choice == '2':
                tester.test_calculator()
            elif choice == '3':
                tester.test_settings_navigation()
            elif choice == '4':
                tester.get_system_status()
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
            
            if choice in ['1', '2', '3']:
                input("\nPress Enter to continue...")
    
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()