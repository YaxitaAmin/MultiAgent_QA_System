# run_real_androidworld_tests.py - FIXED VERSION
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from real_androidworld_runner import RealAndroidWorldRunner

async def main():
    """Main function to run real AndroidWorld tests"""
    
    print("🚀 Starting Real AndroidWorld Testing")
    print("=" * 50)
    
    # Check environment setup
    print("Checking environment setup...")
    
    # Check ADB availability
    import subprocess
    try:
        result = subprocess.run(['adb', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ADB is available")
            print(f"ADB Version: {result.stdout.split()[4]}")
            
            # Check connected devices
            devices_result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            if devices_result.returncode == 0:
                lines = devices_result.stdout.strip().split('\n')[1:]
                connected_devices = [line.split('\t')[0] for line in lines if '\tdevice' in line]
                print(f"Connected devices: {connected_devices}")
                
                if not connected_devices:
                    print("⚠️  No Android devices connected!")
                    print("Please connect an Android device or start an emulator")
                    # Don't return - allow mock mode testing
            else:
                print("❌ Could not check connected devices")
                connected_devices = []
        else:
            print("❌ ADB not available in PATH")
            connected_devices = []
    except FileNotFoundError:
        print("❌ ADB not found - please install Android SDK Platform Tools")
        connected_devices = []
    
    # Check AndroidEnv availability using our wrapper
    try:
        from core.android_env_wrapper import ANDROID_ENV_AVAILABLE
        if ANDROID_ENV_AVAILABLE:
            print("✅ AndroidEnv package is available")
        else:
            print("⚠️ AndroidEnv installed but running in mock mode (Windows limitation)")
            print("For real AndroidWorld testing, use WSL2 or Linux")
    except ImportError:
        print("❌ Core wrapper not available")
        return
    
    # Check API keys
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not set - will use mock LLM")
    else:
        print("✅ Gemini API key configured")
    
    print("\n" + "=" * 50)
    print("Starting Real AndroidWorld Test Suite")
    print("=" * 50)
    
    # Define test tasks
    test_tasks = [
        "Test turning Wi-Fi on and off",
        "Navigate to Bluetooth settings", 
        "Open Calculator and perform calculation",
        "Check device storage",
        "Test airplane mode toggle"
    ]
    
    # Initialize runner
    runner = RealAndroidWorldRunner(
        use_real_device=True,
        adb_device_serial=connected_devices[0] if connected_devices else None
    )
    
    try:
        # Run the tests
        results = await runner.run_real_androidworld_tests(test_tasks)
        
        # Display results
        print("\n" + "🎯 REAL ANDROIDWORLD TEST RESULTS" + "\n")
        print("=" * 50)
        
        summary = results["summary"]
        print(f"Total Tasks: {summary['total_tasks']}")
        print(f"Successful Tasks: {summary['successful_tasks']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Execution Time: {summary['average_execution_time']:.2f}s")
        
        # Check if we have performance metrics
        if 'performance_score' in summary:
            print(f"Performance Score: {summary['performance_score']:.2f}")
        if 'stability_score' in summary:
            print(f"Environment Stability: {summary['stability_score']:.2f}")
        
        # Environment breakdown
        if 'environment_breakdown' in summary:
            breakdown = summary["environment_breakdown"]
            print(f"\nEnvironment Breakdown:")
            print(f"- Real Tests: {breakdown['real_tests']}")
            print(f"- Mock Tests: {breakdown['mock_tests']}")
            print(f"- Real Percentage: {breakdown['real_percentage']:.1%}")
        
        # Recommendations
        if summary.get("recommendations"):
            print(f"\nRecommendations:")
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"{i}. {rec}")
        
        print(f"\n✅ Real AndroidWorld testing completed successfully!")
        print(f"📁 Results saved to: real_androidworld_results/")
        
    except Exception as e:
        print(f"\n❌ Real AndroidWorld testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
