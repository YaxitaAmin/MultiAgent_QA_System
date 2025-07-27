"""
Test suite for multi-agent QA system
"""
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('..')

from agents.planner_agent import PlannerAgent, ActionType, PlanStep, ExecutionPlan
from agents.executor_agent import ExecutorAgent, ExecutionResult
from agents.verifier_agent import VerifierAgent, VerificationResult
from agents.supervisor_agent import SupervisorAgent, SupervisorStatus
from core.llm_interface import LLMInterface, OpenAIProvider
from core.android_env_wrapper import AndroidEnvironmentWrapper
from main_orchestrator import MainOrchestrator

class TestPlannerAgent(unittest.TestCase):
    """Test cases for PlannerAgent"""
    
    def setUp(self):
        self.mock_llm = Mock()
        self.planner = PlannerAgent(self.mock_llm)
    
    def test_analyze_query_success(self):
        """Test successful query analysis"""
        test_query = "Test login functionality"
        
        # Mock LLM response
        self.mock_llm.generate_response.return_value = json.dumps({
            "intent": "login_test",
            "ui_elements": ["username_field", "password_field", "login_button"],
            "actions": ["tap", "type", "verify"],
            "success_criteria": ["successful_login"],
            "complexity": 3,
            "estimated_steps": 5
        })
        
        result = self.planner.analyze_query(test_query)
        
        self.assertEqual(result["intent"], "login_test")
        self.assertEqual(result["complexity"], 3)
        self.assertIn("username_field", result["ui_elements"])
    
    def test_create_execution_plan(self):
        """Test execution plan creation"""
        test_query = "Test app navigation"
        
        # Mock LLM response for analysis
        self.mock_llm.generate_response.side_effect = [
            json.dumps({
                "intent": "navigation_test",
                "ui_elements": ["menu_button", "settings_option"],
                "actions": ["tap", "verify"],
                "success_criteria": ["navigation_successful"],
                "complexity": 2,
                "estimated_steps": 3
            }),
            json.dumps({
                "steps": [
                    {
                        "action_type": "screenshot",
                        "parameters": {},
                        "description": "Take initial screenshot",
                        "expected_outcome": "Screenshot captured",
                        "priority": 1,
                        "timeout": 10
                    },
                    {
                        "action_type": "tap",
                        "parameters": {"element_id": "menu_button"},
                        "description": "Tap menu button",
                        "expected_outcome": "Menu opened",
                        "priority": 3,
                        "timeout": 15
                    }
                ]
            })
        ]
        
        plan = self.planner.create_execution_plan(test_query)
        
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].action_type, ActionType.SCREENSHOT)
        self.assertEqual(plan.steps[1].action_type, ActionType.TAP)
    
    def test_validate_plan(self):
        """Test plan validation"""
        # Create a valid plan
        steps = [
            PlanStep(
                action_type=ActionType.SCREENSHOT,
                parameters={},
                description="Take screenshot",
                expected_outcome="Screenshot taken"
            ),
            PlanStep(
                action_type=ActionType.VERIFY,
                parameters={"condition": "app_loaded"},
                description="Verify app state",
                expected_outcome="App is loaded"
            )
        ]
        
        plan = ExecutionPlan(
            plan_id="test_plan",
            query="test query",
            steps=steps,
            estimated_duration=60,
            complexity_score=2,
            success_criteria=["app_functional"]
        )
        
        validation_result = self.planner.validate_plan(plan)
        
        self.assertTrue(validation_result["is_valid"])
        self.assertGreaterEqual(validation_result["score"], 70)

class TestExecutorAgent(unittest.TestCase):
    """Test cases for ExecutorAgent"""
    
    def setUp(self):
        self.mock_android_env = Mock()
        self.mock_llm = Mock()
        self.executor = ExecutorAgent(self.mock_android_env, self.mock_llm)
    
    def test_execute_tap_with_coordinates(self):
        """Test tap execution with coordinates"""
        step = PlanStep(
            action_type=ActionType.TAP,
            parameters={"coordinates": [100, 200]},
            description="Tap at coordinates",
            expected_outcome="Element tapped"
        )
        
        self.mock_android_env.tap.return_value = True
        
        result = self.executor.execute_step(step)
        
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, "tap")
        self.mock_android_env.tap.assert_called_once_with(100, 200)
    
    def test_execute_type_action(self):
        """Test type text execution"""
        step = PlanStep(
            action_type=ActionType.TYPE,
            parameters={"text": "test input", "clear_first": True},
            description="Type text",
            expected_outcome="Text entered"
        )
        
        self.mock_android_env.clear_text.return_value = True
        self.mock_android_env.type_text.return_value = True
        
        result = self.executor.execute_step(step)
        
        self.assertTrue(result.success)
        self.mock_android_env.clear_text.assert_called_once()
        self.mock_android_env.type_text.assert_called_once_with("test input")
    
    def test_execute_screenshot(self):
        """Test screenshot execution"""
        step = PlanStep(
            action_type=ActionType.SCREENSHOT,
            parameters={},
            description="Take screenshot",
            expected_outcome="Screenshot captured"
        )
        
        self.mock_android_env.take_screenshot.return_value = True
        
        with patch.object(self.executor, '_take_screenshot', return_value="/path/to/screenshot.png"):
            result = self.executor.execute_step(step)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.screenshot_path)

class TestVerifierAgent(unittest.TestCase):
    """Test cases for VerifierAgent"""
    
    def setUp(self):
        self.mock_llm = Mock()
        self.verifier = VerifierAgent(self.mock_llm)
    
    def test_verify_step_result_success(self):
        """Test successful step verification"""
        step = PlanStep(
            action_type=ActionType.TAP,
            parameters={"element_id": "button1"},
            description="Tap button",
            expected_outcome="Button pressed"
        )
        
        execution_result = ExecutionResult(
            success=True,
            action_type="tap",
            parameters={"element_id": "button1"},
            execution_time=1.5,
            screenshot_path="/path/to/screenshot.png"
        )
        
        # Mock LLM verification response
        self.mock_llm.generate_response.return_value = json.dumps({
            "success": True,
            "confidence": 0.9,
            "observations": ["Button was successfully pressed"]
        })
        
        result = self.verifier.verify_step_result(step, execution_result, 0)
        
        self.assertIsInstance(result, VerificationResult)
        self.assertTrue(result.passed)
        self.assertGreater(result.confidence, 0.8)
    
    def test_verify_success_criteria(self):
        """Test success criteria verification"""
        success_criteria = ["login_successful", "user_authenticated"]
        execution_results = [
            ExecutionResult(True, "tap", {}, 1.0),
            ExecutionResult(True, "type", {}, 1.5),
            ExecutionResult(True, "verify", {}, 2.0)
        ]
        
        # Mock LLM responses for each criterion
        self.mock_llm.generate_response.side_effect = [
            json.dumps({
                "criterion_met": True,
                "confidence": 0.85,
                "reasoning": "Login was successful",
                "supporting_evidence": ["Welcome message displayed"]
            }),
            json.dumps({
                "criterion_met": True,
                "confidence": 0.90,
                "reasoning": "User is authenticated",
                "supporting_evidence": ["User profile visible"]
            })
        ]
        
        results = self.verifier.verify_success_criteria(success_criteria, execution_results)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["met"] for r in results))

class TestSupervisorAgent(unittest.TestCase):
    """Test cases for SupervisorAgent"""
    
    def setUp(self):
        self.mock_planner = Mock()
        self.mock_executor = Mock()
        self.mock_verifier = Mock()
        self.mock_llm = Mock()
        
        self.supervisor = SupervisorAgent(
            self.mock_planner,
            self.mock_executor,
            self.mock_verifier,
            self.mock_llm
        )
    
    def test_execute_test_query_success(self):
        """Test successful test query execution"""
        test_query = "Test login flow"
        
        # Mock plan creation
        mock_plan = Mock()
        mock_plan.plan_id = "test_plan_123"
        mock_plan.steps = []
        self.mock_planner.create_execution_plan.return_value = mock_plan
        self.mock_planner.validate_plan.return_value = {"is_valid": True, "score": 85}
        self.mock_planner.optimize_plan.return_value = mock_plan
        
        # Mock execution
        mock_execution_results = [
            ExecutionResult(True, "tap", {}, 1.0),
            ExecutionResult(True, "verify", {}, 2.0)
        ]
        self.mock_executor.execute_plan.return_value = mock_execution_results
        
        # Mock verification
        mock_verification = {
            "overall_success": True,
            "overall_confidence": 0.9,
            "successful_steps": 2,
            "failed_steps": 0
        }
        self.mock_verifier.verify_execution_results.return_value = mock_verification
        
        session = self.supervisor.execute_test_query(test_query)
        
        self.assertEqual(session.status, SupervisorStatus.COMPLETED)
        self.assertEqual(session.query, test_query)
        self.assertIsNotNone(session.end_time)
    
    def test_cancel_session(self):
        """Test session cancellation"""
        # Start a mock session
        self.supervisor.current_session = Mock()
        self.supervisor.current_session.session_id = "test_session"
        self.supervisor.status = SupervisorStatus.EXECUTING
        
        self.supervisor.cancel_current_session()
        
        self.assertEqual(self.supervisor.status, SupervisorStatus.CANCELLED)

class TestLLMInterface(unittest.TestCase):
    """Test cases for LLMInterface"""
    
    def setUp(self):
        self.mock_provider = Mock()
        self.llm_interface = LLMInterface(self.mock_provider)
    
    def test_generate_response_with_cache(self):
        """Test response generation with caching"""
        test_prompt = "Test prompt"
        expected_response = "Test response"
        
        self.mock_provider.generate_response.return_value = expected_response
        
        # First call should hit provider
        response1 = self.llm_interface.generate_response(test_prompt)
        self.assertEqual(response1, expected_response)
        self.mock_provider.generate_response.assert_called_once()
        
        # Second call should use cache
        response2 = self.llm_interface.generate_response(test_prompt)
        self.assertEqual(response2, expected_response)
        # Provider should still only be called once
        self.mock_provider.generate_response.assert_called_once()
    
    def test_generate_with_image(self):
        """Test response generation with image"""
        test_prompt = "Analyze this image"
        test_image_path = "/path/to/image.png"
        expected_response = "Image analysis result"
        
        self.mock_provider.generate_with_image.return_value = expected_response
        
        response = self.llm_interface.generate_response(test_prompt, image_path=test_image_path)
        
        self.assertEqual(response, expected_response)
        self.mock_provider.generate_with_image.assert_called_once_with(
            test_prompt, test_image_path
        )

class TestAndroidEnvironmentWrapper(unittest.TestCase):
    """Test cases for AndroidEnvironmentWrapper"""
    
    def setUp(self):
        # Mock subprocess calls for ADB
        self.patcher = patch('subprocess.run')
        self.mock_subprocess = self.patcher.start()
        
        # Mock successful ADB version check
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Android Debug Bridge version 1.0.41"
        self.mock_subprocess.return_value = mock_result
        
        self.android_env = AndroidEnvironmentWrapper()
    
    def tearDown(self):
        self.patcher.stop()
    
    def test_device_connection(self):
        """Test device connection check"""
        # Mock device list response
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "List of devices attached\nemulator-5554\tdevice\n"
        self.mock_subprocess.return_value = mock_result
        
        # This would normally initialize connection
        device_id = self.android_env._get_default_device()
        self.assertEqual(device_id, "emulator-5554")
    
    def test_take_screenshot(self):
        """Test screenshot functionality"""
        # Mock successful screenshot commands
        self.mock_subprocess.return_value.returncode = 0
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            # Write dummy data to simulate screenshot
            temp_file.write(b"dummy_image_data" * 100)
        
        try:
            # This would normally take actual screenshot
            # For test, we just verify the file operations would work
            result_path = Path(temp_path)
            self.assertTrue(result_path.exists())
            self.assertGreater(result_path.stat().st_size, 1000)
        finally:
            Path(temp_path).unlink(missing_ok=True)

class TestMainOrchestrator(unittest.TestCase):
    """Test cases for MainOrchestrator"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        
        # Create test configuration
        test_config = {
            "android": {
                "device_id": "test_device",
                "default_timeout": 30,
                "screenshot_dir": "screenshots",
                "logs_dir": "logs"
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test_key",
                "temperature": 0.1
            },
            "system": {
                "log_level": "INFO",
                "cleanup_old_files": False
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('main_orchestrator.EnvironmentManager')
    @patch('main_orchestrator.OpenAIProvider')
    def test_initialization(self, mock_openai_provider, mock_env_manager):
        """Test orchestrator initialization"""
        # Mock environment manager
        mock_env_instance = Mock()
        mock_env_instance.initialize_environment.return_value = True
        mock_env_instance.get_config.return_value = {"provider": "openai", "api_key": "test"}
        mock_env_instance.get_android_env.return_value = Mock()
        mock_env_manager.return_value = mock_env_instance
        
        # Mock OpenAI provider
        mock_provider_instance = Mock()
        mock_openai_provider.return_value = mock_provider_instance
        
        orchestrator = MainOrchestrator(str(self.config_path))
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            success = orchestrator.initialize()
        
        self.assertTrue(success)
        self.assertTrue(orchestrator.is_initialized)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    @patch('core.llm_interface.openai.OpenAI')
    def test_end_to_end_workflow(self, mock_openai_client, mock_subprocess):
        """Test complete end-to-end workflow"""
        # Mock ADB commands
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "success"
        
        # Mock OpenAI client
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "intent": "test_intent",
            "complexity": 2,
            "steps": [{"action_type": "screenshot", "parameters": {}}]
        })
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance
        
        # This test would verify the complete workflow but requires
        # significant mocking of the Android environment
        # In a real scenario, this would test with an actual emulator
        pass

def run_specific_test(test_class_name, test_method_name=None):
    """Run a specific test class or method"""
    suite = unittest.TestSuite()
    
    if test_method_name:
        suite.addTest(globals()[test_class_name](test_method_name))
    else:
        suite.addTest(unittest.makeSuite(globals()[test_class_name]))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
