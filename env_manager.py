"""
Environment Manager - Central coordinator for Multi-Agent QA System
Orchestrates Agent-S agents, android_world environment, and test execution
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from agents.planner_agent import PlannerAgent, QAPlan
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from agents.supervisor_agent import SupervisorAgent
from core.android_env_wrapper import AndroidEnvWrapper
from core.logger import QALogger, QATestResult
from config.default_config import config
try:
    from agents.base_agents import AGENT_S_AVAILABLE
except ImportError:
    AGENT_S_AVAILABLE = False
    print("Warning: Could not import AGENT_S_AVAILABLE, defaulting to False")

class EnvironmentManager:
    """
    Central coordinator integrating Agent-S with android_world
    Manages multi-agent QA testing workflow
    """
    
    def __init__(self):
        self.logger = QALogger("EnvironmentManager")
        
        # Initialize agents
        self.planner_agent = PlannerAgent()
        self.executor_agent = ExecutorAgent()
        self.verifier_agent = VerifierAgent()
        self.supervisor_agent = SupervisorAgent()
        
        # Environment and state
        self.android_env = None
        self.current_task = None
        self.active_plan = None
        self.test_results = []
        self.visual_traces = []
        
        # Execution state
        self.is_running = False
        self.completed_steps = []
        
        self.logger.info("EnvironmentManager initialized with Agent-S architecture")
    
    async def initialize(self) -> bool:
        """Initialize all agents and environment"""
        try:
            self.logger.info("Initializing multi-agent QA system")
            
            # Start all agents
            await self.planner_agent.start()
            await self.executor_agent.start()
            await self.verifier_agent.start()
            await self.supervisor_agent.start()
            
            self.logger.info("All agents initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def run_qa_test(self, test_config: Dict[str, Any]) -> QATestResult:
        """
        Run complete QA test using Agent-S and android_world
        """
        test_id = f"test_{int(time.time() * 1000)}"
        self.logger.start_test(test_id, test_config.get("task_name", "unknown"))
        
        try:
            # Extract test configuration
            high_level_goal = test_config.get("goal", "Test mobile app functionality")
            android_world_task = test_config.get("android_world_task", "settings_wifi")
            max_steps = test_config.get("max_steps", config.MAX_PLAN_STEPS)
            
            self.logger.info(f"Starting QA test: {test_id}")
            self.logger.info(f"Goal: {high_level_goal}")
            self.logger.info(f"Android World Task: {android_world_task}")
            
            # Initialize android_world environment
            if not await self._initialize_android_environment(android_world_task):
                raise Exception("Failed to initialize Android environment")
            
            # Step 1: Planning Phase
            plan = await self._planning_phase(high_level_goal, android_world_task)
            if not plan:
                raise Exception("Failed to create test plan")
            
            # Step 2: Execution Phase
            execution_results = await self._execution_phase(plan, max_steps)
            
            # Step 3: Final Verification
            final_verification = await self._final_verification_phase(plan, execution_results)
            
            # Step 4: Supervisor Analysis
            supervisor_analysis = await self._supervisor_analysis_phase(plan, execution_results)
            
            # Create test result
            test_result = self.logger.finish_test(
                android_world_task,
                final_verification.get("final_result", "FAIL"),
                final_verification.get("bug_detected", False),
                supervisor_analysis.get("overall_assessment", "Analysis completed")
            )
            
            # Store results
            self.test_results.append(test_result)
            
            self.logger.info(f"QA test completed: {test_id} - {test_result.final_result}")
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"QA test failed: {e}")
            
            # Create failed test result
            test_result = self.logger.finish_test(
                test_config.get("android_world_task", "unknown"),
                "ERROR",
                False,
                f"Test execution error: {e}"
            )
            
            self.test_results.append(test_result)
            return test_result
    
    async def _initialize_android_environment(self, task_name: str) -> bool:
        """Initialize android_world environment"""
        try:
            self.android_env = AndroidEnvWrapper(task_name=task_name)
            initial_observation = self.android_env.reset()
            
            # Store initial visual trace
            self.visual_traces = [{
                "step": 0,
                "timestamp": time.time(),
                "screenshot": initial_observation.screenshot,
                "ui_hierarchy": initial_observation.ui_hierarchy,
                "action": "environment_reset"
            }]
            
            self.logger.info(f"Android environment initialized for task: {task_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Android environment: {e}")
            return False
    
    async def _planning_phase(self, goal: str, android_world_task: str) -> Optional[QAPlan]:
        """Execute planning phase using PlannerAgent"""
        try:
            self.logger.info("Starting planning phase")
            
            # Get current UI state
            current_ui_state = ""
            if self.android_env and self.android_env.current_observation:
                current_ui_state = self.android_env.current_observation.ui_hierarchy
            
            # Request plan from planner agent
            planning_result = await self.planner_agent.process_task({
                "goal": goal,
                "android_world_task": android_world_task,
                "ui_state": current_ui_state
            })
            
            if planning_result.get("success", False):
                self.active_plan = planning_result["plan"]
                self.logger.info(f"Plan created with {len(self.active_plan.steps)} steps")
                return self.active_plan
            else:
                self.logger.error(f"Planning failed: {planning_result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            self.logger.error(f"Planning phase failed: {e}")
            return None
    
    async def _execution_phase(self, plan: QAPlan, max_steps: int = 10) -> List[Dict[str, Any]]:
        """Execute plan steps and COLLECT ALL ACTIONS 📝✅"""
        execution_results: List[Dict[str, Any]] = []
        step_counter = 0
        self.completed_steps = []
        
        # ✅ CORRECTED: Initialize action collection
        all_collected_actions = []
        
        self.logger.info(f"🚀 Starting execution phase with {len(plan.steps)} planned steps")
        
        while step_counter < max_steps:
            step = plan.get_next_step(self.completed_steps)
            if not step:
                self.logger.info("✅ No more steps to execute")
                break
            
            step_counter += 1
            self.logger.info(f"🧠 Executing step {step.step_id}: {step.description}")
            
            # ---------- EXECUTE --------------------------------------------------
            raw_result = await self.executor_agent.process_task(
                {"plan_step": step, "android_world_task": plan.android_world_task}
            )
            
            # ✅ CORRECTED: Collect the action record from executor
            if "action_record" in raw_result:
                all_collected_actions.append(raw_result["action_record"])
            
            # ---------- VERIFY ---------------------------------------------------
            verification = await self.verifier_agent.process_task(
                {
                    "plan_step": step,
                    "execution_result": raw_result.get("execution_result"),
                    "current_observation": raw_result.get("current_observation"),
                }
            )
            
            # ✅ CORRECTED: Collect verifier action if it returns one
            if isinstance(verification, dict) and "action_record" in verification:
                all_collected_actions.append(verification["action_record"])
            
            # ---------- COLLECT --------------------------------------------------
            execution_results.append({
                "step_id": step.step_id,
                "execution_result": raw_result,
                "verification_result": verification,
                "timestamp": time.time(),
            })
            
            # ---------- SUCCESS / FAILURE ---------------------------------------
            exec_success = raw_result.get("success", False)
            verif_status = verification.get("verification_result", {}).get("status", "UNKNOWN")
            
            if exec_success and verif_status != "FAIL":
                self.completed_steps.append(step.step_id)
                self.logger.info(f"✅ Step {step.step_id} completed successfully")
            else:
                self.logger.warning(f"❌ Step {step.step_id} failed or verification failed")
            
            await asyncio.sleep(0.5)  # Brief pause between steps
        
        # ✅ CORRECTED: Add all collected actions to the current test
        for action in all_collected_actions:
            if self.logger.current_test_actions is not None:
                self.logger.current_test_actions.append(action)
        
        self.logger.info(
            f"🎯 Execution phase completed. "
            f"{len(self.completed_steps)} steps successful, "
            f"{len(all_collected_actions)} actions recorded"
        )
        
        return execution_results

    
    async def _handle_replanning(self, original_plan: QAPlan, failed_step: Any, step_result: Dict[str, Any]) -> None:
        """Handle replanning when step fails"""
        try:
            self.logger.info(f"Initiating replanning due to step {failed_step.step_id} failure")
            
            # Create failure context
            failure_context = {
                "failed_step_id": failed_step.step_id,
                "reason": step_result.get("verification_result", {}).get("verification_result", {}).get("reasoning", "Step failed"),
                "current_ui_state": step_result.get("execution_result", {}).get("current_observation", {}).get("ui_hierarchy", "")
            }
            
            # Request replan
            new_plan = await self.planner_agent.replan(failure_context)
            
            if new_plan:
                # Update active plan with recovery steps
                recovery_steps = [step for step in new_plan.steps if step.step_id not in self.completed_steps]
                original_plan.steps.extend(recovery_steps)
                
                self.logger.info(f"Replanning successful. Added {len(recovery_steps)} recovery steps")
            else:
                self.logger.warning("Replanning failed")
                
        except Exception as e:
            self.logger.error(f"Replanning failed: {e}")
    
    async def _final_verification_phase(self, plan: QAPlan, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform final verification of test results"""
        try:
            self.logger.info("Starting final verification phase")
            
            # Analyze overall execution
            total_steps = len(plan.steps)
            completed_steps = len(self.completed_steps)
            success_rate = completed_steps / total_steps if total_steps > 0 else 0
            
            # Check if primary goal was achieved
            goal_achieved = success_rate >= 0.8  # 80% of steps completed
            
            # Look for any detected bugs
            bug_detected = any(
                result.get("verification_result", {}).get("verification_result", {}).get("issues_detected", [])
                for result in execution_results
            )
            
            # Determine final result
            if goal_achieved and not bug_detected:
                final_result = "PASS"
            elif goal_achieved and bug_detected:
                final_result = "PASS"  # Goal achieved but bugs found
            else:
                final_result = "FAIL"
            
            final_verification = {
                "final_result": final_result,
                "goal_achieved": goal_achieved,
                "bug_detected": bug_detected,
                "success_rate": success_rate,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "verification_summary": self.verifier_agent.get_verification_summary()
            }
            
            self.logger.info(f"Final verification: {final_result} (success rate: {success_rate:.1%})")
            
            return final_verification
            
        except Exception as e:
            self.logger.error(f"Final verification failed: {e}")
            return {
                "final_result": "ERROR",
                "goal_achieved": False,
                "bug_detected": False,
                "error": str(e)
            }
    
    async def _supervisor_analysis_phase(self, plan: QAPlan, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform supervisor analysis of test episode"""
        try:
            self.logger.info("Starting supervisor analysis phase")
            
            # Create temporary test result for analysis
            temp_test_result = QATestResult(
                test_id=f"temp_{int(time.time())}",
                task_name=plan.android_world_task,
                start_time=plan.created_timestamp,
                end_time=time.time(),
                actions=self.logger.current_test_actions,
                final_result="ANALYSIS",
                bug_detected=False
            )
            
            # Request supervisor analysis
            analysis_result = await self.supervisor_agent.process_task({
                "test_result": temp_test_result,
                "visual_trace": self.visual_traces
            })
            
            if analysis_result.get("success", False):
                analysis = analysis_result["analysis"]
                self.logger.info(f"Supervisor analysis completed. Performance score: {analysis.performance_score:.2f}")
                return {
                    "overall_assessment": analysis.overall_assessment,
                    "performance_score": analysis.performance_score,
                    "strengths": analysis.strengths,
                    "weaknesses": analysis.weaknesses,
                    "improvement_suggestions": analysis.improvement_suggestions
                }
            else:
                self.logger.warning("Supervisor analysis failed")
                return {"overall_assessment": "Analysis failed"}
                
        except Exception as e:
            self.logger.error(f"Supervisor analysis failed: {e}")
            return {"overall_assessment": f"Analysis error: {e}"}
    
    async def run_test_suite(self, test_configs: List[Dict[str, Any]]) -> List[QATestResult]:
        """Run multiple QA tests"""
        suite_results = []
        
        self.logger.info(f"Starting test suite with {len(test_configs)} tests")
        
        for i, test_config in enumerate(test_configs):
            self.logger.info(f"Running test {i+1}/{len(test_configs)}")
            
            try:
                result = await self.run_qa_test(test_config)
                suite_results.append(result)
                
                # Brief pause between tests
                await asyncio.sleep(2.0)
                
            except Exception as e:
                self.logger.error(f"Test {i+1} failed: {e}")
                
                # Create error result
                error_result = QATestResult(
                    test_id=f"error_test_{i+1}",
                    task_name=test_config.get("android_world_task", "unknown"),
                    start_time=time.time(),
                    end_time=time.time(),
                    actions=[],
                    final_result="ERROR",
                    bug_detected=False,
                    supervisor_feedback=f"Suite execution error: {e}"
                )
                suite_results.append(error_result)
        
        self.logger.info(f"Test suite completed. {len(suite_results)} tests executed")
        
        return suite_results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get REAL system metrics with corrected Agent-S status"""
        if not self.test_results:
            return {"message": "No test results available 😕"}
        
        # CORRECTED: Check REAL Agent-S status across all agents
        agent_s_statuses = [
            self.planner_agent.is_agent_s_active(),
            self.executor_agent.is_agent_s_active(),
            self.verifier_agent.is_agent_s_active(),
            self.supervisor_agent.is_agent_s_active()
        ]
        
        # Agent-S is active if ANY agent has it working ✅
        agent_s_active = any(agent_s_statuses)
        
        # Additional debug info 🕵️‍♂️
        agent_s_details = {
            "planner": self.planner_agent.is_agent_s_active(),
            "executor": self.executor_agent.is_agent_s_active(), 
            "verifier": self.verifier_agent.is_agent_s_active(),
            "supervisor": self.supervisor_agent.is_agent_s_active(),
            "agent_s_available": AGENT_S_AVAILABLE,
            "mock_mode": config.USE_MOCK_LLM
        }
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.final_result == "PASS")
        
        # REAL agent performance from actual recorded actions 🎬
        all_actions = []
        for result in self.test_results:
            all_actions.extend(result.actions)
        
        agent_metrics = {}
        for action in all_actions:
            agent_name = action.agent_name
            if agent_name not in agent_metrics:
                agent_metrics[agent_name] = {"total": 0, "successful": 0, "total_duration": 0.0}
            
            agent_metrics[agent_name]["total"] += 1
            agent_metrics[agent_name]["total_duration"] += action.duration
            if action.success:
                agent_metrics[agent_name]["successful"] += 1
        
        # Calculate REAL success rates 📊
        for agent_name, metrics in agent_metrics.items():
            if metrics["total"] > 0:
                metrics["success_rate"] = metrics["successful"] / metrics["total"]
                metrics["avg_duration"] = metrics["total_duration"] / metrics["total"]
            else:
                metrics["success_rate"] = 0.0
                metrics["avg_duration"] = 0.0
        
        return {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "agent_performance": agent_metrics,
            "system_integration": {
                "agent_s_active": agent_s_active,  # CORRECTED status ✅
                "agent_s_details": agent_s_details,  # Debug info 🛠️
                "android_world_connected": bool(self.android_env),
                "llm_interface": "mock" if config.USE_MOCK_LLM else "gemini"
            },
            "real_actions_recorded": len(all_actions),
            "visual_traces_captured": len(self.visual_traces)
        }
    async def shutdown(self) -> None:
        """Shutdown all agents and cleanup"""
        try:
            self.logger.info("Shutting down multi-agent QA system")
            
            # Stop all agents
            await self.planner_agent.stop()
            await self.executor_agent.stop()
            await self.verifier_agent.stop()
            await self.supervisor_agent.stop()
            
            # Close Android environment
            if self.android_env:
                self.android_env.close()
            
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
