# agents/planner_agent.py - FULLY CORRECTED VERSION
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

from core.llm_interface import LLMRequest
from core.logger import QALogger

@dataclass
class Subgoal:
    id: int
    action: str
    description: str
    priority: int = 1
    dependencies: List[int] = None
    status: str = "pending"  # pending, active, completed, failed
    retry_count: int = 0
    max_retries: int = 2
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class TestPlan:
    goal: str
    subgoals: List[Subgoal]
    expected_screens: List[str]
    contingencies: List[str]
    current_subgoal_id: int = 0
    estimated_steps: int = 0
    
class PlannerAgent:
    """Agent responsible for decomposing QA goals into executable subgoals"""
    
    def __init__(self, llm_interface, logger: QALogger, config: Dict[str, Any]):
        self.llm = llm_interface
        self.logger = logger
        self.config = config
        self.current_plan: Optional[TestPlan] = None
        self.plan_history: List[TestPlan] = []
        self.next_subgoal_id = 1  # Track unique IDs
        
        print(f"[PLANNER INIT] Planner agent initialized with config: {config}")
        
    async def create_plan(self, goal: str, context: Dict[str, Any] = None) -> TestPlan:
        """Create initial test plan from high-level goal - FIXED VERSION"""
        start_time = time.time()
        context = context or {}
        
        print(f"[PLANNER] Creating plan for goal: {goal}")
        
        try:
            # Build comprehensive planning prompt
            planning_prompt = self._build_planning_prompt(goal, context)
            
            print(f"[PLANNER] Sending planning request to LLM...")
            
            # Generate plan using LLM
            request = LLMRequest(
                prompt=planning_prompt,
                model=self.config.get("model", "mock"),
                temperature=self.config.get("temperature", 0.1),
                max_tokens=self.config.get("max_tokens", 1000),
                system_prompt=self._get_system_prompt()
            )
            
            response = await self.llm.generate(request)
            
            print(f"[PLANNER] Received LLM response, parsing...")
            
            # Parse response into structured plan
            plan = self._parse_plan_response(goal, response.content)
            
            # Validate and fix plan
            plan = self._validate_and_fix_plan(plan)
            
            # Set estimated steps
            plan.estimated_steps = len(plan.subgoals)
            
            self.current_plan = plan
            self.plan_history.append(plan)
            
            execution_time = time.time() - start_time
            
            # Log the planning action
            self.logger.log_agent_action(
                agent_name="planner",
                action="create_plan",
                success=True,
                duration=execution_time,
                details={
                    "goal": goal,
                    "subgoals_count": len(plan.subgoals),
                    "estimated_steps": plan.estimated_steps
                }
            )
            
            print(f"[PLANNER] Plan created successfully with {len(plan.subgoals)} subgoals")
            logger.info(f"Created plan with {len(plan.subgoals)} subgoals for: {goal}")
            
            return plan
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"[PLANNER] ERROR: Plan creation failed: {e}")
            
            self.logger.log_agent_action(
                agent_name="planner",
                action="create_plan",
                success=False,
                duration=execution_time,
                details={"goal": goal, "error": str(e)}
            )
            
            logger.error(f"Planning failed: {e}")
            
            # Return fallback plan instead of failing completely
            fallback_plan = self._create_fallback_plan(goal)
            fallback_plan = self._validate_and_fix_plan(fallback_plan)
            self.current_plan = fallback_plan
            return fallback_plan
    
    async def adapt_plan(self, current_ui_state: Dict[str, Any], issue: str, suggestion: str = "") -> TestPlan:
        """Adapt current plan based on unexpected conditions - IMPROVED VERSION"""
        if not self.current_plan:
            logger.warning("No current plan to adapt")
            return await self.create_plan("fallback plan due to adaptation failure")
        
        start_time = time.time()
        
        print(f"[PLANNER] Adapting plan due to issue: {issue}")
        
        try:
            issue_lower = issue.lower()
            
            if "popup" in issue_lower or "dialog" in issue_lower:
                self._insert_popup_handling_step()
            elif "permission" in issue_lower:
                self._insert_permission_handling_step()
            elif "navigation" in issue_lower or "back" in issue_lower:
                self._insert_navigation_step()
            elif "timeout" in issue_lower or "element not found" in issue_lower:
                self._handle_timeout_issue()
            else:
                # Generic retry strategy
                self._mark_current_for_retry()
            
            # Re-validate plan after adaptation
            self.current_plan = self._validate_and_fix_plan(self.current_plan)
            
            execution_time = time.time() - start_time
            
            self.logger.log_agent_action(
                agent_name="planner",
                action="adapt_plan",
                success=True,
                duration=execution_time,
                details={"issue": issue, "suggestion": suggestion}
            )
            
            print(f"[PLANNER] Plan adapted successfully")
            logger.info(f"Adapted plan due to: {issue}")
            
            return self.current_plan
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"[PLANNER] ERROR: Plan adaptation failed: {e}")
            
            self.logger.log_agent_action(
                agent_name="planner",
                action="adapt_plan",
                success=False,
                duration=execution_time,
                details={"issue": issue, "error": str(e)}
            )
            
            logger.error(f"Plan adaptation failed: {e}")
            return self.current_plan
    
    def get_next_subgoal(self) -> Optional[Subgoal]:
        """Get next pending subgoal to execute - FIXED VERSION"""
        if not self.current_plan or not self.current_plan.subgoals:
            print(f"[PLANNER] No plan or subgoals available")
            return None
        
        print(f"[PLANNER] Looking for next subgoal...")
        
        # Find next pending subgoal with satisfied dependencies, prioritized by priority and ID
        available_subgoals = []
        
        for subgoal in self.current_plan.subgoals:
            if subgoal.status == "pending" and self._dependencies_satisfied(subgoal):
                available_subgoals.append(subgoal)
        
        if not available_subgoals:
            print(f"[PLANNER] No pending subgoals with satisfied dependencies")
            return None
        
        # Sort by priority (higher first), then by ID (lower first)
        next_subgoal = min(available_subgoals, key=lambda sg: (-sg.priority, sg.id))
        
        next_subgoal.status = "active"
        self.current_plan.current_subgoal_id = next_subgoal.id
        print(f"[PLANNER] Selected subgoal {next_subgoal.id} for execution: {next_subgoal.description}")
        
        return next_subgoal
    
    def mark_subgoal_completed(self, subgoal_id: int):
        """Mark subgoal as completed"""
        if self.current_plan:
            for subgoal in self.current_plan.subgoals:
                if subgoal.id == subgoal_id:
                    subgoal.status = "completed"
                    print(f"[PLANNER] Subgoal {subgoal_id} marked as completed: {subgoal.description}")
                    logger.info(f"Subgoal {subgoal_id} completed: {subgoal.description}")
                    break
    
    def mark_subgoal_failed(self, subgoal_id: int, error: str = ""):
        """Mark subgoal as failed"""
        if self.current_plan:
            for subgoal in self.current_plan.subgoals:
                if subgoal.id == subgoal_id:
                    if subgoal.retry_count < subgoal.max_retries:
                        subgoal.status = "pending"  # Allow retry
                        subgoal.retry_count += 1
                        print(f"[PLANNER] Subgoal {subgoal_id} failed, marked for retry ({subgoal.retry_count}/{subgoal.max_retries})")
                    else:
                        subgoal.status = "failed"
                        print(f"[PLANNER] Subgoal {subgoal_id} permanently failed after max retries: {error}")
                    
                    logger.warning(f"Subgoal {subgoal_id} failed: {error}")
                    break
    
    def is_plan_complete(self) -> bool:
        """Check if all subgoals are completed or if critical path is blocked"""
        if not self.current_plan:
            return False
        
        completed_count = sum(1 for subgoal in self.current_plan.subgoals if subgoal.status == "completed")
        failed_count = sum(1 for subgoal in self.current_plan.subgoals if subgoal.status == "failed")
        total_count = len(self.current_plan.subgoals)
        
        print(f"[PLANNER] Plan status: {completed_count}/{total_count} completed, {failed_count} failed")
        
        # Check if plan is complete (all completed)
        if completed_count == total_count:
            return True
        
        # Check if plan is blocked (no more executable subgoals)
        pending_executable = any(
            subgoal.status == "pending" and self._dependencies_satisfied(subgoal)
            for subgoal in self.current_plan.subgoals
        )
        
        if not pending_executable and completed_count + failed_count == total_count:
            print(f"[PLANNER] Plan blocked - no more executable subgoals")
            return True
        
        return False
    
    def _validate_and_fix_plan(self, plan: TestPlan) -> TestPlan:
        """Validate and fix plan structure"""
        if not plan or not plan.subgoals:
            return plan
        
        # Fix subgoal IDs to be sequential
        for i, subgoal in enumerate(plan.subgoals):
            subgoal.id = i + 1
        
        # Validate and fix dependencies
        valid_ids = {sg.id for sg in plan.subgoals}
        
        for subgoal in plan.subgoals:
            # Remove invalid dependencies
            subgoal.dependencies = [dep_id for dep_id in subgoal.dependencies if dep_id in valid_ids and dep_id < subgoal.id]
        
        # Check for circular dependencies and remove them
        self._remove_circular_dependencies(plan)
        
        print(f"[PLANNER] Plan validated and fixed")
        return plan
    
    def _remove_circular_dependencies(self, plan: TestPlan):
        """Remove circular dependencies using topological sort approach"""
        # Build dependency graph
        graph = {sg.id: set(sg.dependencies) for sg in plan.subgoals}
        
        # Remove circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Find and remove problematic dependencies
        for subgoal in plan.subgoals:
            original_deps = subgoal.dependencies.copy()
            for dep_id in original_deps:
                # Temporarily add this dependency and check for cycles
                if has_cycle(subgoal.id):
                    subgoal.dependencies.remove(dep_id)
                    print(f"[PLANNER] Removed circular dependency: {subgoal.id} -> {dep_id}")
    
    def _build_planning_prompt(self, goal: str, context: Dict[str, Any] = None) -> str:
        """Build LLM prompt for initial planning"""
        context = context or {}
        
        prompt = f"""
Create a detailed test plan for Android UI testing.

Goal: {goal}

Requirements:
1. Break down the goal into specific, actionable subgoals
2. Each subgoal should be executable by an Android UI automation agent
3. Consider typical Android app navigation patterns
4. Include proper dependencies between steps (dependencies must have lower IDs)
5. Plan for common UI elements (buttons, toggles, lists, etc.)

Context:
- Platform: Android mobile UI
- Available actions: touch, scroll, type, swipe, back, home
- UI elements can be found by text, ID, class name, or coordinates
- Consider typical Android Settings app structure

Generate a JSON response with this exact structure:
{{
    "subgoals": [
        {{
            "id": 1,
            "action": "action_name", 
            "description": "Clear description of what to do",
            "priority": 1,
            "dependencies": []
        }}
    ],
    "expected_screens": ["list", "of", "expected", "screens"],
    "contingencies": ["handle_popup", "handle_permission_dialog"]
}}

IMPORTANT: 
- IDs must be sequential starting from 1
- Dependencies can only reference earlier subgoals (lower IDs)
- Each step should be atomic and testable

Focus on creating a practical, step-by-step plan for: {goal}

Make each subgoal specific and testable. Use action names like:
- open_settings, open_app_drawer, find_app
- navigate_to_section, tap_option, toggle_switch  
- verify_state, check_display, confirm_change
"""
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for planning"""
        return """
You are an expert Android UI test planner. Your job is to break down high-level testing goals into specific, executable subgoals that can be automated.

Key principles:
1. Be specific about UI interactions (tap, scroll, toggle, etc.)
2. Consider standard Android app patterns and navigation
3. Plan for Settings app, Calculator, Clock, and other system apps
4. Make each step atomic and verifiable
5. Include error handling for popups and permissions
6. Use realistic action names that describe UI interactions
7. Dependencies must only reference earlier steps (lower IDs)

Always respond with valid JSON. Focus on practical actions that work in Android emulators.
Each subgoal should be something a UI automation script can execute.
"""
    
    def _parse_plan_response(self, goal: str, response_content: str) -> TestPlan:
        """Parse LLM response into TestPlan - ROBUST VERSION"""
        print(f"[PLANNER] Parsing LLM response...")
        
        try:
            # Clean response and extract JSON
            response_content = response_content.strip()
            
            # Remove markdown code blocks if present
            if response_content.startswith("```json"):
                response_content = response_content[7:]
            elif response_content.startswith("```"):
                response_content = response_content[3:]
            
            if response_content.endswith("```"):
                response_content = response_content[:-3]
            
            response_content = response_content.strip()
            
            print(f"[PLANNER] Cleaned response: {response_content[:200]}...")
            
            # Parse JSON
            data = json.loads(response_content)
            
            # Parse subgoals with validation
            subgoals = []
            subgoal_data = data.get("subgoals", [])
            
            if not subgoal_data:
                print(f"[PLANNER] No subgoals in response, creating fallback")
                raise ValueError("No subgoals found in response")
            
            for i, sg_data in enumerate(subgoal_data):
                # Ensure proper ID assignment
                subgoal_id = sg_data.get("id", i + 1)
                if not isinstance(subgoal_id, int) or subgoal_id <= 0:
                    subgoal_id = i + 1
                
                # Validate dependencies
                dependencies = sg_data.get("dependencies", [])
                if not isinstance(dependencies, list):
                    dependencies = []
                
                # Filter out invalid dependencies
                dependencies = [dep for dep in dependencies if isinstance(dep, int) and dep > 0 and dep < subgoal_id]
                
                subgoal = Subgoal(
                    id=subgoal_id,
                    action=str(sg_data.get("action", f"step_{i+1}")),
                    description=str(sg_data.get("description", f"Execute step {i+1}")),
                    priority=int(sg_data.get("priority", 1)),
                    dependencies=dependencies
                )
                subgoals.append(subgoal)
                print(f"[PLANNER] Parsed subgoal {subgoal.id}: {subgoal.description}")
            
            plan = TestPlan(
                goal=goal,
                subgoals=subgoals,
                expected_screens=data.get("expected_screens", ["home", "target"]),
                contingencies=data.get("contingencies", ["handle_popup"])
            )
            
            print(f"[PLANNER] Successfully parsed plan with {len(subgoals)} subgoals")
            return plan
            
        except Exception as e:
            print(f"[PLANNER] Failed to parse plan response: {e}")
            logger.error(f"Failed to parse plan response: {e}")
            print(f"[PLANNER] Raw response: {response_content}")
            
            # Create fallback plan
            return self._create_fallback_plan(goal)
    
    def _create_fallback_plan(self, goal: str) -> TestPlan:
        """Create a simple fallback plan when LLM planning fails - CORRECTED VERSION"""
        print(f"[PLANNER] Creating fallback plan for: {goal}")
        
        goal_lower = goal.lower()
        
        if "wifi" in goal_lower or "wi-fi" in goal_lower:
            subgoals = [
                Subgoal(1, "open_settings", "Open Settings app from home screen", dependencies=[]),
                Subgoal(2, "navigate_wifi", "Find and tap Wi-Fi option in Settings", dependencies=[1]),
                Subgoal(3, "toggle_wifi", "Toggle Wi-Fi switch to change state", dependencies=[2]),
                Subgoal(4, "verify_wifi_state", "Verify Wi-Fi state has changed", dependencies=[3])
            ]
            expected_screens = ["home", "settings", "wifi_settings"]
            
        elif "calculator" in goal_lower or "calculation" in goal_lower:
            subgoals = [
                Subgoal(1, "open_app_drawer", "Open app drawer or find Calculator", dependencies=[]),
                Subgoal(2, "open_calculator", "Tap Calculator app to open it", dependencies=[1]),
                Subgoal(3, "perform_calculation", "Enter numbers and perform calculation", dependencies=[2]),
                Subgoal(4, "verify_result", "Verify calculation result is correct", dependencies=[3])
            ]
            expected_screens = ["home", "calculator"]
            
        elif "bluetooth" in goal_lower:
            subgoals = [
                Subgoal(1, "open_settings", "Open Settings app", dependencies=[]),
                Subgoal(2, "navigate_bluetooth", "Navigate to Bluetooth settings", dependencies=[1]),
                Subgoal(3, "toggle_bluetooth", "Toggle Bluetooth switch", dependencies=[2]),
                Subgoal(4, "verify_bluetooth", "Verify Bluetooth state changed", dependencies=[3])
            ]
            expected_screens = ["home", "settings", "bluetooth_settings"]
            
        elif "storage" in goal_lower:
            subgoals = [
                Subgoal(1, "open_settings", "Open Settings app", dependencies=[]),
                Subgoal(2, "navigate_storage", "Navigate to Storage settings", dependencies=[1]),
                Subgoal(3, "check_storage_usage", "View storage usage information", dependencies=[2])
            ]
            expected_screens = ["home", "settings", "storage_settings"]
            
        elif "alarm" in goal_lower or "clock" in goal_lower:
            subgoals = [
                Subgoal(1, "open_clock_app", "Open Clock/Alarm app", dependencies=[]),
                Subgoal(2, "navigate_to_alarms", "Navigate to alarms section", dependencies=[1]),
                Subgoal(3, "create_alarm", "Create new alarm", dependencies=[2]),
                Subgoal(4, "set_alarm_time", "Set alarm time", dependencies=[3]),
                Subgoal(5, "save_alarm", "Save the alarm", dependencies=[4])
            ]
            expected_screens = ["home", "clock", "alarm_setup"]
            
        else:
            # Very generic fallback
            subgoals = [
                Subgoal(1, "execute_task", f"Execute the requested task: {goal}", dependencies=[])
            ]
            expected_screens = ["home", "target"]
        
        print(f"[PLANNER] Fallback plan created with {len(subgoals)} subgoals")
        
        return TestPlan(
            goal=goal,
            subgoals=subgoals,
            expected_screens=expected_screens,
            contingencies=["handle_popup", "handle_permission"]
        )
    
    def _dependencies_satisfied(self, subgoal: Subgoal) -> bool:
        """Check if subgoal dependencies are satisfied"""
        if not subgoal.dependencies:
            return True
        
        if not self.current_plan:
            return False
        
        for dep_id in subgoal.dependencies:
            # Find the dependency subgoal
            dep_satisfied = False
            for sg in self.current_plan.subgoals:
                if sg.id == dep_id and sg.status == "completed":
                    dep_satisfied = True
                    break
            
            if not dep_satisfied:
                print(f"[PLANNER] Dependency {dep_id} not satisfied for subgoal {subgoal.id}")
                return False
        
        return True
    
    def _insert_popup_handling_step(self):
        """Insert popup handling step into current plan"""
        if not self.current_plan:
            return
        
        # Insert popup handling subgoal with next available ID
        popup_subgoal = Subgoal(
            id=self._get_next_id(),
            action="handle_popup",
            description="Handle popup or dialog that appeared",
            priority=2,  # Higher priority to execute soon
            dependencies=[]
        )
        
        self.current_plan.subgoals.append(popup_subgoal)
        print(f"[PLANNER] Inserted popup handling step with ID {popup_subgoal.id}")
    
    def _insert_permission_handling_step(self):
        """Insert permission handling step into current plan"""
        if not self.current_plan:
            return
        
        # Insert permission handling subgoal
        permission_subgoal = Subgoal(
            id=self._get_next_id(),
            action="handle_permission",
            description="Handle permission dialog",
            priority=2,  # Higher priority to execute soon
            dependencies=[]
        )
        
        self.current_plan.subgoals.append(permission_subgoal)
        print(f"[PLANNER] Inserted permission handling step with ID {permission_subgoal.id}")
    
    def _insert_navigation_step(self):
        """Insert navigation step to handle back/navigation issues"""
        if not self.current_plan:
            return
        
        nav_subgoal = Subgoal(
            id=self._get_next_id(),
            action="navigate_back",
            description="Navigate back or to correct screen",
            priority=2,
            dependencies=[]
        )
        
        self.current_plan.subgoals.append(nav_subgoal)
        print(f"[PLANNER] Inserted navigation step with ID {nav_subgoal.id}")
    
    def _handle_timeout_issue(self):
        """Handle timeout or element not found issues"""
        if not self.current_plan:
            return
        
        # Add wait/retry step
        wait_subgoal = Subgoal(
            id=self._get_next_id(),
            action="wait_and_retry",
            description="Wait for elements to load and retry",
            priority=2,
            dependencies=[]
        )
        
        self.current_plan.subgoals.append(wait_subgoal)
        print(f"[PLANNER] Inserted wait/retry step with ID {wait_subgoal.id}")
    
    def _mark_current_for_retry(self):
        """Mark current subgoal for retry"""
        if not self.current_plan:
            return
        
        current_id = self.current_plan.current_subgoal_id
        for subgoal in self.current_plan.subgoals:
            if subgoal.id == current_id and subgoal.retry_count < subgoal.max_retries:
                subgoal.status = "pending"  # Reset to pending for retry
                print(f"[PLANNER] Marked subgoal {current_id} for retry")
                break
    
    def _get_next_id(self) -> int:
        """Get next available ID for new subgoals"""
        if not self.current_plan or not self.current_plan.subgoals:
            return 1
        
        max_id = max(sg.id for sg in self.current_plan.subgoals)
        return max_id + 1