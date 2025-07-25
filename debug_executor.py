# debug_executor.py
import asyncio
import json
from core.llm_interface import MockLLMInterface, LLMRequest

async def debug_executor_response():
    mock_llm = MockLLMInterface()
    
    # Simulate the exact request the executor agent makes
    executor_request = LLMRequest(
        prompt="""
Subgoal to execute: Open Settings app
Action type: open_settings

Current UI Elements:
[{"id": "element_0", "text": "Settings", "content_desc": "", "class": "android.widget.TextView", "clickable": true, "bounds": [100, 100, 300, 150]}]

Context: {}

Please ground this subgoal to a specific Android action. Consider the available UI elements and choose the most appropriate one.

Respond with JSON:
{
    "action_type": "touch|scroll|type|wait",
    "element_id": "target_element_id",
    "coordinates": [x, y],
    "text": "text_to_type",
    "direction": "scroll_direction",
    "duration": wait_seconds,
    "confidence": 0.0-1.0,
    "reasoning": "why this action"
}

Only include fields relevant to the chosen action_type.
""",
        model="mock",
        system_prompt="You are an expert Android UI automation specialist. Your job is to ground high-level subgoals into specific, executable Android actions."
    )
    
    print("=== DEBUGGING EXECUTOR REQUEST ===")
    print("Request prompt (first 200 chars):")
    print(repr(executor_request.prompt[:200]))
    
    response = await mock_llm.generate(executor_request)
    print(f"\nMock LLM Response:")
    print(f"Content: {repr(response.content)}")
    
    try:
        parsed = json.loads(response.content)
        print(f"\nParsed JSON:")
        print(json.dumps(parsed, indent=2))
        print(f"\nHas 'action_type' field? {('action_type' in parsed)}")
        if 'action_type' in parsed:
            print(f"Action type value: {repr(parsed['action_type'])}")
    except Exception as e:
        print(f"\nJSON Parse Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_executor_response())
