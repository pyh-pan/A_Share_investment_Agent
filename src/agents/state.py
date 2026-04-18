from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage
import json
from src.utils.logging_config import setup_logger

# 设置日志记录
logger = setup_logger('agent_state')


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}


def merge_dicts_deep(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dicts: nested dicts are recursively merged, non-dict values overwritten.

    This replaces merge_dicts for state["data"] so that nested sub-states
    (e.g. debate_state, technical_report) are merged incrementally rather
    than replaced wholesale by the last writer.

    Example:
        a = {"debate_state": {"round_count": 1, "bull_history": "B1"}}
        b = {"debate_state": {"round_count": 2, "bear_history": "R1"}}
        result = {"debate_state": {"round_count": 2, "bull_history": "B1", "bear_history": "R1"}}
    """
    result = {**a}
    for key, value in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value
    return result

# Define agent state


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts_deep]
    metadata: Annotated[Dict[str, Any], merge_dicts]


def show_workflow_status(agent_name: str, status: str = "processing"):
    """Display agent workflow status in a clean format.

    Args:
        agent_name: Name of the agent
        status: Status of the agent's work ("processing" or "completed")
    """
    if status == "processing":
        logger.info(f"🔄 {agent_name} 正在分析...")
    else:
        logger.info(f"✅ {agent_name} 分析完成")


def show_agent_reasoning(output, agent_name):
    """Display agent's analysis results."""
    def convert_to_serializable(obj):
        if hasattr(obj, 'to_dict'):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    # logger.info(f"{'='*20} {agent_name} Analysis Details {'='*20}")

    if isinstance(output, (dict, list)):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        logger.info(json.dumps(serializable_output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            logger.info(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            logger.info(output)
