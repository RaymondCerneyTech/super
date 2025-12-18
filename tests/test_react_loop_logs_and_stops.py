from core.interpreter import Interpreter
from core.planner import plan
from core.registry import BehaviorRegistry


def test_react_loop_logs_and_stops() -> None:
    registry = BehaviorRegistry().discover().load_meta()
    interpreter = Interpreter(registry)

    ctx = {"text": "noop", "data": {"react_max_iterations": 3}}
    result = plan([], ctx, registry, interpreter, cluster_bias="analytic", max_expansions=2)

    trace = result.get("react_trace")
    assert isinstance(trace, list) and trace
    first_entry = trace[0]
    assert first_entry["action"] == "finalize"
    assert "Finalizing plan" in first_entry["observation"]
    assert result["goal_satisfied"] is True
    assert ctx["data"].get("react_trace") == trace
