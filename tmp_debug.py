from core.planner import normalise_goal_flags, plan
from core.registry import BehaviorRegistry
from core.interpreter import Interpreter

registry = BehaviorRegistry().discover().load_meta()
interpreter = Interpreter(registry)
original_code = "def fetch_data():\n    return os.path.join('a', 'b')\n"
ctx = {"text": "Refactor fetch_data to async, fix the imports, add a user login endpoint, and ensure the answer is formatted.", "data": {"code": original_code}}
goal_flags = normalise_goal_flags("formatted")
result = plan(goal_flags, ctx, registry, interpreter, cluster_bias="analytic", max_expansions=18)
print(result['goal_satisfied'])
print(result['remaining_flags'])
