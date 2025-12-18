from env import Task


TASKS = [
    Task(
        name="reverse_string",
        initial_solution="""
def solve(text: str) -> str:
    return ""
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize("text", ["abc", "", "palindrome", "12345", "AaBb"])
def test_reverse(text):
    assert solve(text) == text[::-1]
""".strip(),
    ),
    Task(
        name="fizzbuzz",
        initial_solution="""
def solve(n: int) -> list[str]:
    return []
""".strip(),
        tests="""
import pytest
from solution import solve

def fizzbuzz_ref(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out

@pytest.mark.parametrize("n", [1, 3, 5, 15, 32])
def test_cases(n):
    assert solve(n) == fizzbuzz_ref(n)
""".strip(),
    ),
    Task(
        name="palindrome_check",
        initial_solution="""
def solve(text: str) -> bool:
    return False
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize(
    "text,expected",
    [
        ("racecar", True),
        ("", True),
        ("Able was I ere I saw Elba", True),
        ("palindrome", False),
        ("123321", True),
    ],
)
def test_palindrome(text, expected):
    normalized = "".join(ch.lower() for ch in text if ch.isalnum())
    assert solve(text) == (normalized == normalized[::-1])
""".strip(),
    ),
    Task(
        name="balanced_brackets",
        initial_solution="""
def solve(text: str) -> bool:
    return False
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize(
    "text,expected",
    [
        ("", True),
        ("()", True),
        ("([]{})", True),
        ("([)]", False),
        ("(((()", False),
        ("<>({[]})", True),
    ],
)
def test_brackets(text, expected):
    assert solve(text) == expected
""".strip(),
    ),
    Task(
        name="two_sum",
        initial_solution="""
def solve(nums: list[int], target: int) -> tuple[int, int] | None:
    return None
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize(
    "nums,target,expected",
    [
        ([2,7,11,15], 9, (0,1)),
        ([3,2,4], 6, (1,2)),
        ([3,3], 6, (0,1)),
        ([1,5,1], 2, (0,2)),
    ],
)
def test_two_sum(nums, target, expected):
    result = solve(nums, target)
    assert result == expected
""".strip(),
    ),
    Task(
        name="binary_search",
        initial_solution="""
def solve(nums: list[int], target: int) -> int:
    return -1
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize(
    "nums,target,expected",
    [
        ([1,3,5,7], 5, 2),
        ([1,3,5,7], 2, -1),
        ([2], 2, 0),
        ([], 1, -1),
    ],
)
def test_binary_search(nums, target, expected):
    assert solve(nums, target) == expected
""".strip(),
    ),
    Task(
        name="merge_sorted_lists",
        initial_solution="""
def solve(a: list[int], b: list[int]) -> list[int]:
    return []
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1,3,5], [2,4,6], [1,2,3,4,5,6]),
        ([], [1,2], [1,2]),
        ([1,2], [], [1,2]),
        ([1,1,1], [1,1], [1,1,1,1,1]),
    ],
)
def test_merge(a, b, expected):
    assert solve(a, b) == expected
""".strip(),
    ),
    Task(
        name="matrix_transpose",
        initial_solution="""
def solve(matrix: list[list[int]]) -> list[list[int]]:
    return []
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize(
    "matrix,expected",
    [
        ([[1,2],[3,4]], [[1,3],[2,4]]),
        ([[1]], [[1]]),
        ([], []),
    ],
)
def test_transpose(matrix, expected):
    assert solve(matrix) == expected
""".strip(),
    ),
    Task(
        name="factorial_recursive",
        initial_solution="""
def solve(n: int) -> int:
    return 0
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize("n", [0,1,5,7])
def test_factorial(n):
    assert solve(n) == _ref(n)

def _ref(n):
    return 1 if n == 0 else n * _ref(n-1)
""".strip(),
    ),
    Task(
        name="gcd",
        initial_solution="""
def solve(a: int, b: int) -> int:
    return 0
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (54, 24, 6),
        (20, 8, 4),
        (17, 13, 1),
        (0, 5, 5),
        (5, 0, 5),
    ],
)
def test_gcd(a, b, expected):
    assert solve(a, b) == expected
""".strip(),
    ),
]


if __name__ == "__main__":
    for task in TASKS:
        print(task.name)
