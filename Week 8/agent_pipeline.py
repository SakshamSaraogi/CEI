"""
Single Agent Systems & Agent Pipelines - Mini Project
Task-routing agent with: calculator tool, keyword extraction tool,
and a fallback handler. All responses are structured JSON.
"""

import json
import re
import ast
import operator


# ---------------------------------------------------------------------------
# 1. Baseline tool: calculator(expression)
# ---------------------------------------------------------------------------
# Uses a restricted AST-based evaluator instead of raw eval() so that
# arbitrary/malicious code cannot be executed via user input. Only basic
# arithmetic (+, -, *, /, **, parentheses, unary minus) is permitted.

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):  # numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Non-numeric constant in expression")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))
    raise ValueError("Unsupported expression element")


def calculator(expression: str):
    """
    Safely evaluates a basic arithmetic expression string.
    Raises ValueError on invalid/unsafe input.
    """
    expression = expression.strip()
    if not expression:
        raise ValueError("Empty mathematical expression")

    # Only allow digits, whitespace, and basic math symbols before parsing.
    if not re.fullmatch(r"[0-9\.\+\-\*\/\%\(\)\s]+", expression):
        raise ValueError(f"Invalid characters in expression: '{expression}'")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Could not evaluate expression '{expression}': {e}")

    return result


# ---------------------------------------------------------------------------
# 2. Baseline tool: extract_keywords(text)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "of", "for",
    "to", "and", "or", "but", "with", "this", "that", "it", "its", "as",
    "at", "by", "be", "from", "extract", "keywords", "keyword", "please",
    "find", "me", "my", "i", "you", "your", "about",
}


def extract_keywords(text: str):
    """
    Very simple keyword extractor: lowercases, strips punctuation,
    removes stopwords, and returns unique remaining words in order
    of first appearance.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty text supplied for keyword extraction")

    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    seen = []
    for w in words:
        if w not in _STOPWORDS and w not in seen:
            seen.append(w)

    if not seen:
        raise ValueError("No meaningful keywords found in input")

    return seen


# ---------------------------------------------------------------------------
# 3. Agent router: agent(query)
# ---------------------------------------------------------------------------
def _make_response(resp_type: str, result):
    """Enforces the required schema: {"type": ..., "result": ...}"""
    return {"type": resp_type, "result": result}


def agent(query: str) -> dict:
    """
    Routes an incoming query string to the correct tool based on
    conditional keyword checks on the lowercased query, and returns
    a JSON-schema-valid dict: {"type": ..., "result": ...}
    """
    try:
        if not isinstance(query, str) or not query.strip():
            return _make_response("error", "Empty or invalid query received")

        lower_query = query.lower()

        # --- Route 1: calculation ---
        if "calculate" in lower_query:
            # Strip the trigger word and any leading filler, keep the
            # rest of the string as the candidate math expression.
            expr = re.sub(r".*calculate", "", lower_query, count=1).strip(" :")
            try:
                value = calculator(expr)
                return _make_response("calculation", value)
            except ValueError as e:
                return _make_response("error", str(e))

        # --- Route 2: keyword extraction ---
        elif "keywords" in lower_query:
            try:
                kws = extract_keywords(query)
                return _make_response("keywords", kws)
            except ValueError as e:
                return _make_response("error", str(e))

        # --- Route 3: fallback / general ---
        else:
            return _make_response(
                "general",
                "I received your message but it did not match a specific "
                "tool (calculate/keywords). Here is a general acknowledgment "
                f"of your query: '{query.strip()}'"
            )

    except Exception as e:
        # Catch-all safety net so the agent never crashes or drops
        # a response, per the routing/error-handling requirements.
        return _make_response("error", f"Unhandled agent failure: {e}")


# ---------------------------------------------------------------------------
# 4. Automated validation checks (array of test query strings)
# ---------------------------------------------------------------------------
def run_validation_checks():
    test_queries = [
        "calculate 12 + 8 * 2",
        "calculate 10 / 0",
        "calculate banana + 5",          # invalid expression -> error
        "give me keywords for artificial intelligence agent pipelines",
        "keywords",                       # keyword route, empty text -> error
        "hello, how are you today?",      # general fallback
        "",                                # empty query -> error
        "CALCULATE (5 + 5) * 3",           # tests case-insensitivity
    ]

    print("=" * 60)
    print("AUTOMATED VALIDATION CHECKS")
    print("=" * 60)
    for q in test_queries:
        response = agent(q)
        # Verify schema validity before printing (required keys present)
        assert "type" in response and "result" in response, \
            f"Schema violation for query: {q!r}"
        print(f"Query: {q!r}")
        print(json.dumps(response, indent=2))
        print("-" * 60)


# ---------------------------------------------------------------------------
# 5. Interactive loop
# ---------------------------------------------------------------------------
def interactive_loop():
    print("=" * 60)
    print("INTERACTIVE AGENT (type 'exit' or 'quit' to stop)")
    print("=" * 60)
    while True:
        try:
            user_input = input("\nEnter your query: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting agent.")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            print("Exiting agent.")
            break

        response = agent(user_input)
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    run_validation_checks()
    interactive_loop()
