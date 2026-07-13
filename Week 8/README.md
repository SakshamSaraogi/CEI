# Week 8 — Task-Routing Agent Pipeline

A single-agent system that routes incoming text queries to one of three
tools — a calculator, a keyword extractor, or a general fallback handler —
based on explicit keyword checks, and returns every response in a
consistent JSON schema.

## Files

| File | Description |
|---|---|
| `agent_pipeline.py` | The full agent implementation, automated validation checks, and an interactive test loop. |
| `week_8_quiz_answers.docx` | Written answers to the Week 8 conceptual quiz (stateful graphs, nodes/edges, conditional routing, etc.). |

## Requirements

- Python 3.8+
- No external dependencies — uses only the Python standard library (`json`, `re`, `ast`, `operator`).

## How to run

```bash
python agent_pipeline.py
```

Running the script does two things, in order:

### 1. Automated validation checks (runs immediately)

An array of 8 predefined test queries is passed through `agent()` and each
JSON response is printed to the terminal. This covers:

- A valid calculation (`"calculate 12 + 8 * 2"`)
- A calculation error — division by zero (`"calculate 10 / 0"`)
- A calculation error — invalid expression (`"calculate banana + 5"`)
- A valid keyword extraction query
- A keyword extraction error — no meaningful text supplied
- A general/fallback query (plain conversational text)
- An empty query (error case)
- Case-insensitivity check (`"CALCULATE (5 + 5) * 3"`)

Each response is asserted to contain both required keys (`type` and
`result`) before being printed, so if the schema were ever violated the
script would fail loudly right here rather than passing silently.

### 2. Interactive loop

After the automated checks finish, the script drops into a live prompt:

```
Enter your query:
```

Type any query and press Enter to see its JSON response. Type `exit` or
`quit` to stop the loop.

**Try these to test each route:**

```
calculate (45 + 15) / 4
calculate 10 / 0
keywords for renewable energy and climate policy
tell me a joke
calculate banana
```

## Design overview

- **`calculator(expression)`** — evaluates basic arithmetic safely using
  Python's `ast` module rather than raw `eval()`, so arbitrary code can't
  be executed via user input. Only numbers and `+ - * / % **` are
  permitted; anything else raises a `ValueError`.
- **`extract_keywords(text)`** — lowercases the input, strips punctuation,
  removes a small stopword list, and returns unique keywords in order of
  first appearance.
- **`agent(query)`** — the router. Checks the lowercased query string for
  `"calculate"` or `"keywords"` and dispatches to the matching tool;
  anything else goes to a general fallback response. Every branch is
  wrapped in error handling so a bad input never crashes the agent — it
  always returns a structured response instead.
- **Response schema** — every single return value, across all four
  possible outcomes, is built through one shared `_make_response()`
  helper, guaranteeing the shape:

  ```json
  {"type": "calculation | keywords | general | error", "result": ...}
  ```

## Example outputs

```json
{"type": "calculation", "result": 30}
```

```json
{"type": "keywords", "result": ["renewable", "energy", "climate", "policy"]}
```

```json
{"type": "error", "result": "Division by zero"}
```

```json
{"type": "general", "result": "I received your message but it did not match a specific tool (calculate/keywords). Here is a general acknowledgment of your query: 'tell me a joke'"}
```
