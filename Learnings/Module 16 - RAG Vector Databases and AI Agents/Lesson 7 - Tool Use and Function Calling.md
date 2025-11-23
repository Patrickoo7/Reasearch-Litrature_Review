# Lesson 7: Tool Use & Function Calling 🔧

**Module 16: RAG, Vector Databases & AI Agents | Lesson 7 of 8**

Master production-grade tool use - from OpenAI/Anthropic function calling to safe execution and error handling!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Master OpenAI function calling API
2. ✅ Use Anthropic Claude tool use
3. ✅ Define tool schemas and JSON validation
4. ✅ Implement parameter extraction and validation
5. ✅ Build safe tool execution with sandboxing and timeouts
6. ✅ Create production tools: calculator, web search, database query
7. ✅ Handle API integration patterns
8. ✅ Implement error handling and retry logic
9. ✅ Build tool selection and routing systems

---

## Prerequisites

- **Module 16 Lesson 6**: AI Agents Fundamentals (REQUIRED)
- API keys for OpenAI or Anthropic (for production examples)
- Understanding of JSON schemas
- Python asyncio basics

---

## 1. OpenAI Function Calling

### Basic Function Calling

```python
# Example 1: OpenAI function calling basics
"""
# Requires: pip install openai

import openai
import json

# Define functions the model can call
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]

# User message
messages = [
    {"role": "user", "content": "What's the weather like in Boston?"}
]

# Call OpenAI with functions
response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    functions=functions,
    function_call="auto"  # Let model decide when to call
)

# Check if function was called
message = response.choices[0].message

if message.function_call:
    # Model wants to call a function
    function_name = message.function_call.name
    function_args = json.loads(message.function_call.arguments)

    print(f"Function: {function_name}")
    print(f"Arguments: {function_args}")

    # Execute function
    if function_name == "get_current_weather":
        result = get_current_weather(**function_args)

        # Send result back to model
        messages.append(message)
        messages.append({
            "role": "function",
            "name": function_name,
            "content": json.dumps(result)
        })

        # Get final response
        second_response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        print(second_response.choices[0].message.content)
"""

print("OpenAI Function Calling Flow:")
print("\n1. Define function schemas (JSON)")
print("2. Send to OpenAI with messages")
print("3. Model decides to call function")
print("4. Extract function name + arguments")
print("5. Execute function locally")
print("6. Send result back to model")
print("7. Get final natural language response")
```

### Multiple Functions

```python
# Example 2: Multiple function definitions
"""
# Define multiple tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression like '2 + 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

# Model chooses appropriate function based on query
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's 25 * 4 + 10?"}],
    tools=tools,
    tool_choice="auto"
)

# Model will choose 'calculate' function for math query
"""

print("\nMultiple Functions:")
print("  ✅ Model selects appropriate tool")
print("  ✅ Based on function descriptions")
print("  ✅ Extracts correct parameters")
```

### Parallel Function Calling

```python
# Example 3: Parallel function calls (GPT-4+)
"""
# GPT-4 can call multiple functions in one request

messages = [{
    "role": "user",
    "content": "What's the weather in NYC and LA, and calculate 50+50?"
}]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)

# response.choices[0].message.tool_calls is a list
tool_calls = response.choices[0].message.tool_calls

# Process all tool calls
for tool_call in tool_calls:
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    print(f"Call: {function_name}({arguments})")

    # Execute each function
    result = execute_function(function_name, arguments)

    # Add result to messages
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

# Get final response with all results
final_response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages
)
"""

print("\nParallel Function Calling:")
print("  ✅ Multiple functions in single request")
print("  ✅ Faster than sequential")
print("  ✅ Better user experience")
```

---

## 2. Anthropic Claude Tool Use

### Claude Tool Use API

```python
# Example 4: Anthropic Claude tool use
"""
# Requires: pip install anthropic

import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

# Define tools (similar to OpenAI but slightly different format)
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
]

# Call Claude
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ]
)

# Check for tool use
if message.stop_reason == "tool_use":
    for block in message.content:
        if block.type == "tool_use":
            tool_name = block.name
            tool_input = block.input

            print(f"Tool: {tool_name}")
            print(f"Input: {tool_input}")

            # Execute tool
            result = execute_tool(tool_name, tool_input)

            # Send result back
            messages = [
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        }
                    ]
                }
            ]

            # Get final response
            final = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                tools=tools,
                messages=messages
            )

            print(final.content[0].text)
"""

print("Claude Tool Use:")
print("  - Similar to OpenAI but different schema")
print("  - input_schema instead of parameters")
print("  - tool_result instead of function role")
print("  - Supports thinking blocks (Claude's reasoning)")
```

---

## 3. Tool Schema Design

### JSON Schema Best Practices

```python
# Example 5: Well-designed tool schemas
from typing import Dict, Any

def create_tool_schema(
    name: str,
    description: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create well-formed tool schema.

    Best Practices:
    1. Clear, specific descriptions
    2. Descriptive parameter names
    3. Specify types and constraints
    4. Mark required parameters
    5. Provide examples in descriptions
    6. Use enums for limited choices
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    }

# Example: Good schema
good_calculator_schema = create_tool_schema(
    name="calculate_expression",
    description=(
        "Safely evaluate a mathematical expression and return the result. "
        "Supports basic operations (+, -, *, /, **) and common functions (sqrt, sin, cos). "
        "Example: '2 * (3 + 4)' returns 14"
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "Mathematical expression to evaluate. "
                    "Use standard Python syntax. "
                    "Examples: '25 * 4', 'sqrt(16)', 'sin(3.14159/2)'"
                )
            }
        },
        "required": ["expression"]
    }
)

# Example: Bad schema (vague, unclear)
bad_calculator_schema = {
    "name": "calc",  # Too short, unclear
    "description": "Does math",  # Too vague
    "parameters": {
        "type": "object",
        "properties": {
            "x": {"type": "string"}  # Unclear what x is
        }
    }
}

print("Tool Schema Best Practices:")
print("\n✅ Good:")
print("  - Descriptive names (calculate_expression)")
print("  - Detailed descriptions with examples")
print("  - Clear parameter descriptions")

print("\n❌ Bad:")
print("  - Vague names (calc)")
print("  - Minimal descriptions")
print("  - Unclear parameters")
```

### Complex Schema with Validation

```python
# Example 6: Complex schema with nested objects and enums
database_query_schema = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": (
            "Query the customer database with filters. "
            "Returns matching customer records."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "enum": ["customers", "orders", "products"],
                    "description": "Database table to query"
                },
                "filters": {
                    "type": "object",
                    "description": "Filters to apply to query",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active", "inactive", "pending"],
                            "description": "Filter by status"
                        },
                        "created_after": {
                            "type": "string",
                            "description": "ISO date string (YYYY-MM-DD)"
                        },
                        "min_amount": {
                            "type": "number",
                            "description": "Minimum order amount in dollars"
                        }
                    }
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["table"]
        }
    }
}

print("\nComplex Schema Features:")
print("  ✅ Enums for limited choices")
print("  ✅ Nested objects (filters)")
print("  ✅ Number constraints (min/max)")
print("  ✅ Default values")
print("  ✅ Required vs optional parameters")
```

---

## 4. Parameter Validation

### Validate Function Arguments

```python
# Example 7: Validate parameters before execution
import jsonschema
from jsonschema import validate, ValidationError

class ParameterValidator:
    """Validate function call parameters against schema"""

    @staticmethod
    def validate_parameters(parameters: Dict, schema: Dict) -> tuple[bool, str]:
        """
        Validate parameters against JSON schema.

        Returns: (is_valid, error_message)
        """
        try:
            validate(instance=parameters, schema=schema)
            return True, ""
        except ValidationError as e:
            return False, str(e)

# Example
schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "minLength": 1},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
}

# Valid parameters
valid_params = {"location": "Paris", "unit": "celsius"}
is_valid, error = ParameterValidator.validate_parameters(valid_params, schema)
print(f"Valid params: {is_valid}")  # True

# Invalid parameters (missing required)
invalid_params = {"unit": "celsius"}
is_valid, error = ParameterValidator.validate_parameters(invalid_params, schema)
print(f"Invalid params: {is_valid}")  # False
print(f"Error: {error}")

print("\nValidation Catches:")
print("  ✅ Missing required parameters")
print("  ✅ Wrong types (string vs number)")
print("  ✅ Invalid enum values")
print("  ✅ Range violations (min/max)")
```

---

## 5. Safe Tool Execution

### Sandboxed Execution

```python
# Example 8: Safe calculator with restricted eval
import ast
import operator
import math

class SafeCalculator:
    """
    Safe mathematical expression evaluator.

    Prevents code injection while allowing math operations.
    """

    # Allowed operations
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    # Allowed functions
    SAFE_FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'abs': abs,
        'round': round,
    }

    @staticmethod
    def safe_eval(expression: str) -> float:
        """
        Safely evaluate mathematical expression.

        Raises ValueError if expression contains unsafe operations.
        """
        try:
            # Parse expression to AST
            node = ast.parse(expression, mode='eval').body

            # Evaluate AST
            return SafeCalculator._eval_node(node)

        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

    @staticmethod
    def _eval_node(node):
        """Recursively evaluate AST node"""
        if isinstance(node, ast.Num):
            # Number literal
            return node.n

        elif isinstance(node, ast.BinOp):
            # Binary operation (+, -, *, /, **)
            if type(node.op) not in SafeCalculator.SAFE_OPERATORS:
                raise ValueError(f"Unsafe operator: {type(node.op)}")

            left = SafeCalculator._eval_node(node.left)
            right = SafeCalculator._eval_node(node.right)
            op = SafeCalculator.SAFE_OPERATORS[type(node.op)]

            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            # Unary operation (-, +)
            if type(node.op) not in SafeCalculator.SAFE_OPERATORS:
                raise ValueError(f"Unsafe operator: {type(node.op)}")

            operand = SafeCalculator._eval_node(node.operand)
            op = SafeCalculator.SAFE_OPERATORS[type(node.op)]

            return op(operand)

        elif isinstance(node, ast.Call):
            # Function call
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")

            func_name = node.func.id
            if func_name not in SafeCalculator.SAFE_FUNCTIONS:
                raise ValueError(f"Unsafe function: {func_name}")

            # Evaluate arguments
            args = [SafeCalculator._eval_node(arg) for arg in node.args]

            # Call function
            func = SafeCalculator.SAFE_FUNCTIONS[func_name]
            return func(*args)

        else:
            raise ValueError(f"Unsafe node type: {type(node)}")

# Test safe calculator
calc = SafeCalculator()

print("\nSafe Calculator:")

# Safe expressions
safe_exprs = [
    "2 + 2",
    "25 * 4 + 10",
    "sqrt(16)",
    "sin(3.14159/2)"
]

for expr in safe_exprs:
    result = calc.safe_eval(expr)
    print(f"  ✅ {expr} = {result}")

# Unsafe expressions
unsafe_exprs = [
    "__import__('os').system('ls')",  # Code injection attempt
    "exec('print(1)')",  # Exec attempt
]

for expr in unsafe_exprs:
    try:
        calc.safe_eval(expr)
        print(f"  ❌ DANGER: {expr} was allowed!")
    except ValueError as e:
        print(f"  ✅ Blocked: {expr}")
```

### Timeout and Resource Limits

```python
# Example 9: Execute with timeout
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds}s")

    # Set alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable alarm
        signal.alarm(0)

def execute_with_timeout(func, args, timeout_seconds=5):
    """Execute function with timeout"""
    try:
        with timeout(timeout_seconds):
            result = func(*args)
            return {'success': True, 'result': result}
    except TimeoutException as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Example: Long-running function
def slow_function():
    import time
    time.sleep(10)  # Simulate slow operation
    return "Done"

# Execute with timeout
result = execute_with_timeout(slow_function, [], timeout_seconds=2)
print(f"\nTimeout test: {result}")
# Output: {'success': False, 'error': 'Operation timed out after 2s'}
```

---

## 6. Production Tools

### Web Search Tool

```python
# Example 10: Web search integration
"""
# Using Google Custom Search API or Bing API

import requests

class WebSearchTool:
    '''
    Search the web and return top results.
    '''

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, num_results: int = 3) -> list:
        '''
        Search web for query.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of {title, url, snippet} dicts
        '''
        # Example with Google Custom Search
        url = "https://www.googleapis.com/customsearch/v1"

        params = {
            'key': self.api_key,
            'cx': 'your-search-engine-id',
            'q': query,
            'num': num_results
        }

        response = requests.get(url, params=params)
        data = response.json()

        results = []
        for item in data.get('items', []):
            results.append({
                'title': item.get('title'),
                'url': item.get('link'),
                'snippet': item.get('snippet')
            })

        return results

# Tool schema
web_search_schema = {
    "name": "web_search",
    "description": "Search the web for current information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "num_results": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["query"]
    }
}
"""

print("Web Search Tool:")
print("  APIs: Google Custom Search, Bing, DuckDuckGo")
print("  Features: Title, URL, snippet")
print("  Rate limits: Monitor and handle")
```

### Database Query Tool

```python
# Example 11: Safe database query tool
"""
import sqlite3
from typing import List, Dict, Any

class DatabaseQueryTool:
    '''
    Safe database query tool with SQL injection protection.
    '''

    def __init__(self, db_path: str):
        self.db_path = db_path

    def query(
        self,
        table: str,
        columns: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 10
    ) -> List[Dict]:
        '''
        Query database safely with parameterized queries.

        Args:
            table: Table name (validated against whitelist)
            columns: Columns to select (None = all)
            filters: WHERE conditions
            limit: Max results

        Returns:
            List of result dicts
        '''
        # Validate table name (whitelist)
        allowed_tables = ['users', 'orders', 'products']
        if table not in allowed_tables:
            raise ValueError(f"Table {table} not allowed")

        # Build query with parameterized values
        cols = ', '.join(columns) if columns else '*'
        query = f"SELECT {cols} FROM {table}"

        params = []
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value)

            query += " WHERE " + " AND ".join(conditions)

        query += f" LIMIT {int(limit)}"

        # Execute safely
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return results

# Schema
db_query_schema = {
    "name": "query_database",
    "description": "Query database with filters",
    "parameters": {
        "type": "object",
        "properties": {
            "table": {
                "type": "string",
                "enum": ["users", "orders", "products"]
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"}
            },
            "filters": {
                "type": "object"
            },
            "limit": {
                "type": "integer",
                "default": 10
            }
        },
        "required": ["table"]
    }
}
"""

print("\nDatabase Query Tool:")
print("  ✅ Parameterized queries (prevent SQL injection)")
print("  ✅ Table whitelist")
print("  ✅ Result limit")
print("  ✅ Read-only access")
```

---

## 7. Error Handling & Retry

### Robust Error Handling

```python
# Example 12: Comprehensive error handling
class ToolExecutor:
    """Execute tools with robust error handling"""

    def __init__(self, tools, max_retries=3):
        self.tools = tools
        self.max_retries = max_retries

    def execute(self, tool_name: str, parameters: dict):
        """
        Execute tool with error handling and retries.
        """
        for attempt in range(self.max_retries):
            try:
                # Validate tool exists
                if tool_name not in self.tools:
                    return {
                        'success': False,
                        'error': f"Unknown tool: {tool_name}",
                        'error_type': 'TOOL_NOT_FOUND'
                    }

                tool = self.tools[tool_name]

                # Validate parameters
                is_valid, error_msg = self._validate_params(
                    parameters,
                    tool.schema['parameters']
                )

                if not is_valid:
                    return {
                        'success': False,
                        'error': f"Invalid parameters: {error_msg}",
                        'error_type': 'VALIDATION_ERROR'
                    }

                # Execute tool
                result = tool.execute(**parameters)

                return {
                    'success': True,
                    'result': result,
                    'attempts': attempt + 1
                }

            except TimeoutException:
                # Timeout - don't retry
                return {
                    'success': False,
                    'error': 'Tool execution timed out',
                    'error_type': 'TIMEOUT'
                }

            except Exception as e:
                # Other errors - retry
                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    return {
                        'success': False,
                        'error': str(e),
                        'error_type': 'EXECUTION_ERROR',
                        'attempts': attempt + 1
                    }

                # Exponential backoff
                import time
                time.sleep(2 ** attempt)

        return {
            'success': False,
            'error': 'Max retries exceeded',
            'error_type': 'MAX_RETRIES'
        }

    def _validate_params(self, params, schema):
        """Validate parameters (simplified)"""
        # In production: use jsonschema
        return True, ""

print("Error Handling:")
print("  ✅ Tool not found")
print("  ✅ Parameter validation errors")
print("  ✅ Execution errors with retry")
print("  ✅ Timeouts (no retry)")
print("  ✅ Exponential backoff")
```

---

## Practice Exercises

### Exercise 1: Build Tool Suite
Create 5 production tools (calculator, search, weather, database, file system).

### Exercise 2: OpenAI Integration
Integrate OpenAI function calling with your tools. Handle parallel calls.

### Exercise 3: Safe Execution
Implement sandboxing for code execution tools.

### Exercise 4: Error Recovery
Build robust error handling with retries and fallbacks.

### Exercise 5: Tool Monitoring
Add logging and metrics to track tool usage and errors.

---

## Key Takeaways

1. **OpenAI function calling** uses JSON schemas to define tools
2. **Anthropic Claude** tool use has similar but slightly different API
3. **Schema design** is critical - clear descriptions help models choose correctly
4. **Parameter validation** prevents errors before execution
5. **Safe execution** requires sandboxing and timeouts
6. **Production tools** need error handling and rate limiting
7. **Retry logic** with exponential backoff improves reliability
8. **Tool selection** is automatic based on descriptions
9. **Parallel calls** enable faster multi-tool workflows
10. **Monitoring** is essential for production systems

---

## Further Reading

### Documentation
- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling
- **Anthropic Tool Use**: https://docs.anthropic.com/claude/docs/tool-use
- **JSON Schema**: https://json-schema.org/

### Cross-References
- **Module 16 Lesson 6**: AI Agents Fundamentals (previous)
- **Module 16 Lesson 8**: Multi-Agent & Production RAG (next)

---

**Next Lesson**: Multi-Agent Systems & Production RAG - Build complex multi-agent workflows and deploy RAG at scale!
