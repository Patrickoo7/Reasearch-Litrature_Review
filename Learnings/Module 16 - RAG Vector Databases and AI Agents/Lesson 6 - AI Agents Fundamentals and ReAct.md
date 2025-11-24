# Lesson 6: AI Agents Fundamentals & ReAct Pattern 🤖

**Module 16: RAG, Vector Databases & AI Agents | Lesson 6 of 8**

Master autonomous AI agents - from basic loops to the ReAct pattern powering modern agentic systems!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand what AI agents are and how they differ from simple LLM calls
2. ✅ Master agent architectures: perception, reasoning, action
3. ✅ Implement the ReAct (Reasoning + Acting) pattern
4. ✅ Build planning algorithms for multi-step tasks
5. ✅ Create memory systems (short-term and long-term)
6. ✅ Implement tool use foundations
7. ✅ Design agent control flow and decision loops
8. ✅ Handle errors and implement fallback strategies

---

## Prerequisites

- **Module 15**: Large Language Models basics
- Understanding of LLM prompting
- Python and async programming basics
- Familiarity with APIs

---

## 1. What Are AI Agents?

### Agent Definition

```python
# Example 1: Simple LLM call vs AI Agent
"""
Simple LLM Call:
  User: "What's the weather in Paris?"
  LLM: "I don't have access to real-time weather data..."

AI Agent:
  User: "What's the weather in Paris?"
  Agent:
    1. Realizes it needs current data
    2. Uses weather API tool
    3. Calls API with location="Paris"
    4. Receives: "Sunny, 22°C"
    5. Formats response: "The weather in Paris is sunny with 22°C"

Key Difference: Agents can take ACTIONS to accomplish goals!
"""

class SimpleAgent:
    """
    Basic agent structure.

    Components:
    1. Perception: Observe environment/input
    2. Reasoning: Decide what to do
    3. Action: Execute tools/functions
    4. Memory: Remember past interactions
    """

    def __init__(self, llm, tools):
        self.llm = llm  # Language model for reasoning
        self.tools = tools  # Available actions
        self.memory = []  # Conversation history

    def perceive(self, user_input):
        """Perceive user input"""
        return {
            'type': 'user_message',
            'content': user_input
        }

    def reason(self, perception):
        """Decide what action to take"""
        # Build prompt with context
        prompt = self._build_prompt(perception)

        # Get LLM decision
        response = self.llm.generate(prompt)

        return response

    def act(self, decision):
        """Execute action based on decision"""
        # Parse decision to extract tool call
        if 'tool:' in decision:
            tool_name = decision.split('tool:')[1].split()[0]
            if tool_name in self.tools:
                result = self.tools[tool_name].execute()
                return result

        return decision

    def _build_prompt(self, perception):
        """Build prompt with memory and available tools"""
        prompt = "You are a helpful AI agent.\n\n"
        prompt += "Available tools:\n"
        for name, tool in self.tools.items():
            prompt += f"- {name}: {tool.description}\n"
        prompt += f"\nUser: {perception['content']}\n"
        prompt += "Agent:"
        return prompt

print("AI Agent Characteristics:")
print("\n1. ✅ Autonomous: Makes decisions independently")
print("2. ✅ Goal-oriented: Works toward objectives")
print("3. ✅ Reactive: Responds to environment")
print("4. ✅ Proactive: Takes initiative")
print("5. ✅ Tool use: Interacts with external systems")
print("6. ✅ Memory: Learns from past interactions")
```

### Agent vs Traditional Systems

```python
# Example 2: Comparison table
"""
┌────────────────────┬──────────────────┬──────────────────┐
│     Feature        │   Traditional    │   AI Agent       │
├────────────────────┼──────────────────┼──────────────────┤
│ Decision Making    │ Hard-coded rules │ LLM reasoning    │
│ Flexibility        │ Fixed workflows  │ Adaptive         │
│ Tool Use           │ Pre-programmed   │ Dynamic          │
│ Error Handling     │ Try-catch        │ Reasoning        │
│ Learning           │ No               │ In-context       │
│ Natural Language   │ No               │ Yes              │
└────────────────────┴──────────────────┴──────────────────┘

Traditional System:
  IF weather_request THEN call_weather_api()
  ELSE IF math_request THEN call_calculator()
  ...

AI Agent:
  "I need weather data for Paris"
  → Reason: "I should use the weather API tool"
  → Act: Call weather_api(location="Paris")
  → Respond: "The weather is..."
"""

print("Why AI Agents Matter:")
print("\n1. Handle unpredictable requests")
print("2. Combine multiple tools dynamically")
print("3. Natural language interface")
print("4. Reduce hard-coding")
print("5. More robust to edge cases")
```

---

## 2. Agent Architecture

### Basic Agent Loop

```python
# Example 3: Core agent loop
class AgentLoop:
    """
    Basic agent execution loop.

    Loop:
      1. Perceive (get input)
      2. Think (reason about actions)
      3. Act (execute tool/respond)
      4. Repeat until done
    """

    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.memory = []

    def run(self, task):
        """
        Execute agent loop until task is complete.
        """
        print(f"🎯 Task: {task}\n")

        for iteration in range(self.max_iterations):
            print(f"--- Iteration {iteration + 1} ---")

            # 1. Think: What should I do next?
            thought = self.think(task)
            print(f"💭 Thought: {thought['reasoning']}")

            # 2. Act: Execute the action
            if thought['action'] == 'FINISH':
                print(f"✅ Final Answer: {thought['response']}")
                return thought['response']

            action_result = self.act(thought['action'])
            print(f"🔧 Action Result: {action_result}")

            # 3. Observe: Update memory
            self.memory.append({
                'thought': thought,
                'result': action_result
            })

            print()

        return "Max iterations reached without completing task"

    def think(self, task):
        """
        Reason about next action.

        Returns: {
            'reasoning': str,
            'action': str (tool name or 'FINISH'),
            'parameters': dict,
            'response': str (if FINISH)
        }
        """
        # Build prompt with memory
        prompt = f"Task: {task}\n\n"

        if self.memory:
            prompt += "Previous steps:\n"
            for i, step in enumerate(self.memory):
                prompt += f"{i+1}. {step['thought']['reasoning']} → {step['result']}\n"

        prompt += "\nWhat should I do next?\n"

        # Mock LLM response for demo
        # In production: call actual LLM
        mock_response = {
            'reasoning': 'I need to check the weather',
            'action': 'weather_api',
            'parameters': {'location': 'Paris'}
        }

        return mock_response

    def act(self, action):
        """Execute action"""
        if action in self.tools:
            return self.tools[action].execute()
        return f"Tool {action} not found"

# Demo concept
print("Agent Loop Pattern:")
print("\nloop until task_complete:")
print("  1. Perceive current state")
print("  2. Think about next action")
print("  3. Act (use tool or respond)")
print("  4. Update memory")
print("\n✅ Simple but powerful!")
```

---

## 3. ReAct: Reasoning + Acting

### ReAct Pattern Explained

```python
# Example 4: ReAct pattern
"""
ReAct: Interleave REASONING and ACTING

Traditional Chain of Thought (CoT):
  Thought → Thought → Thought → Answer

ReAct:
  Thought → Action → Observation →
  Thought → Action → Observation →
  Thought → Answer

Example Task: "What's the weather in Paris and convert temp to Fahrenheit?"

ReAct Trace:
  Thought 1: I need to get current weather in Paris
  Action 1: weather_api(location="Paris")
  Observation 1: Temperature is 22°C

  Thought 2: Now I need to convert 22°C to Fahrenheit
  Action 2: calculator(expression="22 * 9/5 + 32")
  Observation 2: Result is 71.6°F

  Thought 3: I have all the information needed
  Answer: The weather in Paris is 22°C (71.6°F)

Benefits:
✅ Interpretable: See agent's reasoning
✅ Debuggable: Identify where agent went wrong
✅ Grounded: Actions provide real observations
✅ Robust: Can recover from errors
"""

print("ReAct Pattern:")
print("\nKey Insight: Alternate reasoning and acting!")
print("\nThought → Action → Observation → Thought → Action → ...")
print("\n✅ More reliable than pure CoT")
print("✅ Used by: LangChain, AutoGPT, BabyAGI")
```

### ReAct Implementation

```python
# Example 5: ReAct agent implementation
class ReActAgent:
    """
    ReAct agent: Reasoning + Acting.

    Format:
      Thought: [reasoning]
      Action: [tool_name] [parameters]
      Observation: [result]
      ... (repeat)
      Thought: I now know the final answer
      Final Answer: [response]
    """

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, question, max_steps=5):
        """
        Execute ReAct loop.
        """
        trajectory = []  # Store thought-action-observation
        prompt = self._build_initial_prompt(question)

        for step in range(max_steps):
            print(f"\n=== Step {step + 1} ===")

            # Generate next thought and action
            response = self._generate_response(prompt)

            # Parse response
            parsed = self._parse_response(response)

            if parsed['type'] == 'answer':
                # Done!
                print(f"\n✅ Final Answer: {parsed['content']}")
                return parsed['content']

            elif parsed['type'] == 'action':
                # Execute action
                thought = parsed['thought']
                action = parsed['action']
                params = parsed['parameters']

                print(f"💭 Thought: {thought}")
                print(f"🔧 Action: {action}({params})")

                # Execute tool
                observation = self._execute_tool(action, params)
                print(f"👁️  Observation: {observation}")

                # Add to trajectory
                trajectory.append({
                    'thought': thought,
                    'action': action,
                    'observation': observation
                })

                # Update prompt with observation
                prompt = self._update_prompt(prompt, thought, action, observation)

        return "Max steps reached"

    def _build_initial_prompt(self, question):
        """Build initial ReAct prompt"""
        prompt = f"""Answer the following question using this format:

Thought: [your reasoning about what to do next]
Action: [tool_name] [parameters]
Observation: [result from tool]
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: [your final answer]

Available tools:
"""
        for name, tool in self.tools.items():
            prompt += f"- {name}: {tool.description}\n"

        prompt += f"\nQuestion: {question}\n\n"
        return prompt

    def _generate_response(self, prompt):
        """
        Generate next thought/action from LLM.

        In production: Call actual LLM API
        """
        # Mock response for demo
        return """Thought: I need to check the current weather
Action: weather_api location=Paris"""

    def _parse_response(self, response):
        """Parse LLM response into structured format"""
        lines = response.strip().split('\n')

        result = {'type': 'action'}

        for line in lines:
            if line.startswith('Thought:'):
                result['thought'] = line.replace('Thought:', '').strip()
            elif line.startswith('Action:'):
                action_str = line.replace('Action:', '').strip()
                parts = action_str.split(maxsplit=1)
                result['action'] = parts[0]
                result['parameters'] = parts[1] if len(parts) > 1 else {}
            elif line.startswith('Final Answer:'):
                result['type'] = 'answer'
                result['content'] = line.replace('Final Answer:', '').strip()

        return result

    def _execute_tool(self, tool_name, parameters):
        """Execute tool and return observation"""
        if tool_name in self.tools:
            return self.tools[tool_name].execute(parameters)
        return f"Error: Tool {tool_name} not found"

    def _update_prompt(self, prompt, thought, action, observation):
        """Add latest step to prompt"""
        prompt += f"Thought: {thought}\n"
        prompt += f"Action: {action}\n"
        prompt += f"Observation: {observation}\n\n"
        return prompt

# Demonstration
print("\nReAct Agent Example:")
print("\nQuestion: What's 25 * 4 + 10?")
print("\nExpected ReAct Trace:")
print("  Thought: I need to calculate 25 * 4 first")
print("  Action: calculator 25*4")
print("  Observation: 100")
print("  Thought: Now I need to add 10")
print("  Action: calculator 100+10")
print("  Observation: 110")
print("  Thought: I now know the final answer")
print("  Final Answer: 110")
```

---

## 4. Planning Algorithms

### Chain of Thought Planning

```python
# Example 6: Chain of Thought for planning
class ChainOfThoughtPlanner:
    """
    Break complex task into steps before executing.

    Plan → Execute → Verify
    """

    def __init__(self, llm):
        self.llm = llm

    def plan(self, task):
        """
        Generate step-by-step plan.

        In production: Use LLM to generate plan
        """
        # Mock plan for demo
        plan = [
            "1. Identify required information",
            "2. Gather data using appropriate tools",
            "3. Process and analyze data",
            "4. Format final response"
        ]

        return plan

    def execute_plan(self, plan, tools):
        """Execute each step in plan"""
        results = []

        for step in plan:
            print(f"\n📋 Executing: {step}")

            # Determine which tool to use (simplified)
            # In production: LLM decides which tool
            result = "Step completed"  # Mock

            results.append(result)
            print(f"   ✅ Result: {result}")

        return results

# Demo
planner = ChainOfThoughtPlanner(llm=None)

task = "Analyze weather trends and recommend clothing"
plan = planner.plan(task)

print("Chain of Thought Planning:")
print(f"\nTask: {task}")
print("\nGenerated Plan:")
for step in plan:
    print(f"  {step}")
```

### Tree of Thoughts (ToT)

```python
# Example 7: Tree of Thoughts for complex reasoning
"""
Tree of Thoughts: Explore multiple reasoning paths

      Thought 1
      /   |   \\
   T1.1 T1.2 T1.3
   /  \\   |
T1.1.1 T1.1.2 T1.2.1

Process:
1. Generate multiple thoughts (branching)
2. Evaluate each thought
3. Expand promising paths
4. Prune poor paths
5. Return best solution

Use cases:
- Creative writing
- Complex problem solving
- Strategic planning
"""

class TreeOfThoughts:
    """
    Tree of Thoughts: Explore multiple reasoning paths.
    """

    def __init__(self, llm, beam_width=3, max_depth=3):
        self.llm = llm
        self.beam_width = beam_width  # How many branches to explore
        self.max_depth = max_depth

    def solve(self, problem):
        """
        Solve problem using tree search.
        """
        # Start with initial thought
        root = {'thought': problem, 'score': 0, 'depth': 0}

        # Best-first search
        frontier = [root]
        best_solution = None
        best_score = float('-inf')

        while frontier:
            # Get most promising node
            current = max(frontier, key=lambda x: x['score'])
            frontier.remove(current)

            # Check if solution
            if self._is_solution(current):
                if current['score'] > best_score:
                    best_solution = current
                    best_score = current['score']
                continue

            # Don't expand if too deep
            if current['depth'] >= self.max_depth:
                continue

            # Generate child thoughts
            children = self._generate_thoughts(current)

            # Evaluate and add to frontier
            for child in children:
                child['score'] = self._evaluate(child)
                frontier.append(child)

            # Prune: keep only top beam_width
            frontier = sorted(frontier, key=lambda x: x['score'], reverse=True)[:self.beam_width]

        return best_solution

    def _generate_thoughts(self, node):
        """Generate possible next thoughts"""
        # Mock: generate 3 children
        return [
            {'thought': f"{node['thought']} → idea A", 'depth': node['depth'] + 1},
            {'thought': f"{node['thought']} → idea B", 'depth': node['depth'] + 1},
            {'thought': f"{node['thought']} → idea C", 'depth': node['depth'] + 1}
        ]

    def _evaluate(self, node):
        """Score a thought (higher = better)"""
        # Mock evaluation
        import random
        return random.random()

    def _is_solution(self, node):
        """Check if node is a valid solution"""
        return node['depth'] == self.max_depth

print("\nTree of Thoughts:")
print("  ✅ Explores multiple reasoning paths")
print("  ✅ Better for complex problems")
print("  ⚠️  More expensive (multiple LLM calls)")
```

---

## 5. Memory Systems

### Short-Term Memory (Conversation History)

```python
# Example 8: Short-term memory implementation
class ShortTermMemory:
    """
    Store recent conversation history.

    Implementation: Simple list with max length.
    """

    def __init__(self, max_messages=10):
        self.max_messages = max_messages
        self.messages = []

    def add(self, role, content):
        """Add message to memory"""
        self.messages.append({
            'role': role,  # 'user', 'assistant', 'system'
            'content': content
        })

        # Trim if too long
        if len(self.messages) > self.max_messages:
            # Keep system message, trim oldest user/assistant
            system_msgs = [m for m in self.messages if m['role'] == 'system']
            other_msgs = [m for m in self.messages if m['role'] != 'system']

            # Keep recent messages
            other_msgs = other_msgs[-self.max_messages+len(system_msgs):]

            self.messages = system_msgs + other_msgs

    def get_context(self):
        """Get messages for LLM context"""
        return self.messages

    def clear(self):
        """Clear memory"""
        self.messages = []

# Demo
memory = ShortTermMemory(max_messages=5)

# Simulate conversation
memory.add('system', 'You are a helpful assistant')
memory.add('user', 'What is 2+2?')
memory.add('assistant', '2+2 equals 4')
memory.add('user', 'What about 3+3?')
memory.add('assistant', '3+3 equals 6')

print("Short-Term Memory:")
print("\nConversation history:")
for msg in memory.get_context():
    print(f"  {msg['role']}: {msg['content']}")
```

### Long-Term Memory (Vector Store)

```python
# Example 9: Long-term memory with vector store
from sentence_transformers import SentenceTransformer
import numpy as np

class LongTermMemory:
    """
    Store agent experiences in vector database.

    Use case: Remember past tasks, user preferences, learned facts.
    """

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memories = []  # List of memory dicts
        self.embeddings = None

    def store(self, memory_text, metadata=None):
        """Store a memory"""
        memory = {
            'text': memory_text,
            'metadata': metadata or {},
            'timestamp': np.datetime64('now')
        }

        self.memories.append(memory)

        # Update embeddings
        all_texts = [m['text'] for m in self.memories]
        self.embeddings = self.model.encode(all_texts, normalize_embeddings=True)

    def recall(self, query, top_k=3):
        """Recall relevant memories"""
        if not self.memories:
            return []

        # Embed query
        query_emb = self.model.encode(query, normalize_embeddings=True)

        # Find similar memories
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(-similarities)[:top_k]

        return [self.memories[i] for i in top_indices]

# Demo
ltm = LongTermMemory()

# Store experiences
ltm.store("User prefers metric units over imperial", metadata={'type': 'preference'})
ltm.store("Python is user's favorite programming language", metadata={'type': 'preference'})
ltm.store("User works on machine learning projects", metadata={'type': 'context'})

print("\n\nLong-Term Memory:")
print("\nStored 3 memories")

# Recall
query = "What programming language does user like?"
relevant = ltm.recall(query, top_k=2)

print(f"\nQuery: '{query}'")
print("Recalled memories:")
for mem in relevant:
    print(f"  - {mem['text']}")
```

### Episodic Memory

```python
# Example 10: Episodic memory (task episodes)
class EpisodicMemory:
    """
    Store complete task episodes for learning.

    Episode = (task, actions taken, result, success/failure)
    """

    def __init__(self):
        self.episodes = []

    def store_episode(self, task, actions, result, success):
        """Store complete task episode"""
        episode = {
            'task': task,
            'actions': actions,  # List of actions taken
            'result': result,
            'success': success,
            'timestamp': np.datetime64('now')
        }

        self.episodes.append(episode)

    def recall_similar_episodes(self, current_task, top_k=3):
        """Find similar past episodes (simplified similarity)"""
        # In production: use embeddings for similarity
        # For demo: return most recent successful episodes

        successful = [e for e in self.episodes if e['success']]
        return successful[-top_k:] if successful else []

# Demo
episodic = EpisodicMemory()

# Store episode
episodic.store_episode(
    task="Calculate weather statistics",
    actions=["get_weather", "calculate_average", "format_response"],
    result="Successfully calculated average temperature",
    success=True
)

print("\n\nEpisodic Memory:")
print("  ✅ Store complete task traces")
print("  ✅ Learn from past successes/failures")
print("  ✅ Few-shot learning for agents")
```

---

## 6. Tool Use Foundations

### Tool Definition

```python
# Example 11: Tool abstraction
from typing import Callable, Dict, Any

class Tool:
    """
    Base class for agent tools.

    Tools encapsulate external actions the agent can take.
    """

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters_schema: Dict[str, Any]
    ):
        self.name = name
        self.description = description
        self.function = function
        self.parameters_schema = parameters_schema

    def execute(self, **kwargs):
        """Execute tool with parameters"""
        try:
            result = self.function(**kwargs)
            return {
                'success': True,
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_schema(self):
        """Get tool schema for LLM"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters_schema
        }

# Example tool
def calculator_func(expression: str) -> float:
    """Safely evaluate math expression"""
    # In production: use safer evaluation
    return eval(expression)

calculator_tool = Tool(
    name="calculator",
    description="Perform mathematical calculations",
    function=calculator_func,
    parameters_schema={
        'expression': {
            'type': 'string',
            'description': 'Math expression to evaluate'
        }
    }
)

# Use tool
result = calculator_tool.execute(expression="25 * 4")
print("\nTool Execution:")
print(f"  Tool: {calculator_tool.name}")
print(f"  Input: 25 * 4")
print(f"  Result: {result}")
```

---

## 7. Error Handling & Fallbacks

### Retry Logic

```python
# Example 12: Error handling and retries
class RobustAgent:
    """
    Agent with error handling and retry logic.
    """

    def __init__(self, llm, tools, max_retries=3):
        self.llm = llm
        self.tools = tools
        self.max_retries = max_retries

    def execute_with_retry(self, action, parameters):
        """Execute action with retry logic"""
        for attempt in range(self.max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{self.max_retries}")

                # Execute action
                result = self.tools[action].execute(**parameters)

                if result['success']:
                    return result

                # Failed but no exception - retry with modification
                print(f"  ⚠️  Failed: {result['error']}")

                # Modify parameters or try different approach
                # In production: use LLM to adjust strategy

            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")

                if attempt == self.max_retries - 1:
                    # Last attempt - use fallback
                    return self._fallback(action, parameters, e)

        return {'success': False, 'error': 'Max retries exceeded'}

    def _fallback(self, action, parameters, error):
        """Fallback strategy when all retries fail"""
        print("  🔄 Using fallback strategy")

        # Options:
        # 1. Use alternative tool
        # 2. Ask user for help
        # 3. Gracefully degrade
        # 4. Return partial result

        return {
            'success': False,
            'error': f'Failed after retries: {str(error)}',
            'fallback': 'Asked user for clarification'
        }

print("\nError Handling Strategies:")
print("  1. Retry with exponential backoff")
print("  2. Try alternative tools")
print("  3. Ask user for clarification")
print("  4. Graceful degradation")
print("  5. Return partial results")
```

---

## Practice Exercises

### Exercise 1: Build ReAct Agent
Implement a complete ReAct agent with 3+ tools (calculator, weather, search).

### Exercise 2: Memory Integration
Add both short-term and long-term memory to an agent.

### Exercise 3: Planning System
Implement a planner that breaks complex tasks into steps before executing.

### Exercise 4: Error Recovery
Build an agent that recovers gracefully from tool failures.

### Exercise 5: Tool Composition
Create an agent that chains multiple tools to solve multi-step problems.

---

## Key Takeaways

1. **AI agents** are autonomous systems that reason and act to achieve goals
2. **ReAct pattern** interleaves reasoning and acting for better reliability
3. **Agent loop**: Perceive → Think → Act → Repeat
4. **Planning** improves success on complex multi-step tasks
5. **Short-term memory** maintains conversation context
6. **Long-term memory** (vector store) enables learning from experiences
7. **Tools** encapsulate external actions agents can take
8. **Error handling** with retries and fallbacks is essential
9. **Episodic memory** enables few-shot learning from past tasks
10. **Agents vs LLMs**: Agents take actions, LLMs just respond

---

## Further Reading

### Papers
- **ReAct**: Yao et al. (2023) - https://arxiv.org/abs/2210.03629
- **Tree of Thoughts**: Yao et al. (2023) - https://arxiv.org/abs/2305.10601
- **Reflexion**: Shinn et al. (2023) - https://arxiv.org/abs/2303.11366

### Documentation
- **LangChain Agents**: https://python.langchain.com/docs/modules/agents/
- **AutoGPT**: https://github.com/Significant-Gravitas/AutoGPT

### Cross-References
- **Module 16 Lesson 7**: Tool Use & Function Calling (next lesson)
- **Module 15**: LLM Fundamentals

---

**Next Lesson**: Tool Use & Function Calling - Master OpenAI/Anthropic tool calling and safe execution!
