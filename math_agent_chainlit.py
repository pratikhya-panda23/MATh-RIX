import chainlit as cl
from typing import Optional
from math_agent_workflow import create_math_agent
from chainlit.input_widget import Select
import json
from datetime import datetime
import os
import dspy

agent = create_math_agent()

@cl.on_chat_start
def start():
    cl.user_session.set("agent", agent)
    return cl.Message(
        content="Hello! I'm your Mathematical Professor AI. Ask me any math question and I'll provide step-by-step solutions!"
    )

# Helper to find feedback for a question
async def get_feedback_for_question(question, feedback_file="feedback_log.json"):
    if not os.path.exists(feedback_file):
        return []
    feedbacks = []
    try:
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("question", "").strip().lower() == question.strip().lower():
                    feedbacks.append(entry)
    except Exception as e:
        print(f"[Feedback read error]: {e}")
    return feedbacks

@cl.on_message
async def main(message: cl.Message):
    try:
        agent = cl.user_session.get("agent")
        # Check if awaiting clarification
        awaiting_clarification = cl.user_session.get("awaiting_clarification", False)
        import re
        if awaiting_clarification:
            # The user just sent a clarification message
            clarification = message.content
            last_question = cl.user_session.get("last_question", "")
            last_llm_answer = cl.user_session.get("last_llm_answer", "")
            # Compose a new prompt for clarification
            clarification_prompt = (
                f"The user asked: {last_question}\n"
                f"The previous answer was: {last_llm_answer}\n"
                f"The user now requests clarification: {clarification}\n"
                f"Please provide a more detailed or clarified answer."
            )
            optimized_prompt = PROMPT_OPTIMIZER.prompt if hasattr(PROMPT_OPTIMIZER, 'prompt') and PROMPT_OPTIMIZER.prompt else None
            # Use clarification_mode to skip retrieval/web search and only use LLM
            result = agent.solve(clarification_prompt, optimized_prompt=optimized_prompt, clarification_mode=True)
            cl.user_session.set("awaiting_clarification", False)
            feedback_note = ""
        else:
            user_query = message.content
            # Always reset feedback for every new response (not just new question)
            cl.user_session.set("feedback_given", False)
            # Check for past feedback
            feedbacks = await get_feedback_for_question(user_query)
            feedback_note = ""
            if feedbacks:
                negative = [f for f in feedbacks if f["feedback_type"] in ["error", "clarify"]]
                if negative:
                    feedback_note = "\n\nNote: Previous users reported issues or requested clarification for this question. Please be extra clear, detailed, and double-check your solution."
            # Use DSPy optimized prompt if available
            optimized_prompt = PROMPT_OPTIMIZER.prompt if hasattr(PROMPT_OPTIMIZER, 'prompt') and PROMPT_OPTIMIZER.prompt else None
            state = {"question": user_query, "result": None, "feedback_note": feedback_note, "optimized_prompt": optimized_prompt}
            result = agent.solve(state["question"], feedback_note=feedback_note, optimized_prompt=optimized_prompt)
        # If there was a workflow error, show it and return
        if "error" in result:
            print(f"[DEBUG] Error result: {result}")
            await cl.Message(content=f"Sorry, an error occurred: {result['error']}").send()
            return
        # If no solution was found, show a fallback message
        if not result.get("solution"):
            print(f"[DEBUG] No solution result: {result}")
            await cl.Message(content="No similar problem found.").send()
            return
        # Render LaTeX like ChatGPT: ensure equations are on their own line, not double-wrapped, and not in code blocks
        llm_answer = result.get('solution', '[No LLM answer]')
        # Remove empty/duplicate $$ blocks
        llm_answer = re.sub(r'\$\$\s*\$\$', '', llm_answer)
        llm_answer = re.sub(r'(\$\$\s*)+', '$$', llm_answer)
        # Remove code block wrappers (```)
        llm_answer = re.sub(r'```(?:latex)?', '', llm_answer)
        # Ensure block equations are on their own line
        llm_answer = re.sub(r'\$\$\s*([^$]+?)\s*\$\$', lambda m: f'\n$$\n{m.group(1).strip()}\n$$\n', llm_answer)
        # Remove any leading/trailing whitespace
        llm_answer = llm_answer.strip()
        # Ensure no extra blank lines
        llm_answer = re.sub(r'\n{3,}', '\n\n', llm_answer)
        # Extract sources from the solution or verification if present
        sources = []
        if 'Source:' in llm_answer:
            for line in llm_answer.split('\n'):
                if line.startswith('Source:'):
                    url = line.replace('Source:', '').strip()
                    if url and url != 'N/A':
                        sources.append(url)
        # Always allow feedback for every response (not just first)
        actions = [
            cl.Action(name="rate_solution", value="rate", description="Rate this solution", payload={}),
            cl.Action(name="request_clarification", value="clarify", description="Request clarification", payload={}),
            cl.Action(name="report_error", value="error", description="Report an error", payload={})
        ]
        # Add a note if an optimized prompt is in use
        prompt_note = "\n\n*Using DSPy-optimized prompt for this answer.*" if optimized_prompt else ""
        # Compose sources section
        sources_section = ""
        if sources:
            sources_section = "\n\n**Relevant Sources:**\n" + "\n".join([f"- [{url}]({url})" for url in sources])
        # Add topic, difficulty, score if available
        topic = result.get('topic', 'unknown')
        difficulty = result.get('difficulty', 'unknown')
        score = result.get('score', 0.0)
        message_content = f"**Question:** {result['question']}\n\n**Step-by-step Solution:**\n{llm_answer}{prompt_note}{sources_section}\n\n**Topic:** {topic}\n**Difficulty:** {difficulty}\n**Score:** {score:.2f}"
        print(f"[DEBUG] Outgoing message content:\n{message_content}")
        await cl.Message(
            content=message_content,
            actions=actions
        ).send()
        cl.user_session.set("last_question", result['question'])
        cl.user_session.set("last_llm_answer", llm_answer)
    except Exception as e:
        print(f"[ERROR] Exception in Chainlit handler: {e}")
        await cl.Message(content=f"Sorry, an unexpected error occurred: {e}").send()

@cl.action_callback("rate_solution")
async def on_rate_solution(action):
    cl.user_session.set("feedback_given", True)
    await log_feedback("rate", action)
    await cl.Message(content="Thank you for your feedback! (Rating received)").send()

@cl.action_callback("request_clarification")
async def on_request_clarification(action):
    cl.user_session.set("feedback_given", True)
    cl.user_session.set("awaiting_clarification", True)
    await log_feedback("clarify", action)
    await cl.Message(content="Please specify what you would like clarified about the previous answer.").send()

@cl.action_callback("report_error")
async def on_report_error(action):
    cl.user_session.set("feedback_given", True)
    await log_feedback("error", action)
    await cl.Message(content="Thank you for reporting the error. We will review this solution.").send()

# DSPy: Feedback dataset path
FEEDBACK_DATASET_PATH = "feedback_dspy.jsonl"
FEEDBACK_TRAIN_THRESHOLD = 5

# DSPy: Simple prompt optimizer (example)
class SimplePromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = None
    def forward(self, input):
        # In a real scenario, you would use DSPy APIs to optimize prompt or model
        # Here, just return input for demonstration
        return input

PROMPT_OPTIMIZER = SimplePromptOptimizer()

# Helper to count lines in feedback file
def count_feedback_entries(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# DSPy: Real training function using feedback
from dspy import Example
from dspy.datasets.dataset import Dataset
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
from dspy.predict import Predict

# DSPy: Real train function

def train_with_feedback():
    print("[DSPy] Training with feedback...")
    # Load feedback dataset
    examples = []
    try:
        with open(FEEDBACK_DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                # Use all feedback types (rate, clarify, error) for training
                if entry.get("feedback") in ["rate", "clarify", "error"]:
                    examples.append(Example(input=entry["input"], output=entry["output"]))
    except Exception as e:
        print(f"[DSPy] Error loading feedback: {e}")
        return
    if not examples:
        print("[DSPy] No suitable feedback for training.")
        return
    # Create DSPy dataset (fix: pass only examples list)
    dataset = Dataset(examples)
    # Use BootstrapFewShotWithRandomSearch for more robust prompt optimization
    teleprompter = BootstrapFewShotWithRandomSearch(metric="exact_match", max_bootstrapped_demos=3, num_candidate_programs=3)
    predictor = Predict("input -> output")
    # Train prompt
    teleprompter.compile(predictor, dataset)
    # Save the optimized prompt for use in the agent (in-memory for now)
    PROMPT_OPTIMIZER.prompt = teleprompter.prompt
    print("[DSPy] Training complete. Optimized prompt:")
    print(PROMPT_OPTIMIZER.prompt)

# Update feedback logging to DSPy-compatible format and trigger training
async def log_feedback(feedback_type, action):
    feedback = {
        "timestamp": datetime.utcnow().isoformat(),
        "feedback_type": feedback_type,
        "question": cl.user_session.get("last_question", ""),
        "llm_answer": cl.user_session.get("last_llm_answer", ""),
        "action_payload": action.payload if hasattr(action, 'payload') else {},
        "user_id": getattr(action, 'user_id', None)
    }
    # Write to both the old log and DSPy dataset
    try:
        with open("feedback_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback) + "\n")
        # Write DSPy-compatible feedback
        dspy_feedback = {
            "input": feedback["question"],
            "output": feedback["llm_answer"],
            "feedback": feedback["feedback_type"]
        }
        with open(FEEDBACK_DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(dspy_feedback) + "\n")
        # Check if we should trigger training
        if count_feedback_entries(FEEDBACK_DATASET_PATH) % FEEDBACK_TRAIN_THRESHOLD == 0:
            train_with_feedback()
    except Exception as e:
        print(f"[Feedback logging error]: {e}")
