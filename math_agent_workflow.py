# math_agent_workflow.py
# Agentic workflow for math problem solving with knowledge base search

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import openai
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


# Base Agent State Classes
class AgentState:
    def __post_init__(self):
        if self.trace is None:
            self.trace = []
    def add_trace(self, message: str):
        if self.trace is None:
            self.trace = []
        self.trace.append(message)

from dataclasses import dataclass
@dataclass
class MathProblemState(AgentState):
    question: str
    retrieved_info: Optional[Dict] = None
    web_search_results: Optional[Dict] = None
    solution: Optional[str] = None
    verification_result: Optional[Dict] = None
    revised_solution: Optional[str] = None
    final_verification: Optional[Dict] = None
    feedback_note: Optional[str] = None
    optimized_prompt: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    trace: List[str] = None

# Base Agent Class
class Agent(ABC):
    """Abstract base class for all agents in the workflow"""
    def __init__(self):
        self.load_environment()
    
    def load_environment(self):
        load_dotenv()
    
    @abstractmethod
    def run(self, state: MathProblemState) -> MathProblemState:
        """Execute the agent's primary task"""
        pass

# Concrete Agent Implementations
class RetrievalAgent(Agent):
    """Agent responsible for retrieving similar problems from knowledge base"""
    def __init__(self):
        super().__init__()
        # Try to use GPU for SentenceTransformer if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        if device == "cuda":
            print("[INFO] Using GPU (CUDA) for SentenceTransformer embedding.")
        else:
            print("[INFO] Using CPU for SentenceTransformer embedding.")
        self.client = QdrantClient("localhost", port=6333)
        # Use the correct collection name as in vector_db_setup.py
        self.collection_name = "gsm8k_questions"

    def run(self, state: MathProblemState) -> MathProblemState:
        try:
            query_embedding = self.model.encode([state.question])[0]
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=1
            )
            if results:
                r = results[0]
                # Defensive: handle missing 'answer' in payload
                answer = r.payload.get("answer", None)
                if answer is None:
                    state.add_trace("No answer found in vector DB payload.")
                state.retrieved_info = {
                    "question": r.payload.get("question", ""),
                    "answer": answer,
                    "topic": r.payload.get("topic", "unknown"),
                    "difficulty": r.payload.get("difficulty", "unknown"),
                    "score": r.score
                }
                print("[INFO] Response fetched from vector database.")
                state.add_trace(f"Retrieved similar problem with score {r.score:.2f}")
            else:
                state.add_trace("No similar problems found in knowledge base")
            
        except Exception as e:
            state.success = False
            state.error = f"Retrieval error: {str(e)}"
            state.add_trace(f"Retrieval failed: {str(e)}")
        return state

class WebSearchAgent(Agent):
    """Agent responsible for web search fallback"""
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_client = TavilyClient(self.api_key)
        self.domains = [
            "khanacademy.org", "mathstackexchange.com", "mathoverflow.net", "wolfram.com",
            "brilliant.org", "artofproblemsolving.com", "mathisfun.com", "purplemath.com",
            "symbolab.com", "desmos.com", "byjus.com", "cuemath.com", "mathhelp.com",
            "mathway.com", "chegg.com", "socratic.org", "edx.org", "coursera.org",
            "openstax.org", "mit.edu", "stanford.edu", "harvard.edu"
        ]

    def is_relevant(self, result: Dict, question: str) -> bool:
        content = result.get('content', '').lower()
        question = question.lower()
        qwords = [w for w in question.split() if len(w) > 2]
        match_count = sum(1 for w in qwords if w in content)
        return match_count >= max(1, len(qwords) // 2)

    def run(self, state: MathProblemState) -> MathProblemState:
        try:
            response = self.tavily_client.search(
                query=f"mathematical solution step by step {state.question}",
                search_depth="advanced",
                max_results=5,
                include_domains=self.domains
            )
            if response and response.get('results'):
                filtered = [r for r in response['results'] if self.is_relevant(r, state.question)]
                results = filtered[:1] if filtered else response['results'][:1]
                state.web_search_results = {
                    'content': "\n\n".join([
                        f"Source: {r.get('url', 'N/A')}\nContent: {r.get('content', 'No answer found.').strip()}"
                        for r in results
                    ])
                }
                print("[INFO] Response fetched from Tavily web search.")
                state.add_trace(f"Found {len(results)} relevant web results")
            else:
                state.add_trace("No relevant web results found")
        except Exception as e:
            state.success = False
            state.error = f"Web search error: {str(e)}"
            state.add_trace(f"Web search failed: {str(e)}")
        return state

class SolutionAgent(Agent):
    """Agent responsible for generating math solutions using LLM"""
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

    def get_prompt(self, state: MathProblemState) -> str:
        # Remove all LaTeX/unicode formatting instructions, just ask for equations on separate lines
        if state.optimized_prompt:
            return state.optimized_prompt.format(
                question=state.question,
                retrieved_info=state.retrieved_info.get('answer') if state.retrieved_info else state.web_search_results.get('content', ''),
                feedback_note=state.feedback_note or ''
            )
        return f"""
You are a helpful math tutor. Given the following math question and retrieved information, provide a clear, step-by-step solution.

For each step, use a short explanation (if needed), then put the equation on its own line. Do not use any LaTeX or unicode formatting instructions. Just write the equations on their own line, and explanations on separate lines.

Be concise and only show the essential steps and final answer. If the solution requires more steps, do not omit any important step.

Question: {state.question}

Retrieved Information:
{state.retrieved_info.get('answer') if state.retrieved_info else state.web_search_results.get('content', '')}

Step-by-step solution:
"""

    def run(self, state: MathProblemState) -> MathProblemState:
        try:
            if not self.api_key:
                raise ValueError("OpenAI API key not set")
                
            prompt = self.get_prompt(state)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.2,
                top_p=0.95
            )
            
            state.solution = response.choices[0].message.content.strip()
            state.add_trace("Generated solution using LLM")
            
        except Exception as e:
            state.success = False
            state.error = f"Solution generation error: {str(e)}"
            state.add_trace(f"Solution generation failed: {str(e)}")
        
        return state

class VerificationAgent(Agent):
    """Agent responsible for verifying solutions and suggesting revisions"""
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

    def verify_solution(self, question: str, solution: str) -> Dict:
        # Independent solution
        prompt_independent = (
            f"You are a math professor. Solve the following question step by step, as if you have not seen any previous answer. Format all math in LaTeX.\n\n"
            f"Question: {question}\n\n"
            f"Step-by-step solution:"
        )
        try:
            response1 = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_independent}],
                max_tokens=512,
                temperature=0.2
            )
            independent_solution = response1.choices[0].message.content.strip()
        except Exception as e:
            independent_solution = f"[Error generating independent solution: {str(e)}]"

        # Backtrack check
        prompt_backtrack = (
            f"You are a math professor. Given the following answer, try to reconstruct the original question or check if the answer is a valid solution to the question. Explain your reasoning step by step.\n\n"
            f"Question: {question}\n\n"
            f"Provided Answer: {solution}\n\n"
            f"Backtrack/Validation:"
        )
        try:
            response2 = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_backtrack}],
                max_tokens=512,
                temperature=0.2
            )
            backtrack_check = response2.choices[0].message.content.strip()
        except Exception as e:
            backtrack_check = f"[Error during backtrack check: {str(e)}]"

        # Verdict
        verdict = "Verified" if ("correct" in backtrack_check.lower() or "valid" in backtrack_check.lower()) else "Needs Review"

        return {
            "independent_solution": independent_solution,
            "backtrack_check": backtrack_check,
            "verdict": verdict
        }

    def revise_solution(self, question: str, original_solution: str, verification_feedback: str) -> str:
        prompt = f"""
You are a math professor. The following solution was flagged as needing review. Here is the original question and the previous answer. Please revise the answer to address any issues and provide a clear, correct, step-by-step solution in LaTeX.

Question: {question}

Previous Answer: {original_solution}

Verification Feedback: {verification_feedback}

Revised Step-by-step Solution:
"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def run(self, state: MathProblemState) -> MathProblemState:
        try:
            # Initial verification
            state.verification_result = self.verify_solution(state.question, state.solution)
            state.add_trace(f"Initial verification verdict: {state.verification_result['verdict']}")
            
            # If verification fails, attempt revision
            if state.verification_result['verdict'] != "Verified":
                state.revised_solution = self.revise_solution(
                    state.question,
                    state.solution,
                    state.verification_result['backtrack_check']
                )
                state.add_trace("Generated revised solution")
                
                # Verify the revised solution
                state.final_verification = self.verify_solution(state.question, state.revised_solution)
                state.add_trace(f"Final verification verdict: {state.final_verification['verdict']}")
                
        except Exception as e:
            state.success = False
            state.error = f"Verification error: {str(e)}"
            state.add_trace(f"Verification failed: {str(e)}")
        
        return state

class MathAgentWorkflow:
    """Main workflow coordinator"""
    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.web_search_agent = WebSearchAgent()
        self.solution_agent = SolutionAgent()
        self.verification_agent = VerificationAgent()

    def solve(self, question: str, feedback_note: Optional[str] = None, optimized_prompt: Optional[str] = None, clarification_mode: bool = False) -> Dict:
        # Initialize state
        state = MathProblemState(
            question=question,
            feedback_note=feedback_note,
            optimized_prompt=optimized_prompt
        )
        try:
            if clarification_mode:
                # Only use the LLM to generate a clarified answer, skip retrieval and web search
                state = self.solution_agent.run(state)
                result = {
                    "question": question,
                    "solution": state.solution,
                    "trace": state.trace
                }
                if not state.success:
                    result["error"] = state.error
                print(f"[DEBUG] Clarification mode result: {result}")
                return result
            # Normal workflow
            # Step 1: Knowledge base retrieval
            state = self.retrieval_agent.run(state)
            print(f"[DEBUG] After retrieval: {state.retrieved_info}")
            # Step 2: Web search only if no result or low similarity
            skip_web_search = False
            if state.retrieved_info:
                # If the retrieved question matches the user question (case-insensitive, stripped), or score >= 0.9, skip web search
                user_q = state.question.strip().lower()
                retrieved_q = state.retrieved_info.get("question", "").strip().lower()
                score = state.retrieved_info.get("score", 0)
                if user_q == retrieved_q or score >= 0.9:
                    skip_web_search = True
            print(f"[DEBUG] skip_web_search: {skip_web_search}")
            if not state.retrieved_info or not skip_web_search:
                state = self.web_search_agent.run(state)
                print(f"[DEBUG] After web search: {state.web_search_results}")
            # Step 3: Generate solution
            state = self.solution_agent.run(state)
            print(f"[DEBUG] After solution: {state.solution}")
            # Step 4: Verify and possibly revise
            state = self.verification_agent.run(state)
            print(f"[DEBUG] After verification: {state.verification_result}")
            # Prepare result
            result = {
                "question": question,
                "solution": state.solution,
                "verification": state.verification_result,
                "trace": state.trace
            }
            if state.revised_solution:
                result.update({
                    "revised_solution": state.revised_solution,
                    "final_verification": state.final_verification
                })
            if not state.success:
                result["error"] = state.error
            print(f"[DEBUG] Final result: {result}")
            return result
        except Exception as e:
            print(f"[ERROR] Workflow exception: {str(e)}")
            return {
                "error": f"Workflow error: {str(e)}",
                "trace": state.trace
            }


# Factory function for Chainlit and other consumers
def create_math_agent():
    return MathAgentWorkflow()

# For direct script execution
if __name__ == "__main__":
    workflow = MathAgentWorkflow()
    user_query = input("Enter your math question: ")
    result = workflow.solve(user_query)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        print("\nExecution trace:")
        for step in result.get('trace', []):
            print(f"- {step}")
    else:

        import re
        def extract_latex_blocks(text):
            """Extracts all LaTeX blocks wrapped in $$...$$ from the text."""
            return re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL)

        def display_latex_and_plain(text, label=None):
            latex_blocks = extract_latex_blocks(text)
            if label:
                print(f"\n{label} (LaTeX and Plaintext):")
            if latex_blocks:
                for i, block in enumerate(latex_blocks, 1):
                    # Print LaTeX block
                    print(f"[LaTeX {i}]: $$ {block.strip()} $$")
                    # Print plain version (remove LaTeX commands for readability)
                    plain = re.sub(r'\\[a-zA-Z]+|\$|\{|\}', '', block)
                    print(f"[Plain {i}]: {plain.strip()}")
            else:
                print(text)

        print("\nSolution:")
        display_latex_and_plain(result['solution'], label="Solution")
        print("\nVerification:")
        print(f"Verdict: {result['verification']['verdict']}")
        display_latex_and_plain(result['verification']['backtrack_check'], label="Backtrack Check")

        if 'revised_solution' in result:
            print("\nRevised Solution:")
            display_latex_and_plain(result['revised_solution'], label="Revised Solution")
            print(f"\nFinal Verification: {result['final_verification']['verdict']}")
            display_latex_and_plain(result['final_verification']['backtrack_check'], label="Final Backtrack Check")

        print("\nExecution trace:")
        for step in result['trace']:
            print(f"- {step}")
    
    # Print the source of the answer
    if result.get('solution'):
        if result.get('retrieved_info') and result['retrieved_info'].get('answer'):
            print("[RESULT] Answer retrieved from vector database.")
        elif result.get('web_search_results') and result['web_search_results'].get('content'):
            print("[RESULT] Answer retrieved from web search.")
        else:
            print("[RESULT] Answer generated directly by LLM.")
