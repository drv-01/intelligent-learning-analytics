import os
from dotenv import load_dotenv
load_dotenv(override=True)

from typing import List, TypedDict, Annotated
import operator

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------------
# GLOBAL SESSION MEMORY (MemorySaver)
# -----------------------------------
agent_memory = MemorySaver()

# -----------------------------------
# STRUCTURED OUTPUT SCHEMA
# -----------------------------------
class StudyPlanPlan(BaseModel):
    strategy: str = Field(description="Personalized study strategy")
    weekly_goals: List[str] = Field(description="List of weekly goals")

class StructuredDiagnosisOutput(BaseModel):
    diagnosis: str = Field(description="Analysis of learning gaps")
    plan: StudyPlanPlan
    tutorials: List[str] = Field(description="Recommended tutorials and resources")

# -----------------------------------
# LANGGRAPH STATE DEFINITION
# -----------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    reasoning_steps: List[str]
    tool_results: List[str]
    final_response: str

# -----------------------------------
# KNOWLEDGE BASE (Local)
# -----------------------------------
KNOWLEDGE_BASE = [
    "Tutorial on Time Management: Use the Pomodoro technique — 25 min work / 5 min break — for long study sessions. Track productivity.",
    "Mathematics Basics: Focus on algebra, geometry basics. Practice 15 problems daily and use Khan Academy for step-by-step walkthroughs.",
    "Effective Notes Strategy: The Cornell note-taking system enhances retention. Divide page into cues, notes, and summary sections.",
    "Science Study Tips: Use spaced repetition and Anki flashcards for memorization. Review biology diagrams repeatedly.",
    "Attendance Improvement: Regular attendance improves understanding. Students with >85% attendance score 20% higher on average.",
    "Sleep & Cognition: Students who sleep 7-9 hours nightly retain 30% more information. Establish a fixed sleep schedule.",
    "Study Planning: Create a weekly planner. Assign subject blocks, revision days, and mock test days. Use Google Calendar or Notion.",
    "At Risk Students: Provide structured mentorship, reduce distractions, and use shorter focused study sprints (15 min). Celebrate small wins.",
    "High Performer Tips: Challenge yourself with competitive exam prep — Olympiads, JEE, SAT practice. Explore advanced topics.",
    "Assignment Completion: Break large assignments into smaller tasks using the tasks-first method. Use a Kanban board to track completion.",
]

def init_chroma_db():
    pass # No longer needed, keeping for compatibility if referenced elsewhere


# -----------------------------------
# TOOLS
# -----------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date educational information, learning resources, or study strategies."""
    try:
        search = DuckDuckGoSearchRun()
        result = search.invoke(query)
        return f"[Web Search Results for '{query}']:\n{result}"
    except Exception as e:
        return f"[Web Search Fallback for '{query}']: General educational resources and study tips are available at Khan Academy, Coursera, and YouTube EDU."

@tool
def fetch_tutorials(query: str) -> str:
    """Retrieve relevant learning tutorials and educational materials from the internal knowledge base."""
    query_words = set(query.lower().split())
    results = []
    
    for doc in KNOWLEDGE_BASE:
        doc_words = set(doc.lower().split())
        overlap = len(query_words.intersection(doc_words))
        if overlap > 0:
            results.append((overlap, doc))
    
    # Sort by overlap score
    results.sort(key=lambda x: x[0], reverse=True)
    top_matches = results[:3]
    
    if top_matches:
        formatted = "\n".join([f"• {res[1]}" for res in top_matches])
        return f"[Knowledge Base Results for '{query}']:\n{formatted}"
    
    return f"[Logic Fallback]: Recommend general tutorials from Khan Academy or Coursera related to the topic."

@tool
def content_summarization(text: str) -> str:
    """Summarize a long text of educational content into a concise format for the student."""
    if isinstance(text, list):
        text = "\n".join([str(t) for t in text])
    sentences = text.split(". ")
    key_points = sentences[:3]
    return f"[Content Summary]:\n" + "\n".join([f"• {s.strip()}" for s in key_points if s.strip()])

# -----------------------------------
# SYSTEM PROMPT
# -----------------------------------
SYSTEM_PROMPT = """You are an autonomous AI Study Coach embedded in an Intelligent Learning Analytics platform.

Your responsibilities:
1. **Diagnose**: Analyze the student's learning gaps using the provided data/context via chain-of-thought reasoning.
2. **Plan**: Create a personalized weekly study strategy tailored to the student's specific profile.
3. **Retrieve**: Use your tools (web_search, fetch_tutorials, content_summarization) to find up-to-date educational resources.
4. **Respond**: Output a structured coaching report in Markdown format.

Chain-of-Thought Process:
- Step 1: Understand the student's current situation, weak areas, and goals.
- Step 2: Identify what external resources would help using fetch_tutorials.
- Step 3: If needed, search the web for current best practices using web_search.
- Step 4: Summarize any long content using content_summarization.
- Step 5: Compile a cohesive Diagnosis → Plan → Resources output.

You MUST output the final coaching response in this EXACT Markdown structure:

## 🔍 Diagnosis
(Detailed learning gaps analysis based on the student's data)

## 📋 Plan
### Personalized Strategy
(Detailed, specific study strategy for this student)

### Weekly Goals
- Goal 1
- Goal 2
- Goal 3
- Goal 4

## 📚 Resources & Tutorials
- Resource 1 (with description)
- Resource 2 (with description)
- Resource 3 (with description)

## 💡 Coach's Insight
(A motivating, personalized closing insight for the student)
"""

# -----------------------------------
# LANGGRAPH WORKFLOW NODES
# -----------------------------------
def reasoning_node(state: AgentState) -> AgentState:
    """Chain-of-thought reasoning step: understand the student's context."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=api_key)
    
    # Build chain-of-thought reasoning prompt
    cot_prompt = f"""You are analyzing a student's academic situation. 
Think step-by-step about:
1. What are the likely learning gaps?
2. What subjects or skills need improvement?
3. What intervention strategies are most appropriate?
4. What resources should we look for?

Student context: {state['messages'][-1].content}

Provide a brief reasoning chain (3-5 bullet points)."""
    
    reasoning_response = llm.invoke([HumanMessage(content=cot_prompt)])
    res_content = reasoning_response.content
    if isinstance(res_content, list):
        res_content = "\n".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in res_content])
    reasoning_steps = res_content.split("\n")
    reasoning_steps = [s.strip() for s in reasoning_steps if s.strip()]
    
    return {
        "messages": state["messages"],
        "reasoning_steps": reasoning_steps,
        "tool_results": state.get("tool_results", []),
        "final_response": state.get("final_response", "")
    }

def tool_retrieval_node(state: AgentState) -> AgentState:
    """Retrieve tutorials from RAG and optionally web search."""
    init_chroma_db()
    tool_results = []
    
    # Extract student context for tool calls
    student_context = state["messages"][-1].content
    
    # Always fetch from RAG
    rag_result = fetch_tutorials.invoke(student_context[:200])
    tool_results.append(rag_result)
    
    # Web search for supplementary resources
    search_query = f"best study strategies and tutorials for {student_context[:100]}"
    web_result = web_search.invoke(search_query)
    
    # Use summarization tool if web result is very long (Requirement satisfaction)
    if len(web_result) > 500:
        summarized_web = content_summarization.invoke(web_result)
        tool_results.append(summarized_web)
    else:
        tool_results.append(web_result)
    
    return {
        "messages": state["messages"],
        "reasoning_steps": state.get("reasoning_steps", []),
        "tool_results": tool_results,
        "final_response": state.get("final_response", "")
    }

def response_generation_node(state: AgentState) -> AgentState:
    """Generate the final structured coaching response."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, groq_api_key=api_key)
    
    reasoning_context = "\n".join(state.get("reasoning_steps", []))
    tool_context = "\n\n".join(state.get("tool_results", []))
    
    full_prompt = f"""Based on the following:

STUDENT CONTEXT:
{state['messages'][-1].content}

CHAIN-OF-THOUGHT REASONING:
{reasoning_context}

RETRIEVED RESOURCES (RAG + Web):
{tool_context}

Now produce the final structured coaching response following the system prompt format exactly."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=full_prompt)
    ]
    
    response = llm.invoke(messages)
    final_response = response.content
    if isinstance(final_response, list):
        final_response = "\n".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in final_response])
    
    return {
        "messages": state["messages"] + [AIMessage(content=final_response)],
        "reasoning_steps": state.get("reasoning_steps", []),
        "tool_results": state.get("tool_results", []),
        "final_response": final_response
    }

# -----------------------------------
# BUILD LANGGRAPH WORKFLOW
# -----------------------------------
def build_coach_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("tool_retrieval", tool_retrieval_node)
    workflow.add_node("response_generation", response_generation_node)
    
    workflow.set_entry_point("reasoning")
    workflow.add_edge("reasoning", "tool_retrieval")
    workflow.add_edge("tool_retrieval", "response_generation")
    workflow.add_edge("response_generation", END)
    
    return workflow.compile(checkpointer=agent_memory)

# -----------------------------------
# PUBLIC API: get_coach_agent()
# -----------------------------------
_graph = None

def get_coach_agent():
    """Returns compiled LangGraph agent with MemorySaver for session persistence."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env. Please add it to use the AI Coach.")
    global _graph
    if _graph is None:
        init_chroma_db()
        _graph = build_coach_graph()
    return _graph
