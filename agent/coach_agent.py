import os
from dotenv import load_dotenv
load_dotenv(override=True)


from pydantic import BaseModel, Field
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma

# Global memory to act as in-memory session persistence
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
# REAL CHROMA RAG SETUP
# -----------------------------------
vector_store = None

def init_chroma_db():
    global vector_store
    if vector_store is not None:
        return
    
    # Check if API key is present for embeddings
    if not os.environ.get("GOOGLE_API_KEY"):
        return
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    docs = [
        Document(page_content="Tutorial on Time Management: Use the Pomodoro technique for long study sessions.", metadata={"topic": "time management"}),
        Document(page_content="Math Basics Tutorial: Focus on practicing algebra problems daily.", metadata={"topic": "mathematics"}),
        Document(page_content="Effective Notes: The Cornell note-taking system enhances retention.", metadata={"topic": "note taking"}),
        Document(page_content="Science Study Tips: Use spaced repetition flashcards for memorization.", metadata={"topic": "science"})
    ]
    vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="local_tutorials")

# -----------------------------------
# TOOLS
# -----------------------------------
@tool
def fetch_tutorials(query: str) -> str:
    """Fetch relevant learning tutorials and educational materials using Chroma DB RAG based on the query."""
    init_chroma_db()
    if vector_store:
        results = vector_store.similarity_search(query, k=2)
        if results:
            return "\\n".join([f"RAG TUTORIAL CONTENT: {res.page_content}" for res in results])
    return f"No specific RAG content found for topic '{query}'. Suggest general tutorials instead."

@tool
def content_summarization(text: str) -> str:
    """Summarize a large piece of learning content for the student."""
    return f"[Summarized Content]: Extracting key insights... \\n{text[:150]}..."

@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date educational information or tools."""
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception:
        return f"Web search results regarding {query}: General educational tips available online."

# -----------------------------------
# AGENT SETUP
# -----------------------------------
SYSTEM_PROMPT = """You are an AI Study Coach for Intelligent Learning Analytics.
Your task is to analyze the user's/student's performance, understand their goals, and provide structured, multi-step reasoned advice.

You must follow chain-of-thought prompting:
1. Understand the student's gaps based on Data/Analytics.
2. Determine required external resources using tools (Web search, Content summarization, Tutorial retrieval).
3. Draft a personalized plan.

Finally, output the coaching response EXACTLY adhering to the requested structured format. Do not deviate. Provide the output in Markdown representing the underlying JSON-like structured data:

## Diagnosis
(Your Learning gaps analysis)

## Plan
### Personalized strategy
(Your detailed personalized study strategy)

### Weekly goals
- (Goal 1)
- (Goal 2)

## Resources & Tutorials
- (Tutorial or resource 1 fetched from tools)
- (Tutorial or resource 2 fetched from tools)
"""

def get_coach_agent():
    # Load API key explicitly so we can safely check it
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the .env file. Please add it to use the AI Coach.")
        
    # Initialize Chroma lazily after key is set
    init_chroma_db()
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    tools = [web_search, fetch_tutorials, content_summarization]
    
    agent_executor = create_react_agent(
        llm, 
        tools, 
        checkpointer=agent_memory,
    )
    return agent_executor
