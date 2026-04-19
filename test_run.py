import os
from agent.coach_agent import get_coach_agent
from langchain_core.messages import HumanMessage

def run_test():
    try:
        print("Initializing agent...")
        agent = get_coach_agent()
        print("Agent initialized successfully.")
        
        # Checking if API key is actually set
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("WARNING: GOOGLE_API_KEY is not set in environment or .env file. The agent will likely fail during invocation.")
            return

        print("Testing agent invocation...")
        config = {"configurable": {"thread_id": "test_session"}}
        response = agent.invoke({"messages": [HumanMessage(content="Hello!")]}, config)
        print("Agent responded:")
        print(response["messages"][-1].content)
        print("SUCCESS!")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    run_test()
