"""
Supervisor swarm pattern - one agent coordinates multiple specialist agents.
This is one of the most common multi-agent patterns.
"""

from dotenv import load_dotenv
from typing import Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from typing_extensions import TypedDict

load_dotenv()


# Define specialized agents
class ResearchAgent:
    """Agent specialized in researching information."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.system_prompt = """You are a research specialist. 
        Your job is to gather and synthesize information on topics.
        Be thorough and cite sources when possible."""
    
    def process(self, messages):
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            *messages
        ])
        return {"messages": [response]}


class WriterAgent:
    """Agent specialized in writing content."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.system_prompt = """You are a professional writer.
        Take research and create clear, engaging content.
        Focus on clarity and readability."""
    
    def process(self, messages):
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            *messages
        ])
        return {"messages": [response]}


class ReviewerAgent:
    """Agent specialized in reviewing and critiquing work."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.system_prompt = """You are a critical reviewer.
        Review content for accuracy, clarity, and completeness.
        Provide constructive feedback."""
    
    def process(self, messages):
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            *messages
        ])
        return {"messages": [response]}


class SupervisorAgent:
    """Supervisor that routes tasks to appropriate specialists."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.system_prompt = """You are a supervisor managing a team of specialists:
        - researcher: Gathers information and does research
        - writer: Creates written content
        - reviewer: Reviews and provides feedback
        
        Based on the current state of work, decide which specialist should work next.
        When the task is complete, respond with 'FINISH'.
        
        Respond with ONLY the name of the next agent (researcher/writer/reviewer) or FINISH."""
    
    def route(self, messages) -> Literal["researcher", "writer", "reviewer", "finish"]:
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            *messages
        ])
        
        next_agent = response.content.strip().lower()
        
        if "finish" in next_agent:
            return "finish"
        elif "researcher" in next_agent:
            return "researcher"
        elif "writer" in next_agent:
            return "writer"
        elif "reviewer" in next_agent:
            return "reviewer"
        else:
            return "finish"


def create_supervisor_swarm():
    """Create a supervisor-based multi-agent system."""
    
    # Initialize agents
    supervisor = SupervisorAgent()
    researcher = ResearchAgent()
    writer = WriterAgent()
    reviewer = ReviewerAgent()
    
    # Define the graph
    builder = StateGraph(MessagesState)
    
    # Add agent nodes
    builder.add_node("researcher", researcher.process)
    builder.add_node("writer", writer.process)
    builder.add_node("reviewer", reviewer.process)
    builder.add_node("supervisor", lambda state: {
        "messages": state["messages"]
    })
    
    # Add edges - supervisor routes to all agents
    builder.add_edge(START, "supervisor")
    
    def supervisor_router(state: MessagesState):
        next_agent = supervisor.route(state["messages"])
        if next_agent == "finish":
            return END
        return next_agent
    
    builder.add_conditional_edges("supervisor", supervisor_router)
    
    # All agents return to supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("reviewer", "supervisor")
    
    return builder.compile()


def main():
    graph = create_supervisor_swarm()
    
    task = """Create a brief article (3-4 sentences) about the benefits of meditation.
    Make sure it's well-researched, well-written, and reviewed for quality."""
    
    print(f"Task: {task}\n")
    print("="*60)
    
    result = graph.invoke({
        "messages": [HumanMessage(content=task)]
    })
    
    print("\n=== Conversation Flow ===")
    for msg in result["messages"]:
        role = msg.__class__.__name__
        print(f"\n{role}:")
        print(msg.content)
    
    print("\n" + "="*60)
    print("\n=== Final Output ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
