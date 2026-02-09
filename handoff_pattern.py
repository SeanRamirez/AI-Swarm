def create_handoff_swarm():
    """Create a handoff-based multi-agent system."""
    
    triage = TriageAgent()
    technical = TechnicalAgent()
    creative = CreativeAgent()
    data = DataAgent()
    
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("triage", triage.process)
    builder.add_node("technical", technical.process)
    builder.add_node("creative", creative.process)
    builder.add_node("data", data.process)
    
    # Router based on next_agent state
    def route_next(state: AgentState) -> Literal["technical", "creative", "data", "__end__"]:
        if state.get("task_complete", False):
            return END
        
        next_agent = state.get("next_agent", "finish")
        if next_agent in ["technical", "creative", "data"]:
            return next_agent
        return END
    
    # Set up edges
    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", route_next)
    builder.add_conditional_edges("technical", route_next)
    builder.add_conditional_edges("creative", route_next)
    builder.add_conditional_edges("data", route_next)
    
    return builder.compile()