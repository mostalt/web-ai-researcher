from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from state import AgentState
from agent import update_scratchpad
from tools import tools

def select_tool(state: AgentState):
  # Any time the agent completes, this function is called to route the output to a tool or to the end user.
  action = state["prediction"]["action"]
  if action == "ANSWER":
    return END
  if action == "retry":
    return "agent"
  return action


def build_graph(agent):
  graph_builder = StateGraph(AgentState)

  graph_builder.add_node("agent", agent)
  graph_builder.set_entry_point("agent")
  graph_builder.add_node("update_scratchpad", update_scratchpad)
  graph_builder.add_edge("update_scratchpad", "agent")

  for node_name, tool in tools.items():
    graph_builder.add_node(
      node_name,
      # The lambda ensures the function's string output is mapped to the "observation"
      # key in the AgentState
      RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    # Always return to the agent (by means of the update-scratchpad node)
    graph_builder.add_edge(node_name, "update_scratchpad")

  graph_builder.add_conditional_edges("agent", select_tool)

  return graph_builder