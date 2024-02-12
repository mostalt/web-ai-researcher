from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import Page


class BBox(TypedDict):
  x: float
  y: float
  text: str
  type: str
  ariaLabel: str


class Prediction(TypedDict):
  action: str
  args: Optional[List[str]]


# This represents the state of the agent as it proceeds through execution
class AgentState(TypedDict):
  page: Page  # The Playwright web page lets us interact with the web environment
  input: str  # User request
  img: str  # b64 encoded screenshot
  bboxes: List[BBox]  # The bounding boxes from the browser annotation function
  prediction: Prediction  # The Agent's output
  scratchpad: List[BaseMessage] # A system message (or messages) containing the intermediate steps
  observation: str  # The most recent response from a tool
