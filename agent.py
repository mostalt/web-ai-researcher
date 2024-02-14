import asyncio
import base64
import re

from langchain_core.runnables import chain as chain_decorator, RunnablePassthrough
from langchain import hub
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from state import AgentState

# Some javascript we will run on each stepto take a screenshot of the page,
# select the elements to annotate, and add bounding boxes
with open("annotate.js") as f:
  annotate_page_script = f.read()


@chain_decorator
async def mark_page(page):
  await page.evaluate(annotate_page_script)
  for _ in range(10):
    try:
      bboxes = await page.evaluate("markPage()")
      break
    except:
      # May be loading...
      await asyncio.sleep(3)
  await page.screenshot(path="screenshot.png")
  screenshot = await page.screenshot()

  # Ensure the bboxes don't follow us around
  await page.evaluate("unmarkPage()")
  return {
    "img": base64.b64encode(screenshot).decode(),
    "bboxes": bboxes,
  }

async def annotate(state):
  marked_page = await mark_page.with_retry().ainvoke(state["page"])
  return {**state, **marked_page}

def format_descriptions(state):
  labels = []
  for i, bbox in enumerate(state["bboxes"]):
    text = bbox.get("ariaLabel") or ""
    if not text.strip():
      text = bbox["text"]
    el_type = bbox.get("type")
    labels.append(f'{i} (<{el_type}/>): "{text}"')
  bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
  return {**state, "bbox_descriptions": bbox_descriptions}

def parse(text: str) -> dict:
  action_prefix = "Action: "
  if not text.strip().split("\n")[-1].startswith(action_prefix):
    return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
  action_block = text.strip().split("\n")[-1]

  action_str = action_block[len(action_prefix) :]
  split_output = action_str.split(" ", 1)
  if len(split_output) == 1:
    action, action_input = split_output[0], None
  else:
    action, action_input = split_output
  action = action.strip()
  if action_input is not None:
    action_input = [
      inp.strip().strip("[]") for inp in action_input.strip().split(";")
    ]
  return {"action": action, "args": action_input}

def update_scratchpad(state: AgentState):
  """After a tool is invoked, we want to update
  the scratchpad so the agent is aware of its previous steps"""

  old = state.get("scratchpad")
  if old:
    txt = old[0].content
    last_line = txt.rsplit("\n", 1)[-1]
    step = int(re.match(r"\d+", last_line).group()) + 1
  else:
    txt = "Previous action observations:\n"
    step = 1
  txt += f"\n{step}. {state['observation']}"

  return {**state, "scratchpad": [SystemMessage(content=txt)]}

def run_agent():
  # Will need a later version of langchain to pull this image prompt template
  prompt = hub.pull("wfh/web-voyager")

  llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=4096)
  agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
  )

  return agent

async def call_agent(question: str, page, graph, max_steps: int = 150):
  event_stream = graph.astream(
    {
      "page": page,
      "input": question,
      "scratchpad": [],
    },
    {
      "recursion_limit": max_steps,
    },
  )
  final_answer = None
  steps = []

  async for event in event_stream:
    # We'll display an event stream here
    if "agent" not in event:
      continue
    pred = event["agent"].get("prediction") or {}
    action = pred.get("action")
    action_input = pred.get("args")
    action_log = f"{len(steps) + 1}. {action}: {action_input}"
    steps.append(action_log)
    print(action_log + "\n")

    if "ANSWER" in action:
      final_answer = action_input[0]
      break
  return final_answer