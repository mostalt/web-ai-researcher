import sys
import asyncio

from dotenv import load_dotenv
from playwright.async_api import async_playwright

from agent import call_agent, run_agent
from graph import build_graph

load_dotenv()

async def main():
  async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True, args=None)

    try:
      page = await browser.new_page()
      page.set_default_timeout(1000*60*5) #5min

      await page.goto("https://www.google.com")

      agent = run_agent()
      graph_builder = build_graph(agent)
      graph = graph_builder.compile()

      res = await call_agent("Could you explain the WebVoyager paper (on arxiv)?", page, graph)
      print(f"Final response: {res}")
    finally:
      await browser.close()
      exit_program()

def exit_program():
    print("Exiting the program...")
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())