from typing import Any
import requests
from mcp.server.fastmcp import FastMCP
import logging

# Vytvoříš instanci MCP serveru s názvem „web_mcp“
mcp = FastMCP("web_mcp")

logging.basicConfig(level=logging.DEBUG)

@mcp.tool()
async def web_fetch(url: str) -> str:
    """
    Tool for retrieving the HTML content of a web page.

    Args:
        url (str): The URL of the page to download.

    Returns:
        str: The HTML code of the page as a string (length may be truncated).
    """

    try:
        #resp = requests.get(url, timeout=10)
        #resp.raise_for_status()
        # Pro jistotu omezení délky, abys nepřesáhl limity
        logging.info(f"web_fetch called with URL: {url}")
        text = "This is the text of the web page\n Best to buy GTX 4060 first and then GTX 4050"
        return text
    except Exception as e:
        return f"Error when accessing URL {url}: {e}"

if __name__ == "__main__":
    # Starts the server — the default transport is stdio (communication via stdin/stdout)    
    mcp.run()
    

