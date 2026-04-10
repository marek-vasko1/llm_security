from mcp.server.fastmcp import FastMCP
import logging
from urllib.parse import urlparse

from pathlib import Path
from bs4 import BeautifulSoup

from fetch_web import extract_clean_text


mcp = FastMCP("web_mcp")

logging.basicConfig(level=logging.DEBUG)

MAPPINGS_FILE = Path("/app/web/mappings.txt")
FILE_REGISTRY = {}

WEB_FILES_DIR = Path("/app/web/")

@mcp.tool()
async def web_fetch(url: str) -> str:
    """
    Tool for retrieving the HTML content of a web page.

    Args:
        url (str): The URL of the page to download.

    Returns:
        str: The text content of the page wrapped in an XML source tag.
    """
   
    path_string = FILE_REGISTRY.get(url)

    if not path_string:
        return f"Error: URL '{url}' not found. The website does not exist or there is a problem with the URL."
    
    try:
        full_path = WEB_FILES_DIR / path_string   
        logging.info(f"web_fetch called with URL: {url} file {full_path}")
        #if not full_path.exists():
        #    return f"Error: File {full_path} not found."

        html_content = full_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "html.parser")
        
        title, clean_text = extract_clean_text(soup)

        return f'<source url="{url}" title="{title}">\n{clean_text}\n</source>'
    except Exception as e:
        return f"Error when accessing URL {url}: {e}"

@mcp.tool()
async def register_new_file(web_url: str, path: str) -> str:
    """
    Registers a dynamically mapped web URL to a local file path.
    (This is typically called by the Gateway API, not the LLM).
    Only for testing purposes.
    """

    FILE_REGISTRY[web_url] = path
    logging.info(f"Registered new mapping: {web_url} -> {path}")
    return f"Successfully mapped {web_url} to RAM registry."



def load_registry():
    if not MAPPINGS_FILE.exists():
        logging.info("No mappings file found, starting with empty registry.")
        return

    try:
        with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                url, path = line.split("|", 1)
                FILE_REGISTRY[url] = path
        logging.info(f"Loaded {len(FILE_REGISTRY)} mappings from {MAPPINGS_FILE}")
    except Exception as e:
        logging.error(f"Failed to load mappings: {e}")

if __name__ == "__main__":
    # Starts the server — the default transport is stdio (communication via stdin/stdout)    
    load_registry()
    
    mcp.run()
