from mcp.server.fastmcp import FastMCP
import logging
from urllib.parse import urlparse

from pathlib import Path
from bs4 import BeautifulSoup

from fetch_web import extract_clean_text


mcp = FastMCP("web_mcp")

logging.basicConfig(level=logging.DEBUG)

@mcp.tool()
async def web_fetch(url: str) -> str:
    """
    Tool for retrieving the HTML content of a web page.

    Args:
        url (str): The URL of the page to download.

    Returns:
        str: The text content of the page wrapped in an XML source tag.
    """
    parsed_url = urlparse(url)

    path = parsed_url.netloc

    file_name = Path(path).stem
    
    full_path = Path("/app/web") / f"{file_name}.html"
    try:
        
        logging.info(f"web_fetch called with URL: {url} file {full_path}")
        if not full_path.exists():
            return f"Error: File {full_path} not found."

        html_content = full_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "html.parser")
        
        title, clean_text = extract_clean_text(soup)

        return f'<source url="{url}" title="{title}">\n{clean_text}\n</source>'
    except Exception as e:
        return f"Error when accessing URL {url}: {e}"

if __name__ == "__main__":
    # Starts the server — the default transport is stdio (communication via stdin/stdout)    
    mcp.run()
