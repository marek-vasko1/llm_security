import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
import unicodedata
import hashlib

# Install with: pip install requests beautifulsoup4
import requests
from bs4 import BeautifulSoup

"""
From https://www.scrapingbee.com/blog/parsel-python/#handling-real-world-html-content 29.03.2026
"""

@dataclass(frozen=True)
class PageText:
    """One extracted page of clean text."""
    url: str
    title: str
    text: str

def normalize_text(text: str) -> str:
    """Normalize whitespace while keeping meaningful line breaks."""
    text = text.replace("\u00a0", " ")  # non-breaking spaces are common in HTML
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append("")
            continue

        blank_run = 0
        cleaned_lines.append(re.sub(r"\s+", " ", line))

    return "\n".join(cleaned_lines).strip()


def extract_clean_text(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Remove common noise from a parsed page and return (title, clean_text).
    """
    # Drop obvious noise first
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Optional: drop common layout blocks on typical sites.
    for tag in soup.select("header, footer, nav, aside, .cookie, .cookie-banner, .ad, .ads, .sidebar"):
        tag.decompose()

    # Hidden content patterns (varies by site)
    for tag in soup.select(".hidden, [aria-hidden='true']"):
        tag.decompose()

    # Extract text with paragraph-ish breaks
    raw_text = soup.get_text(separator="\n\n", strip=True)
    clean_text = normalize_text(raw_text)

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return title, clean_text
