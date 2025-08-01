import wikipedia
import time
import os
import random
from bs4 import BeautifulSoup
import requests

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "..", "data", "raw", "wiki_crawl")
VISITED_FILE = os.path.join(SAVE_DIR, "visited.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

def load_visited():
    if os.path.exists(VISITED_FILE):
        with open(VISITED_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f.readlines())
    return set()

def save_visited(visited):
    with open(VISITED_FILE, "w", encoding="utf-8") as f:
        for item in visited:
            f.write(item + "\n")

def get_links(title):
    try:
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        links = set()
        for a in soup.select("a[href^='/wiki/']"):
            href = a['href']
            if ':' in href:
                continue
            article = href.split('/wiki/')[-1]
            if article:
                links.add(article.replace("_", " "))
        return links
    except Exception as e:
        print(f"[!] Failed to get links for {title}: {e}")
        return set()

def fetch_and_save(title):
    try:
        content = wikipedia.page(title, auto_suggest=False).content
        filename = os.path.join(SAVE_DIR, title.replace("/", "_") + ".txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Saved: {title}")
        return True
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        print(f"[-] Skipped: {title} (ambiguous or not found)")
    except Exception as e:
        print(f"[!] Error fetching {title}: {e}")
    return False

def collect(seed_topics, max_pages=1000, delay=3):
    visited = load_visited()
    queue = list(seed_topics)

    while queue and len(visited) < max_pages:
        topic = queue.pop(0)
        if topic in visited:
            continue

        success = fetch_and_save(topic)
        if success:
            visited.add(topic)
            new_links = get_links(topic)
            queue.extend(link for link in new_links if link not in visited)

        save_visited(visited)
        time.sleep(delay + random.uniform(0, 1))

    print(f"✅ Finished crawling {len(visited)} Wikipedia pages.")