

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document



BASE_URL    = "https://iiitbhopal.ac.in"
DATA_FOLDER = "data"
PDF_FOLDER  = os.path.join(DATA_FOLDER, "website_pdfs")
TEXT_FOLDER = os.path.join(DATA_FOLDER, "web_text")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IIITBhopalBot/1.0; "
        "+https://iiitbhopal.ac.in)"
    )
}



def clean_text(soup: BeautifulSoup) -> str:
    """Strip boilerplate tags and return clean plain text."""
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def is_valid_url(url: str) -> bool:
    """Only follow links that stay on the IIIT Bhopal domain."""
    parsed = urlparse(url)
    return (
        parsed.scheme in ("http", "https")
        and BASE_URL.replace("https://", "") in parsed.netloc
        and not parsed.path.endswith((".jpg", ".jpeg", ".png", ".gif", ".svg",
                                      ".css", ".js", ".ico", ".xml", ".zip"))
    )


def safe_get(url: str, timeout: int = 8) -> requests.Response | None:
    """GET with retries and a polite delay."""
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == 0:
                time.sleep(1)
            else:
                print(f"   ⚠️  Could not fetch {url}: {exc}")
    return None



def get_links(url: str) -> list[str]:
    """Return all valid in-domain links found on a page."""
    resp = safe_get(url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        absolute = urljoin(url, a["href"].strip())
        if is_valid_url(absolute):
            
            clean = absolute.split("?")[0].split("#")[0].rstrip("/")
            links.add(clean)

    return list(links)


def crawl(start_url: str = BASE_URL, max_pages: int = 50) -> list[str]:
    """
    BFS crawl starting from start_url.
    Returns list of successfully visited URLs.
    """
    visited: set[str] = set()
    queue:   list[str] = [start_url]
    crawled: list[str] = []

    print(f"\n{'─'*55}")
    print(f"  🌐  Starting crawl → {start_url}")
    print(f"{'─'*55}")

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue

        visited.add(url)
        print(f"  [{len(visited):>3}/{max_pages}]  {url}")

        time.sleep(0.3)          
        crawled.append(url)

        for link in get_links(url):
            if link not in visited:
                queue.append(link)

    print(f"\n    Crawled {len(crawled)} pages total.\n")
    return crawled




def download_pdfs(links: list[str]) -> int:
    
    downloaded = 0

    print(f"{'─'*55}")
    print("   Downloading PDFs…")
    print(f"{'─'*55}")

    for link in links:
        if not link.lower().endswith(".pdf"):
            continue

        filename = os.path.basename(urlparse(link).path) or "file.pdf"
        path = os.path.join(PDF_FOLDER, filename)

        if os.path.exists(path):
            print(f"     Already exists: {filename}")
            continue

        resp = safe_get(link, timeout=15)
        if resp is None:
            continue

        try:
            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"   Saved: {filename}")
            downloaded += 1
        except OSError as exc:
            print(f"   Write error for {filename}: {exc}")

    print(f"\n    Downloaded {downloaded} new PDF(s).\n")
    return downloaded




def extract_and_save_text(links: list[str]) -> list[Document]:
    
    docs: list[Document] = []

    print(f"{'─'*55}")
    print("  📝  Extracting page text…")
    print(f"{'─'*55}")

    for link in links:
        if link.lower().endswith(".pdf"):
            continue                      

        resp = safe_get(link)
        if resp is None:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        text = clean_text(soup)

        if len(text) < 200:               
            continue

       
        fname = (
            link.replace("https://", "")
                .replace("http://", "")
                .replace("/", "_")[:120]
        )
        path = os.path.join(TEXT_FOLDER, fname + ".txt")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as exc:
            print(f"   Could not save {link}: {exc}")
            continue

        docs.append(Document(
            page_content=text,
            metadata={"source": link, "type": "website"},
        ))
        print(f"   Saved: {link}")

    print(f"\n  📊  Extracted text from {len(docs)} page(s).\n")
    return docs



def load_all_documents() -> list[Document]:
    """
    Walk the data/ folder and load every PDF and .txt file into
    LangChain Documents.  Call this from ragpipe.py AFTER crawling.
    """
    all_docs: list[Document] = []

    if not os.path.isdir(DATA_FOLDER):
        print(
            "    'data/' folder not found.\n"
            "     Run  python inges.py  first to crawl and download content."
        )
        return all_docs

    print(f"\n{'─'*55}")
    print("    Loading documents from disk…")
    print(f"{'─'*55}")

    for root, _, files in os.walk(DATA_FOLDER):
        for file in files:
            path = os.path.join(root, file)

            try:
                if file.lower().endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    docs   = loader.load()
                    for d in docs:
                        d.metadata.setdefault("type", "pdf")
                        d.metadata.setdefault("source", path)
                    all_docs.extend(docs)
                    print(f"    PDF loaded ({len(docs)} pages): {file}")

                elif file.lower().endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                    docs   = loader.load()
                    for d in docs:
                        d.metadata.setdefault("type", "website")
                        d.metadata.setdefault("source", path)
                    all_docs.extend(docs)
                    print(f"   TXT loaded: {file}")

            except Exception as exc:        # noqa: BLE001
                print(f"   Failed to load {file}: {exc}")

    print(f"\n   TOTAL DOCUMENTS LOADED: {len(all_docs)}\n")
    return all_docs



if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  🚀  IIIT Bhopal Ingestion Pipeline")
    print("═"*55)

    print("\n[1/4] Crawling website…")
    links = crawl(BASE_URL, max_pages=50)

    print("[2/4] Downloading PDFs…")
    download_pdfs(links)

    print("[3/4] Extracting page text…")
    extract_and_save_text(links)

    print("[4/4] Verifying document load…")
    docs = load_all_documents()

    print("═"*55)
    print(f"    Ingestion complete! {len(docs)} document chunks ready.")
    print("     Now run:  python rag_pipes.py")
    print("═"*55 + "\n")