from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse, urlunparse
import datetime
import json
import openai
from tqdm import tqdm
import base64
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
embedding_file_path = os.path.join(BASE_DIR, "site_embeddings.json")


app = Flask(__name__)
CORS(app, origins=["https://nelsojo.github.io"])



def normalize_url(url):
    """
    Normalize a URL for consistent comparison.
    - Strips trailing slashes
    - Lowercases the scheme and host
    - Removes fragments and query strings
    - Resolves '/index.html' to '/'
    """
    parsed = urlparse(url)

    # Lowercase scheme and netloc
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Remove fragment and query
    path = parsed.path
    if path.endswith('/index.html'):
        path = path[:-10]  # remove '/index.html'
    path = path.rstrip('/') or '/'

    normalized = urlunparse((scheme, netloc, path, '', '', ''))
    return normalized

def clean_and_dedupe(texts, min_length=50):
    seen = set()
    cleaned_texts = []

    for text in texts:
        cleaned = text.strip()

        # Skip very short or empty text
        if len(cleaned) < min_length:
            continue

        # Skip duplicates
        if cleaned in seen:
            continue

        seen.add(cleaned)
        cleaned_texts.append(cleaned)

    return cleaned_texts


def is_valid_http_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https')

def is_internal_link(base_url, link):
    parsed_base = urlparse(base_url)
    parsed_link = urlparse(link)
    return (parsed_link.netloc == "" or parsed_link.netloc == parsed_base.netloc)


def rewrite_links_in_html(soup, base_url):
    """Rewrite all relative href/src attributes to absolute URLs with logging."""
    parsed_base = urlparse(base_url)

    # Detect repo prefix for GitHub Pages (username.github.io/repo)
    base_prefix = ""
    if parsed_base.netloc.endswith("github.io"):
        path_parts = parsed_base.path.strip("/").split("/")
        if len(path_parts) >= 1 and path_parts[0]:
            base_prefix = "/" + path_parts[0]  # first segment = repo name

    # Site root (with prefix if needed)
    site_root = f"{parsed_base.scheme}://{parsed_base.netloc}{base_prefix}"

    print(f"Base URL for rewriting links: {base_url}")
    print(f"Detected base prefix: {base_prefix if base_prefix else '(none)'}")
    print(f"Site root for root-relative URLs: {site_root}")

    for tag in soup.find_all(["a", "link", "script", "img"]):
        attr = "href" if tag.name in ["a", "link"] else "src"
        if tag.has_attr(attr):
            orig_url = tag[attr]
            print(f"Original {attr} in <{tag.name}>: {orig_url}")

            fixed_url = orig_url

            # Case 1: Root-relative path (/...)
            if orig_url.startswith("/"):
                fixed_url = urljoin(site_root + "/", orig_url.lstrip("/"))

            # Case 2: Relative to current page directory
            else:
                fixed_url = urljoin(base_url, orig_url)

            tag[attr] = fixed_url
            print(f"Rewritten {attr} to absolute URL: {fixed_url}")

    return soup




def has_repeated_segments(url, max_repeat=3):
    """Detect if any path segment repeats more than max_repeat times."""
    path = urlparse(url).path
    segments = path.strip("/").split("/")
    counts = {}
    for seg in segments:
        counts[seg] = counts.get(seg, 0) + 1
        if counts[seg] > max_repeat:
            return True
    return False


def scrape_html_from_url(url, visited, base_netloc=None, base_path_prefix=None, depth=0, max_depth=5):
    """Scrape HTML from a URL, following internal links safely with recursion limits."""
    if depth > max_depth:
        print(f"Max depth reached at {url}")
        return []

    norm_url = normalize_url(url)
    if norm_url in visited:
        return []

    visited.add(norm_url)
    site_data = []
    print(f"Scraping: {url} (depth={depth})")

    # First attempt: Requests
    soup = None
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            soup = BeautifulSoup(response.text, "lxml")
    except Exception as e:
        print(f"Requests failed for {url}: {e}")

    # Fallback: Playwright
    if soup is None or len(soup.get_text(strip=True)) < 300:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=20000)
                html = page.content()
                browser.close()
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            print(f"Playwright failed for {url}: {e}")
            return []

    # Ensure path ends with slash
    parsed_url = urlparse(url)
    path = parsed_url.path
    if not path.endswith('/'):
        path += '/'
    url_with_slash = urlunparse(parsed_url._replace(path=path))
    soup = rewrite_links_in_html(soup, url_with_slash)

    def get_clean_text(el):
        return ' '.join(el.stripped_strings)

    # Extract content
    page_data = {
        "url": url,
        "title": soup.title.string.strip() if soup.title and soup.title.string else "",
        "headings": [h.get_text(strip=True) for h in soup.find_all(['h1','h2','h3','h4','h5','h6'])],
        "paragraphs": [get_clean_text(el) for el in soup.find_all(['p','div','span','td']) if get_clean_text(el)],
        "lists": [[li.get_text(strip=True) for li in ul.find_all('li') if li.get_text(strip=True)]
                  for ul in soup.find_all(['ul','ol'])],
        "links": [{"text": a.get_text(strip=True), "href": a["href"]}
                  for a in soup.find_all("a", href=True) if a["href"].strip() and a.get_text(strip=True)],
        "tables": []
    }

    for table in soup.find_all('table'):
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        rows = []
        for tr in table.find_all('tr'):
            if tr.find_all('th'):
                continue
            row = [td.get_text(strip=True) for td in tr.find_all('td')]
            if row:
                rows.append(row)
        if headers or rows:
            page_data["tables"].append({"headers": headers, "rows": rows})

    site_data.append(page_data)

    # Set base URL info for internal link filtering
    if base_netloc is None:
        base_netloc = parsed_url.netloc.lower()
    if base_path_prefix is None:
        base_path_prefix = parsed_url.path.rstrip('/')
        if base_path_prefix == "":
            base_path_prefix = "/"
        elif not base_path_prefix.endswith('/'):
            base_path_prefix += '/'

    # Recursively scrape internal links
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        if not href:
            continue

        full_url = urljoin(url, href)
        parsed_href = urlparse(full_url)

        # Skip external links
        if parsed_href.netloc.lower() != base_netloc:
            continue

        # Skip links outside base path prefix
        if not parsed_href.path.startswith(base_path_prefix):
            continue

        # Skip non-HTML resources
        if not parsed_href.path.endswith(('.html', '/')):
            continue

        # Skip URLs with repeated path segments
        if has_repeated_segments(full_url):
            print(f"Skipping repeated segments: {full_url}")
            continue

        norm_full_url = normalize_url(full_url)
        if norm_full_url not in visited:
            site_data.extend(
                scrape_html_from_url(full_url, visited, base_netloc, base_path_prefix, depth=depth+1, max_depth=max_depth)
            )

    return site_data



@app.route('/site_embeddings.json')
def serve_embeddings():
    if not os.path.exists(embedding_file_path):
        return jsonify([])  # Return empty array if file not found
    return send_file(embedding_file_path, mimetype='application/json')


@app.route('/scrape', methods=['POST'])
def scrape_route():
    data = request.get_json()
    url = data.get('url')
    if not url or not is_valid_http_url(url):
        return jsonify({"error": "Invalid URL"}), 400

    visited = set()
    results = scrape_html_from_url(url, visited)
    return jsonify(results)

# Decode your API key once at startup
encoded_api_key = "c2stcHJvai1WSVhfUnJ5bEw4ZW5ZbHFTNnFndzBmNjYyNVl1YVZIS3FIbHhwR05uM2tfc24taTlfMGhtWVRicUhkZnpZT3N6dUo4N2NsV09BMVQzQmxia0ZKd2s5ajBqQ0VUUDMtR19kdjlRRnNDZ052THZHR1RrN2EyYUlPY19DM2hTSjVDai1kWXRzeDlzRkVLdVBoWXQzSThWd3JTRzdVSUE="
decoded_api_key = base64.b64decode(encoded_api_key).decode('utf-8')
client = openai.OpenAI(api_key=decoded_api_key)

@app.route('/upload', methods=['POST'])
def upload_json():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        site_data = json.load(file)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON file: {str(e)}"}), 400

    preview = []
    count = 0
    first_item = True

    # Stream embeddings to file immediately
    with open(embedding_file_path, "w", encoding="utf-8") as f:
        f.write("[\n")  # start JSON array

        for page_idx, page in enumerate(site_data, start=1):
            # Combine texts from page
            texts_to_embed = []
            texts_to_embed.extend(page.get("paragraphs", []))
            for lst in page.get("lists", []):
                texts_to_embed.extend([item for item in lst if len(item.strip()) >= 30])
            for table in page.get("tables", []):
                for row in table.get("rows", []):
                    texts_to_embed.extend([cell for cell in row if len(cell.strip()) >= 30])
            texts_to_embed.extend(page.get("headings", []))

            # Deduplicate and clean
            texts_to_embed = clean_and_dedupe(texts_to_embed)

            for text_idx, text in enumerate(texts_to_embed, start=1):
                try:
                    response = client.embeddings.create(
                        input=text,
                        model="text-embedding-ada-002"
                    )
                    embedding = response.data[0].embedding
                    item = {
                        "metadata": {"url": page.get("url", ""), "title": page.get("title", "")},
                        "text": text,
                        "embedding": embedding
                    }

                    # Keep first 3 items for frontend preview
                    if len(preview) < 3:
                        preview.append(item)

                    # Write immediately to file
                    if not first_item:
                        f.write(",\n")
                    else:
                        first_item = False
                    json.dump(item, f, ensure_ascii=False)

                    count += 1

                    # Optional: print progress
                    if count % 10 == 0:
                        print(f"Processed {count} embeddings...")

                except Exception as e:
                    print(f"Embedding failed at page {page.get('url', '')} text #{text_idx}: {e}")
                    continue

        f.write("\n]")  # close JSON array

    print(f"Saved embeddings file at {embedding_file_path} ({count} embeddings)")
    return jsonify({"status": "completed", "count": count, "preview": preview})





if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
