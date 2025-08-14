from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
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
CORS(app)
#CORS(app, origins=["https://nelsojo.github.io"])



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




def scrape_html_from_url(url, visited, base_netloc=None, base_path_prefix=None):
    norm_url = normalize_url(url)
    if norm_url in visited:
        return []

    visited.add(norm_url)
    site_data = []
    print(f"Scraping v3: {url}")

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if 'text/html' not in response.headers.get('Content-Type', ''):
            print(f"Skipping non-HTML content at {url}")
            return []
        response.raise_for_status()
    except requests.HTTPError as http_err:
        print(f"HTTP error for {url}: {http_err} (Status code: {http_err.response.status_code if http_err.response else 'N/A'})")
        return []
    except requests.RequestException as req_err:
        print(f"Request failed for {url}: {req_err}")
        return []

    soup = BeautifulSoup(response.text, 'lxml')

    # 🔹 Rewrite all relative links to absolute before scraping content
    parsed_url = urlparse(url)
    path = parsed_url.path

    


    # Ensure path ends with a slash (directory)
    if not path.endswith('/'):
        path += '/'

    url_with_slash = urlunparse(parsed_url._replace(path=path))
    soup = rewrite_links_in_html(soup, url_with_slash)

    def get_clean_text(el):
        return ' '.join(el.stripped_strings)

    page = {
        "url": url,
        "title": soup.title.string.strip() if soup.title and soup.title.string else "",
        "headings": [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
        "paragraphs": [get_clean_text(el) for el in soup.find_all(['p', 'div', 'span', 'td']) if get_clean_text(el)],
        "lists": [[li.get_text(strip=True) for li in ul.find_all('li') if li.get_text(strip=True)] for ul in soup.find_all(['ul', 'ol'])],
        "links": [
            {
                "text": a.get_text(strip=True),
                "href": a["href"]  # Already absolute now
            }
            for a in soup.find_all("a", href=True)
            if a["href"].strip() and a.get_text(strip=True)
        ],
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
            page["tables"].append({
                "headers": headers,
                "rows": rows
            })

    site_data.append(page)

    if base_netloc is None:
        base_netloc = urlparse(url).netloc.lower()

    if base_path_prefix is None:
        base_path_prefix = urlparse(url).path.rstrip('/')
        if base_path_prefix == "":
            base_path_prefix = "/"
        elif not base_path_prefix.endswith('/'):
            base_path_prefix += '/'


    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        if not href:
            continue

        # Resolve relative URLs to absolute URLs before any checks
        full_url = urljoin(url, href)
        parsed_href = urlparse(full_url)

        # Only follow internal links
        if parsed_href.netloc.lower() != base_netloc:
            continue

        # Only follow links within base path prefix
        if not parsed_href.path.startswith(base_path_prefix):
            continue

        # Only follow directories or HTML pages
        if not parsed_href.path.endswith(('.html', '/')):
            continue  # skip non-HTML resources

        # Normalize and check if visited
        norm_full_url = normalize_url(full_url)
        if norm_full_url not in visited:
            site_data.extend(scrape_html_from_url(full_url, visited, base_netloc, base_path_prefix))

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

import threading

def generate_embeddings_in_background(site_data, embedding_file_path):
    texts_to_embed = []
    metadata = []

    # Prepare texts and metadata
    for page in site_data:
        combined_texts = []
        combined_texts.extend(page.get("paragraphs", []))
        for lst in page.get("lists", []):
            combined_texts.extend([item for item in lst if len(item.strip()) >= 30])
        for table in page.get("tables", []):
            for row in table.get("rows", []):
                combined_texts.extend([cell for cell in row if len(cell.strip()) >= 30])
        combined_texts.extend(page.get("headings", []))

        filtered_texts = clean_and_dedupe(combined_texts)

        for text in filtered_texts:
            texts_to_embed.append(text)
            metadata.append({
                "url": page.get("url", ""),
                "title": page.get("title", "")
            })

    # Stream embeddings to file
    with open(embedding_file_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for i, text in enumerate(texts_to_embed):
            try:
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                item = {"metadata": metadata[i], "text": text, "embedding": embedding}

                if not first:
                    f.write(",\n")
                else:
                    first = False

                json.dump(item, f, ensure_ascii=False)
            except Exception as e:
                print(f"Embedding failed at text #{i}: {e}")
                continue
        f.write("\n]")
    print(f"Background embedding generation finished: {embedding_file_path}")


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

    # Start background thread
    thread = threading.Thread(target=generate_embeddings_in_background, args=(site_data, embedding_file_path))
    thread.start()

    return jsonify({"status": "started", "message": "Embedding generation running in background"})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
