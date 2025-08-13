from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import json
import openai
import base64
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
embedding_file_path = os.path.join(BASE_DIR, "site_embeddings.json")

app = Flask(__name__)
CORS(app, origins=["https://nelsojo.github.io"])

def normalize_url(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    netloc = parsed.netloc.lower()
    # Remove query and fragment for normalization
    normalized = urlunparse(parsed._replace(path=path, netloc=netloc, query="", fragment=""))
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

def scrape_html_from_url(url, visited, base_netloc=None, base_path_prefix="/"):
    norm_url = normalize_url(url)
    if norm_url in visited:
        return []

    visited.add(norm_url)
    site_data = []
    print(f"Scraping: {url} (Visited count: {len(visited)})")

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if 'text/html' not in response.headers.get('Content-Type', ''):
            return []
        response.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.text, 'lxml')

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
                "href": urljoin(url, a["href"])
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

    if not base_path_prefix:
        base_path_prefix = "/"

    # Crawl all internal links (same domain)
    for a_tag in soup.find_all('a', href=True):
        full_url = urljoin(url, a_tag['href'])
        parsed_full = urlparse(full_url)

        if parsed_full.netloc.lower() == base_netloc:
            # Enforce the path prefix restriction here:
            path = parsed_full.path
            prefix = base_path_prefix.rstrip('/')

            if prefix == "":
                prefix = "/"

            # Only crawl if path == prefix or path starts with prefix + "/"
            if path == prefix or path.startswith(prefix + "/"):
                norm_full_url = normalize_url(full_url)
                if norm_full_url not in visited:
                    site_data.extend(scrape_html_from_url(
                        full_url,
                        visited,
                        base_netloc=base_netloc,
                        base_path_prefix=base_path_prefix
                    ))


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

    parsed = urlparse(url)
    base_netloc = parsed.netloc.lower()
    base_path_prefix = parsed.path.rstrip('/')
    if not base_path_prefix:
        base_path_prefix = "/"

    visited = set()
    results = scrape_html_from_url(
        url,
        visited,
        base_netloc=base_netloc,
        base_path_prefix=base_path_prefix
    )

    return jsonify({
        "base_netloc": base_netloc,
        "base_path_prefix": base_path_prefix,
        "pages": results
    })

# Decode your API key once at startup
encoded_api_key = "c2stcHJvai1WSVhfUnJ5bEw4ZW5ZbHFTNnFndzBmNjYyNVl1YVZIS3FIbHhwR05uM2tfc24taTlfMGhtWVRicUhkZnpZT3N6dUo4N2NsV09BMVQzQmxia0ZKd2s5ajBqQ0VUUDMtR19kdjlRRnNDZ052THZHR1RrN2EyYUlPY19DM2hTSjVDai1kWXRzeDlzRkVLdVBoWXQzSThWd3JTRzdVSUE="
decoded_api_key = base64.b64decode(encoded_api_key).decode('utf-8')
client = openai.OpenAI(api_key=decoded_api_key)


@app.route('/upload', methods=['POST'])
def upload_json():
    # Expecting 'file' (JSON) + 'base_netloc' + 'base_path_prefix' in form data (multipart)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get base_netloc and base_path_prefix from form data or JSON
    base_netloc = request.form.get('base_netloc') or request.args.get('base_netloc')
    base_path_prefix = request.form.get('base_path_prefix') or request.args.get('base_path_prefix') or "/"
    if not base_netloc:
        return jsonify({"error": "Missing base_netloc"}), 400

    # Normalize base_path_prefix
    base_path_prefix = base_path_prefix.rstrip('/')
    if not base_path_prefix:
        base_path_prefix = "/"

    try:
        site_data = json.load(file)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON file: {str(e)}"}), 400

    # If site_data is a dict with "pages" key, extract it
    if isinstance(site_data, dict) and "pages" in site_data:
        pages = site_data["pages"]
    elif isinstance(site_data, list):
        pages = site_data
    else:
        return jsonify({"error": "JSON file must be a list or contain a 'pages' key."}), 400

    filtered_site_data = []
    for page in pages:
        url = page.get("url", "")
        parsed_url = urlparse(url)
        if parsed_url.netloc.lower() == base_netloc:
            path = parsed_url.path
            if (path == base_path_prefix or path.startswith(base_path_prefix.rstrip("/") + "/")):
                filtered_site_data.append(page)


    if not filtered_site_data:
        return jsonify({"error": "No pages matched the specified domain and path prefix"}), 400

    texts_to_embed = []
    metadata = []

    for page in filtered_site_data:
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

    site_embeddings = []

    for i, text in enumerate(texts_to_embed):
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            site_embeddings.append({
                "metadata": metadata[i],
                "text": text,
                "embedding": embedding
            })
        except Exception as e:
            return jsonify({"error": f"Embedding failed at text #{i}: {str(e)}"}), 500

    with open(embedding_file_path, "w", encoding="utf-8") as f:
        json.dump(site_embeddings, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings file at {embedding_file_path}")
    print(f"File exists: {os.path.exists(embedding_file_path)}")

    return jsonify(site_embeddings)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
