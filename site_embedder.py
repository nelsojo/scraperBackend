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

def normalize_url(url):
    # Normalize URL by stripping trailing slash, lowercasing host
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    netloc = parsed.netloc.lower()
    normalized = urlunparse(parsed._replace(path=path, netloc=netloc))
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

def scrape_html_from_url(url, visited, base_netloc=None):
    norm_url = normalize_url(url)
    if norm_url in visited:
        return []

    visited.add(norm_url)
    site_data = []
    print(f"Scraping: {url}")

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

    # Determine base netloc once
    if base_netloc is None:
        base_netloc = urlparse(url).netloc.lower()

    for a_tag in soup.find_all('a', href=True):
        full_url = urljoin(url, a_tag['href'])
        parsed_full = urlparse(full_url)
        # Only crawl internal links (same domain)
        if parsed_full.netloc.lower() == base_netloc:
            norm_full_url = normalize_url(full_url)
            if norm_full_url not in visited:
                site_data.extend(scrape_html_from_url(full_url, visited, base_netloc=base_netloc))

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

    texts_to_embed = []
    metadata = []

    for page in site_data:
        combined_texts = []
        
        # Add paragraphs
        combined_texts.extend(page.get("paragraphs", []))
        
        # Add list items (with optional length filter)
        for lst in page.get("lists", []):
            combined_texts.extend([item for item in lst if len(item.strip()) >= 30])

        for table in page.get("tables", []):
            for row in table.get("rows", []):
                combined_texts.extend([cell for cell in row if len(cell.strip()) >= 30])    
        
        # Add headings
        combined_texts.extend(page.get("headings", []))

        # Clean and dedupe once, after everything has been gathered
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

    # Optionally save to file here or just return as response
    with open(embedding_file_path, "w", encoding="utf-8") as f:
        json.dump(site_embeddings, f, ensure_ascii=False, indent=2)
    print(f"Saved embeddings file at {embedding_file_path}")
    print(f"File exists: {os.path.exists(embedding_file_path)}")

    return jsonify(site_embeddings)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
