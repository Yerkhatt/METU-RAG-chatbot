import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Set, List, Dict

def is_metu_domain(url: str) -> bool:
    parsed_url = urlparse(url)
    return parsed_url.netloc.endswith('.metu.edu.tr') or parsed_url.netloc == 'metu.edu.tr'

def load_previous_data(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}. Starting fresh.")
        return []

def scrape_page(url: str, session: requests.Session, scraped_urls: Set[str], 
                results: List[Dict], visited_urls: Set[str]) -> Set[str]:
    if not is_metu_domain(url):
        return set()
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        if 'text/html' not in response.headers.get('Content-Type', '').lower():
            return set()
        soup = BeautifulSoup(response.text, 'html.parser')
        if url not in scraped_urls:
            page_text = ' '.join(text for text in soup.stripped_strings if len(text) > 1)
            if page_text:
                results.append({'URL': url, 'content': page_text})
                scraped_urls.add(url)
        new_urls = set()
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if not href.startswith(('javascript:', '#', 'mailto:')):
                full_url = urljoin(url, href)
                if is_metu_domain(full_url) and full_url not in visited_urls:
                    new_urls.add(full_url)
        return new_urls
    except requests.RequestException:
        return set()

def save_results(file_path: str, results: List[Dict]) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def process_batch(urls: Set[str], session: requests.Session, scraped_urls: Set[str], 
                 scraped_results: List[Dict], visited_urls: Set[str]) -> Set[str]:
    next_urls = set()
    unvisited_urls = urls - visited_urls
    if not unvisited_urls:
        return next_urls
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [
            executor.submit(scrape_page, url, session, scraped_urls, 
                          scraped_results, visited_urls)
            for url in unvisited_urls
        ]
        visited_urls.update(unvisited_urls)
        for future in as_completed(futures):
            new_urls = future.result()
            next_urls.update(new_urls)
    return next_urls

file_path = 'metu_dataset.json'  # file to save the results
previous_data = load_previous_data(file_path)
scraped_urls = {entry['URL'] for entry in previous_data}
visited_urls = set()
scraped_results = previous_data

start_url = 'https://metu.edu.tr'   # try different starting URLs
max_depth = 10

with requests.Session() as session:
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    current_level_urls = {start_url}
    for level in range(max_depth):
        print(f"\nProcessing Level {level}...")
        next_level_urls = set()
        while current_level_urls:
            current_batch = set(list(current_level_urls)[:50])
            current_level_urls -= current_batch
            new_urls = process_batch(current_batch, session, scraped_urls, 
                                   scraped_results, visited_urls)
            next_level_urls.update(new_urls)
            save_results(file_path, scraped_results)
        print(f"Level {level} completed:")
        print(f"- URLs scraped: {len(scraped_results)}")
        print(f"- New URLs found: {len(next_level_urls)}")
        if not next_level_urls:
            print("No more URLs to process")
            break
        current_level_urls = next_level_urls

