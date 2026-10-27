"""Verified downloads and deterministic cleaning of Amazon Video Games (2014)."""
import ast
import gzip
import hashlib
import json
import os
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = 'https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html'
BASE_URL = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
FILES = {
 'reviews_Video_Games_5.json.gz': '035690169fa2bba30cd0ae0c3e7dbf7bc58efde65fc1742a9ce85986e283b09c',
 'meta_Video_Games.json.gz': 'b7bf4c724632874b931056e48957775822c6f0c9092f5bbf2d449694bec36180',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(cache=None):
    cache = Path(cache or os.environ.get('RECSYS_DATA_DIR', '.cache/amazon2014'))
    cache.mkdir(parents=True, exist_ok=True)
    for filename, checksum in FILES.items():
        path = cache / filename
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            print(f'Downloading {BASE_URL + filename}', flush=True)
            urllib.request.urlretrieve(BASE_URL + filename, temporary)
            if sha256(temporary) != checksum:
                raise ValueError(f'SHA256 mismatch: {temporary}')
            temporary.replace(path)
        if sha256(path) != checksum:
            raise ValueError(f'Corrupt file: {path}; remove it and retry download')
    return cache


def load_data(cache=None):
    cache = download(cache)
    rows = []
    with gzip.open(cache / 'reviews_Video_Games_5.json.gz', 'rt') as stream:
        for line in stream:
            review = json.loads(line)
            rows.append({
                'user': review.get('reviewerID'), 'item': review.get('asin'),
                'rating': review.get('overall'), 'timestamp': review.get('unixReviewTime'),
            })
    raw = pd.DataFrame(rows)
    events = raw.dropna(subset=['user', 'item', 'rating', 'timestamp']).copy()
    events = events[events.rating.between(1, 5) & (events.timestamp > 0)]
    events = events.sort_values(['timestamp', 'user', 'item'], kind='stable')
    # Keep the FIRST review of a pair so deduplication cannot import a future rating.
    events = events.drop_duplicates(['user', 'item'], keep='first').reset_index(drop=True)
    events.timestamp = events.timestamp.astype('int64')
    events.rating = events.rating.astype('float64')
    wanted = set(events.item)
    metadata = []
    metadata_raw_count = 0
    with gzip.open(cache / 'meta_Video_Games.json.gz', 'rt') as stream:
        for line in stream:
            product = ast.literal_eval(line)  # source is Python literals, NEVER eval
            metadata_raw_count += 1
            if product['asin'] in wanted:
                paths = product.get('categories', [])
                leaves = sorted({str(p[-1]) for p in paths if p})
                metadata.append({
                    'item': product['asin'], 'title': product.get('title', ''),
                    'categories': '|'.join(leaves), 'brand': product.get('brand', ''),
                    'has_description': bool(product.get('description', '')),
                })
    metadata = pd.DataFrame(metadata).drop_duplicates('item').set_index('item').reindex(sorted(wanted))
    metadata[['title','categories','brand']] = metadata[['title','categories','brand']].fillna('')
    audit = {
        'source': SOURCE, 'raw_reviews': len(raw), 'clean_reviews': len(events),
        'raw_users': raw.user.nunique(), 'raw_items': raw.item.nunique(),
        'clean_users': events.user.nunique(), 'clean_items': events.item.nunique(),
        'missing_fields': raw.isna().sum().to_dict(),
        'duplicate_pairs': int(raw.duplicated(['user','item']).sum()),
        'metadata_raw_products': metadata_raw_count,
        'metadata_missing_titles': int((metadata.title == '').sum()),
        'metadata_missing_categories': int((metadata.categories == '').sum()),
        'sha256': FILES,
    }
    assert 100_000 <= len(events) <= 2_000_000
    assert events.user.nunique() >= 1000 and events.item.nunique() >= 1000
    return events, metadata, audit
