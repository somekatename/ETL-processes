#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import random
import gc
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon, cdist
from scipy.stats import wasserstein_distance, skewnorm, lognorm
from sklearn.metrics import jaccard_score, mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from scipy.linalg import svd
import matplotlib.pyplot as plt
from gensim.models.fasttext import FastText
import psutil

warnings.filterwarnings('ignore')

ORIGINAL_DIR = "!short_files"
STATS_FILE = "stats_for_synthesis.json"
FASTTEXT_FILE = "service_files/fasttext_model.bin"

SIZES = {
    'products': 1000,
    'customers': 1000,
    'orders': 1000,
    'reviews': 1000,
    'items': 1000,
}

OUTPUT_DIR = "short_files_Qwen"

ANALYSIS_SAMPLE_SIZE = 1000
GENERATE_GRAPHS = True
SAVE_JSON_REPORT = True

def read_csv_smart(filepath, **kwargs):
    with open(filepath, 'r') as f:
        first_line = f.readline()
        if ';' in first_line:
            return pd.read_csv(filepath, sep=';', **kwargs)
        else:
            return pd.read_csv(filepath, **kwargs)

class StatsCollector:
    def __init__(self, sample_size=10000, text_unique_limit=5000):
        self.sample_size = sample_size
        self.text_unique_limit = text_unique_limit

    def collect_numeric_stats(self, series: pd.Series) -> dict:
        series = series.dropna()
        if len(series) == 0:
            return {}
        stats = {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'skew': float(series.skew()),
            'kurtosis': float(series.kurtosis()),
            'q1': float(series.quantile(0.25)),
            'median': float(series.quantile(0.50)),
            'q3': float(series.quantile(0.75)),
            'zero_ratio': float((series == 0).mean()),
        }
        for q in [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]:
            stats[f'quantile_{q}'] = float(series.quantile(q))
        if len(series) > self.sample_size:
            stats['sample'] = series.sample(self.sample_size, random_state=42).tolist()
        else:
            stats['sample'] = series.tolist()
        return stats

    def collect_categorical_stats(self, series: pd.Series) -> dict:
        series = series.dropna().astype(str)
        if len(series) == 0:
            return {}
        vc = series.value_counts()
        return {
            'categories': vc.index.tolist(),
            'probs': (vc / len(series)).tolist(),
            'n_unique': len(vc)
        }

    def collect_date_stats(self, series: pd.Series) -> dict:
        dates = pd.to_datetime(series, errors='coerce').dropna()
        if len(dates) == 0:
            return {}
        return {
            'min_date': dates.min().isoformat(),
            'max_date': dates.max().isoformat(),
        }

    def collect_text_stats(self, series: pd.Series) -> dict:
        series = series.dropna().astype(str)
        if len(series) == 0:
            return {}
        unique = series.unique()
        if len(unique) > self.text_unique_limit:
            unique = np.random.choice(unique, self.text_unique_limit, replace=False)
        lengths = series.str.len()
        return {
            'unique': unique.tolist(),
            'length_mean': float(lengths.mean()),
            'length_std': float(lengths.std()),
            'length_min': int(lengths.min()),
            'length_max': int(lengths.max()),
            'length_quantiles': [float(lengths.quantile(q)) for q in [0.25, 0.5, 0.75]]
        }

    def run(self, original_dir, output_file):
        all_stats = {}
        for table_name in ['products', 'customers', 'orders', 'order_items', 'reviews']:
            path = os.path.join(original_dir, f"{table_name}.csv")
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping {table_name}")
                continue
            print(f"Collecting stats for {table_name}...")
            try:
                df = read_csv_smart(path)
            except Exception as e:
                print(f"  Error reading {path}: {e}")
                all_stats[table_name] = {'error': str(e)}
                continue
            stats = {}
            if table_name == 'products':
                if 'product_name' in df.columns:
                    stats['product_name'] = self.collect_text_stats(df['product_name'])
                else:
                    print(f"  Warning: 'product_name' not found in {table_name}")
                    stats['product_name'] = {'error': 'column missing'}
                if 'category' in df.columns:
                    stats['category'] = self.collect_categorical_stats(df['category'])
                else:
                    stats['category'] = {'error': 'column missing'}
                if 'price' in df.columns:
                    stats['price'] = self.collect_numeric_stats(df['price'])
                else:
                    stats['price'] = {'error': 'column missing'}
                if 'stock_quantity' in df.columns:
                    stats['stock_quantity'] = self.collect_numeric_stats(df['stock_quantity'])
                else:
                    stats['stock_quantity'] = {'error': 'column missing'}
                if 'brand' in df.columns:
                    stats['brand'] = self.collect_categorical_stats(df['brand'])
                else:
                    stats['brand'] = {'error': 'column missing'}
            elif table_name == 'customers':
                if 'name' in df.columns:
                    stats['name'] = self.collect_text_stats(df['name'])
                else:
                    stats['name'] = {'error': 'column missing'}
                if 'email' in df.columns:
                    stats['email'] = self.collect_text_stats(df['email'])
                else:
                    stats['email'] = {'error': 'column missing'}
                if 'gender' in df.columns:
                    stats['gender'] = self.collect_categorical_stats(df['gender'])
                else:
                    stats['gender'] = {'error': 'column missing'}
                if 'signup_date' in df.columns:
                    stats['signup_date'] = self.collect_date_stats(df['signup_date'])
                else:
                    stats['signup_date'] = {'error': 'column missing'}
                if 'country' in df.columns:
                    stats['country'] = self.collect_categorical_stats(df['country'])
                else:
                    stats['country'] = {'error': 'column missing'}
            elif table_name == 'orders':
                if 'total_amount' in df.columns:
                    stats['total_amount'] = self.collect_numeric_stats(df['total_amount'])
                else:
                    stats['total_amount'] = {'error': 'column missing'}
                if 'payment_method' in df.columns:
                    stats['payment_method'] = self.collect_categorical_stats(df['payment_method'])
                else:
                    stats['payment_method'] = {'error': 'column missing'}
                if 'shipping_country' in df.columns:
                    stats['shipping_country'] = self.collect_categorical_stats(df['shipping_country'])
                else:
                    stats['shipping_country'] = {'error': 'column missing'}
                if 'order_date' in df.columns:
                    stats['order_date'] = self.collect_date_stats(df['order_date'])
                else:
                    stats['order_date'] = {'error': 'column missing'}
            elif table_name == 'order_items':
                if 'quantity' in df.columns:
                    stats['quantity'] = self.collect_numeric_stats(df['quantity'])
                else:
                    stats['quantity'] = {'error': 'column missing'}
                if 'unit_price' in df.columns:
                    stats['unit_price'] = self.collect_numeric_stats(df['unit_price'])
                else:
                    stats['unit_price'] = {'error': 'column missing'}
                if 'order_id' in df.columns:
                    items_per_order = df.groupby('order_id').size()
                    stats['_relation_items_per_order'] = dict(Counter(items_per_order))
                else:
                    stats['_relation_items_per_order'] = {'error': 'order_id missing'}
            elif table_name == 'reviews':
                if 'rating' in df.columns:
                    ratings = pd.to_numeric(df['rating'], errors='coerce').dropna().astype(int)
                    if len(ratings) > 0:
                        vc = ratings.value_counts().sort_index()
                        stats['rating_distribution'] = {
                            'values': vc.index.tolist(),
                            'probs': (vc / len(ratings)).tolist()
                        }
                    else:
                        stats['rating_distribution'] = {'error': 'no valid ratings'}
                else:
                    stats['rating_distribution'] = {'error': 'column missing'}
                if 'review_text' in df.columns:
                    stats['review_text'] = self.collect_text_stats(df['review_text'])
                else:
                    stats['review_text'] = {'error': 'column missing'}
                if 'review_date' in df.columns:
                    stats['review_date'] = self.collect_date_stats(df['review_date'])
                else:
                    stats['review_date'] = {'error': 'column missing'}

            all_stats[table_name] = stats

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to {output_file}")

def compute_text_embedding_metrics(orig_path, synth_path, text_col, sample_size=2000, ft_model=None):
    if ft_model is None:
        return {"error": "fastText model not provided"}
    try:
        df_orig = read_csv_smart(orig_path, nrows=sample_size)
        df_synth = read_csv_smart(synth_path, nrows=sample_size)
        if text_col not in df_orig.columns or text_col not in df_synth.columns:
            return {"error": f"Column {text_col} not found"}
        text_orig = df_orig[text_col].dropna().astype(str)
        text_synth = df_synth[text_col].dropna().astype(str)
        if len(text_orig) < 10 or len(text_synth) < 10:
            return {"error": "Sample too small"}

        def text_to_vector(text):
            words = str(text).lower().split()
            vecs = [ft_model.wv[w] for w in words if w in ft_model.wv]
            return np.mean(vecs, axis=0) if vecs else np.zeros(100)

        vec_orig = np.array([text_to_vector(t) for t in text_orig])
        vec_synth = np.array([text_to_vector(t) for t in text_synth])
        valid_orig = ~np.all(vec_orig == 0, axis=1)
        valid_synth = ~np.all(vec_synth == 0, axis=1)
        vec_orig = vec_orig[valid_orig]
        vec_synth = vec_synth[valid_synth]
        if len(vec_orig) == 0 or len(vec_synth) == 0:
            return {"error": "No valid vectors"}

        mean_orig = np.mean(vec_orig, axis=0)
        mean_synth = np.mean(vec_synth, axis=0)
        cos_sim = float(np.dot(mean_orig, mean_synth) / (np.linalg.norm(mean_orig) * np.linalg.norm(mean_synth)))

        gamma = 1.0 / (vec_orig.shape[1] * np.var(vec_orig, axis=0).mean())
        K_oo = rbf_kernel(vec_orig, gamma=gamma)
        K_ss = rbf_kernel(vec_synth, gamma=gamma)
        K_os = rbf_kernel(vec_orig, vec_synth, gamma=gamma)
        mmd = float(K_oo.mean() + K_ss.mean() - 2 * K_os.mean())

        cos_dist_orig = pairwise_distances(vec_orig, metric='cosine')
        cos_dist_synth = pairwise_distances(vec_synth, metric='cosine')
        triu_orig = cos_dist_orig[np.triu_indices_from(cos_dist_orig, k=1)]
        triu_synth = cos_dist_synth[np.triu_indices_from(cos_dist_synth, k=1)]
        if len(triu_orig) > 0 and len(triu_synth) > 0:
            bins = 30
            hist_orig, bin_edges = np.histogram(triu_orig, bins=bins, density=True)
            hist_synth, _ = np.histogram(triu_synth, bins=bin_edges, density=True)
            js_cos = float(jensenshannon(hist_orig, hist_synth))
        else:
            js_cos = None

        n_clusters = min(10, len(vec_orig)//10, len(vec_synth)//10)
        if n_clusters >= 2:
            combined = np.vstack([vec_orig, vec_synth])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(combined)
            labels_orig = kmeans.labels_[:len(vec_orig)]
            labels_synth = kmeans.labels_[len(vec_orig):]
            freq_orig = np.bincount(labels_orig, minlength=n_clusters) / len(labels_orig)
            freq_synth = np.bincount(labels_synth, minlength=n_clusters) / len(labels_synth)
            cluster_l1 = 0.5 * np.sum(np.abs(freq_orig - freq_synth))
        else:
            cluster_l1 = None

        return {
            "cosine_similarity_of_mean_vectors": cos_sim,
            "MMD": mmd,
            "pairwise_cosine_JS_divergence": js_cos,
            "cluster_distribution_L1": float(cluster_l1) if cluster_l1 is not None else None,
            "n_vectors_original": len(vec_orig),
            "n_vectors_synthetic": len(vec_synth)
        }
    except Exception as e:
        return {"error": str(e)}

def compare_stats(original_json, synthetic_json, output_file,
                  original_dir=ORIGINAL_DIR, synthetic_dir=OUTPUT_DIR,
                  fasttext_model_path=FASTTEXT_FILE,
                  text_sample_size=2000):
    with open(original_json, 'r') as f:
        orig = json.load(f)
    with open(synthetic_json, 'r') as f:
        synth = json.load(f)

    ft_model = None
    if os.path.exists(fasttext_model_path):
        try:
            ft_model = FastText.load(fasttext_model_path)
            print("FastText model loaded")
        except Exception as e:
            print(f"Could not load FastText model: {e}")

    report = {}
    for table, stats_orig in orig.items():
        if table not in synth:
            print(f"Warning: {table} not found in synthetic stats, skipping")
            continue
        stats_synth = synth[table]
        report[table] = {}

        for col, data_orig in stats_orig.items():
            if col not in stats_synth:
                continue
            data_synth = stats_synth[col]

            if isinstance(data_orig, dict) and 'error' in data_orig:
                continue
            if isinstance(data_synth, dict) and 'error' in data_synth:
                continue
            if 'min' in data_orig and 'max' in data_orig and 'mean' in data_orig:
                js_val = None
                wasser = None
                if 'sample' in data_orig and 'sample' in data_synth and data_orig['sample'] and data_synth['sample']:
                    sample_o = np.array(data_orig['sample'])
                    sample_s = np.array(data_synth['sample'])
                    if len(sample_o) > 0 and len(sample_s) > 0:
                        bins = 50
                        hist_o, bin_edges = np.histogram(sample_o, bins=bins, density=True)
                        hist_s, _ = np.histogram(sample_s, bins=bin_edges, density=True)
                        js_val = float(jensenshannon(hist_o, hist_s))
                        wasser = float(wasserstein_distance(sample_o, sample_s))
                report[table][col] = {
                    'type': 'numeric',
                    'original': {k: v for k, v in data_orig.items() if k != 'sample'},
                    'synthetic': {k: v for k, v in data_synth.items() if k != 'sample'},
                    'js_divergence': js_val,
                    'wasserstein_distance': wasser,
                    'mean_ratio': data_orig.get('mean') / data_synth.get('mean') if data_synth.get('mean') else None
                }
            elif 'categories' in data_orig and 'probs' in data_orig:
                probs_o = np.array(data_orig['probs'])
                probs_s = np.array(data_synth['probs'])
                cats_o = data_orig['categories']
                cats_s = data_synth['categories']
                all_cats = sorted(set(cats_o) | set(cats_s))
                freq_o = [data_orig['probs'][cats_o.index(c)] if c in cats_o else 0 for c in all_cats]
                freq_s = [data_synth['probs'][cats_s.index(c)] if c in cats_s else 0 for c in all_cats]
                l1 = 0.5 * sum(abs(fo - fs) for fo, fs in zip(freq_o, freq_s))
                js = jensenshannon(freq_o, freq_s)
                report[table][col] = {
                    'type': 'categorical',
                    'original': {k: v for k, v in data_orig.items()},
                    'synthetic': {k: v for k, v in data_synth.items()},
                    'l1_variation_distance': l1,
                    'js_divergence': float(js)
                }
            elif 'length_mean' in data_orig and 'length_std' in data_orig:
                length_stats = {
                    'original_length': {k: data_orig.get(k) for k in ['length_mean', 'length_std', 'length_min', 'length_max', 'length_quantiles']},
                    'synthetic_length': {k: data_synth.get(k) for k in ['length_mean', 'length_std', 'length_min', 'length_max', 'length_quantiles']},
                    'mean_ratio': data_orig.get('length_mean') / data_synth.get('length_mean') if data_synth.get('length_mean') else None
                }
                report[table][col] = {
                    'type': 'text',
                    **length_stats
                }
                if ft_model is not None:
                    orig_path = os.path.join(original_dir, f"{table}.csv")
                    synth_path = os.path.join(synthetic_dir, f"{table}.csv")
                    if os.path.exists(orig_path) and os.path.exists(synth_path):
                        emb = compute_text_embedding_metrics(orig_path, synth_path, col, text_sample_size, ft_model)
                        report[table][col]['embedding_metrics'] = emb
                    else:
                        report[table][col]['embedding_metrics'] = {"error": "CSV files not found"}
                else:
                    report[table][col]['embedding_metrics'] = {"error": "fastText model not loaded"}
            elif 'min_date' in data_orig and 'max_date' in data_orig:
                report[table][col] = {
                    'type': 'date',
                    'original_range': [data_orig.get('min_date'), data_orig.get('max_date')],
                    'synthetic_range': [data_synth.get('min_date'), data_synth.get('max_date')]
                }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Comparison report saved to {output_file}")

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gc.enable()

    collector = StatsCollector(sample_size=10000, text_unique_limit=5000)
    original_stats_file = os.path.join(OUTPUT_DIR, 'original_stats.json')
    collector.run(ORIGINAL_DIR, original_stats_file)

    synthetic_stats_file = os.path.join(OUTPUT_DIR, 'synthetic_stats.json')
    collector.run(OUTPUT_DIR, synthetic_stats_file)

    comparison_report = os.path.join(OUTPUT_DIR, 'comparison_report.json')
    compare_stats(original_stats_file, synthetic_stats_file, comparison_report)
