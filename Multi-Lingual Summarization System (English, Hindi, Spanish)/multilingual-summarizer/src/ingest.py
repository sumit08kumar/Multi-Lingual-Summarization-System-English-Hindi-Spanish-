"""
Data Ingestion Module

This module contains functions for ingesting data from various sources
including text files, URLs, and datasets for multilingual summarization.
"""

import os
import requests
from typing import List, Dict, Optional
import json


def load_text_file(file_path: str) -> str:
    """Load text content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {str(e)}")


def load_multiple_files(file_paths: List[str]) -> List[Dict[str, str]]:
    """Load multiple text files and return as list of documents."""
    documents = []
    for i, file_path in enumerate(file_paths):
        try:
            content = load_text_file(file_path)
            documents.append({
                'id': f'doc_{i}',
                'content': content,
                'source': file_path
            })
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {str(e)}")
    return documents


def fetch_url_content(url: str) -> str:
    """Fetch text content from a URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Error fetching URL {url}: {str(e)}")


def load_dataset_sample(dataset_name: str = "cnn_dailymail", split: str = "train", num_samples: int = 100) -> List[Dict]:
    """Load a sample from a summarization dataset for testing."""
    # This is a placeholder - in a real implementation, you would use datasets library
    # from datasets import load_dataset
    # dataset = load_dataset(dataset_name, split=split)
    # return dataset.select(range(num_samples))
    
    # For now, return mock data
    mock_data = []
    for i in range(min(num_samples, 5)):  # Just 5 samples for demo
        mock_data.append({
            'id': f'sample_{i}',
            'article': f"This is a sample article {i} for testing the summarization system. " * 10,
            'highlights': f"This is a sample summary {i}.",
            'language': 'en'
        })
    return mock_data


def save_processed_data(data: List[Dict], output_path: str):
    """Save processed data to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


