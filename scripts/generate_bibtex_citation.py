#!/usr/bin/env python3
"""
Auto-generate BibTeX citation for blog posts based on config.yml and post YAML frontmatter.

Usage:
    python generate_bibtex_citation.py <post_file_path>
    
Example:
    python generate_bibtex_citation.py content/posts/250721-blogging.md
"""

import argparse
import os
import re
import sys
import yaml
from datetime import datetime
from pathlib import Path


def load_config(config_path):
    """Load the Hugo config.yml file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config.yml: {e}")
        sys.exit(1)


def parse_post_frontmatter(post_path):
    """Parse the YAML frontmatter from a blog post."""
    try:
        with open(post_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: Post file not found at {post_path}")
        sys.exit(1)
    
    # Extract YAML frontmatter
    pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(pattern, content, re.DOTALL)
    
    if not match:
        print(f"Error: No YAML frontmatter found in {post_path}")
        sys.exit(1)
    
    try:
        frontmatter = yaml.safe_load(match.group(1))
        return frontmatter
    except yaml.YAMLError as e:
        print(f"Error parsing frontmatter: {e}")
        sys.exit(1)


def generate_citation_key(author_last_name, year, title):
    """Generate a BibTeX citation key."""
    # Clean title: remove special characters and take first few words
    title_clean = re.sub(r'[^\w\s]', '', title.lower())
    title_words = title_clean.split()[:3]  # Take first 3 words
    title_part = ''.join(title_words)
    
    return f"{author_last_name.lower()}{year}{title_part}"


def get_post_url(base_url, post_path):
    """Generate the full URL for the blog post."""
    # Extract the post filename without extension
    post_filename = Path(post_path).stem
    
    # Remove the base_url trailing slash if present
    base_url = base_url.rstrip('/')
    
    return f"{base_url}/posts/{post_filename}/"


def format_date(date_str):
    """Parse and format date from various formats."""
    try:
        # Try parsing the ISO format from the frontmatter
        if 'T' in date_str:
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            # Try parsing simple date format
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        return {
            'year': str(date_obj.year),
            'month': date_obj.strftime('%B')  # Full month name
        }
    except ValueError as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        current_year = datetime.now().year
        return {'year': str(current_year), 'month': 'January'}


def generate_bibtex(config, frontmatter, post_path):
    """Generate BibTeX citation."""
    # Extract information
    title = frontmatter.get('title', 'Untitled')
    date = frontmatter.get('date', '2025-01-01')
    
    # Get author from config
    author_info = config.get('params', {}).get('author', ['Unknown Author'])
    if isinstance(author_info, list):
        author_name = author_info[0] if author_info else 'Unknown Author'
    else:
        author_name = author_info
    
    # Parse author name (assuming "First Last" format)
    name_parts = author_name.split()
    if len(name_parts) >= 2:
        author_last = name_parts[-1]
        author_first = ' '.join(name_parts[:-1])
        author_formatted = f"{author_last}, {author_first}"
    else:
        author_formatted = author_name
        author_last = author_name
    
    # Parse date
    date_info = format_date(date)
    
    # Generate citation key
    citation_key = generate_citation_key(author_last, date_info['year'], title)
    
    # Get URLs
    base_url = config.get('baseURL', 'https://example.com')
    post_url = get_post_url(base_url, post_path)
    
    # Get journal/blog name
    blog_title = config.get('title', 'Personal Blog')
    
    # Generate BibTeX
    bibtex = f"""@article{{{citation_key},
  title = {{{title}}},
  author = {{{author_formatted}}},
  journal = {{{blog_title}}},
  year = {{{date_info['year']}}},
  month = {{{date_info['month']}}},
  url = "{post_url}"
}}"""
    
    return bibtex


def main():
    parser = argparse.ArgumentParser(description='Generate BibTeX citation for blog posts')
    parser.add_argument('post_path', help='Path to the blog post markdown file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config("config.yml")
    
    # Parse post frontmatter
    frontmatter = parse_post_frontmatter(args.post_path)
    
    # Generate BibTeX
    bibtex = generate_bibtex(config, frontmatter, args.post_path)
    
    # Output
    print(bibtex)


if __name__ == '__main__':
    main()