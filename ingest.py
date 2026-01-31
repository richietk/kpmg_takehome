"""
Load and process finance dataset from Kaggle
Source: https://www.kaggle.com/datasets/akhiltheerthala/wikipedia-finance
"""
import json
from pathlib import Path
from tqdm import tqdm


def load_finance_jsonl(input_path, output_path="data/wiki_sample.json"):
    """
    Load finance/economics JSONL dataset and convert to expected format

    input_path: str, path to JSONL file
    output_path: str, output path for converted data
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    articles = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc="Loading finance articles")):
            item = json.loads(line)
            title = item.get("Title", "")
            # Generate Wikipedia URL from title
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else ""

            article = {
                "id": f"finance_{idx}",
                "title": title,
                "text": item.get("text_content", ""),
                "url": url
            }
            articles.append(article)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(articles)} articles from {input_path} to {output_path}")
