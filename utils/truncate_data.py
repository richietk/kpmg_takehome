"""
Truncates original 313MB file to first 10,000 articles for demo purposes
Source: https://www.kaggle.com/datasets/akhiltheerthala/wikipedia-finance
"""
import json
from pathlib import Path

project_root = Path(__file__).parent.parent

input_file = project_root / "data" / "wikipedia_finance_agg.jsonl"
output_file = project_root / "data" / "wikipedia_finance_trunc1.jsonl"

max_entries = 10000
count = 0

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        if count >= max_entries:
            break
        line = line.strip()
        if not line:
            continue
        try:
            json_obj = json.loads(line)
            outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            count += 1
        except json.JSONDecodeError:
            continue

print(f"Wrote {count} entries to {output_file}")
