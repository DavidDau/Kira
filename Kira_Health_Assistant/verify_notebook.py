import json

with open('medical_chatbot_finetune.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"✓ Format: Jupyter {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"✓ Number of cells: {len(nb['cells'])}")
print(f"✓ Cell types: markdown={sum(1 for c in nb['cells'] if c['cell_type']=='markdown')}, code={sum(1 for c in nb['cells'] if c['cell_type']=='code')}")
print("\n" + "="*50)
print("✓ Your notebook is ALREADY in GitHub-compatible format!")
print("✓ You can push it to GitHub right now.")
print("="*50)
