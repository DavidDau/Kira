"""Debug script to check VSCode notebook format"""
with open('medical_chatbot_finetune.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Check first 500 characters
print("First 500 characters:")
print(content[:500])
print("\n" + "="*50 + "\n")

# Count cells
cell_count = content.count('<VSCode.Cell')
print(f"Number of cells found: {cell_count}")

# Show first cell
if '<VSCode.Cell' in content:
    start = content.find('<VSCode.Cell')
    end = content.find('</VSCode.Cell>', start) + len('</VSCode.Cell>')
    print("\nFirst cell:")
    print(content[start:end])
