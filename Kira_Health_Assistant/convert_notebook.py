"""
Convert VSCode XML notebook format to standard Jupyter JSON format
"""
import json
import xml.etree.ElementTree as ET

def parse_vscode_notebook(xml_content):
    """Parse VSCode XML notebook and convert to Jupyter JSON format"""
    cells = []
    
    # Wrap in a root element to make it valid XML
    wrapped_xml = f"<root>{xml_content}</root>"
    
    try:
        root = ET.fromstring(wrapped_xml)
        
        # Find all VSCode.Cell elements
        for cell_elem in root.findall('.//{http://schemas.microsoft.com/vscode}Cell') + root.findall('.//VSCode.Cell'):
            language = cell_elem.get('language', 'python')
            content = cell_elem.text or ""
            
            # Split into lines and format properly for Jupyter
            lines = content.split('\n')
            # Remove empty first/last lines if present
            if lines and not lines[0].strip():
                lines = lines[1:]
            if lines and not lines[-1].strip():
                lines = lines[:-1]
            
            # Add newlines to all but the last line
            source_lines = [line + '\n' for line in lines[:-1]]
            if lines:
                source_lines.append(lines[-1])
            
            cell = {
                "cell_type": "markdown" if language == "markdown" else "code",
                "metadata": {},
                "source": source_lines
            }
            
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            
            cells.append(cell)
    except:
        # If XML parsing fails, try simple text parsing
        lines = xml_content.split('\n')
        current_cell = None
        current_content = []
        
        for line in lines:
            if '<VSCode.Cell' in line:
                # Save previous cell if exists
                if current_cell is not None and current_content:
                    content_text = '\n'.join(current_content)
                    source_lines = [l + '\n' for l in current_content[:-1]]
                    if current_content:
                        source_lines.append(current_content[-1])
                    
                    cell = {
                        "cell_type": current_cell,
                        "metadata": {},
                        "source": source_lines
                    }
                    if current_cell == "code":
                        cell["execution_count"] = None
                        cell["outputs"] = []
                    cells.append(cell)
                
                # Start new cell
                if 'language="markdown"' in line:
                    current_cell = "markdown"
                else:
                    current_cell = "code"
                current_content = []
            elif '</VSCode.Cell>' in line:
                # End current cell
                if current_cell is not None and current_content:
                    source_lines = [l + '\n' for l in current_content[:-1]]
                    if current_content:
                        source_lines.append(current_content[-1])
                    
                    cell = {
                        "cell_type": current_cell,
                        "metadata": {},
                        "source": source_lines
                    }
                    if current_cell == "code":
                        cell["execution_count"] = None
                        cell["outputs"] = []
                    cells.append(cell)
                    current_cell = None
                    current_content = []
            elif current_cell is not None:
                current_content.append(line)
    
    # Create Jupyter notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

# Read the VSCode XML notebook
with open('medical_chatbot_finetune.ipynb', 'r', encoding='utf-8') as f:
    xml_content = f.read()

# Convert to Jupyter format
jupyter_notebook = parse_vscode_notebook(xml_content)

# Save as GitHub-compatible notebook
with open('medical_chatbot_finetune_github.ipynb', 'w', encoding='utf-8') as f:
    json.dump(jupyter_notebook, f, indent=1, ensure_ascii=False)

print("✓ Conversion complete!")
print("  Created: medical_chatbot_finetune_github.ipynb")
print("  This file is GitHub-compatible and ready to upload.")
