repo_type = "model"

# Tags for models
model_tasks = [
    "token-classification",
]

def collection_exists(collection_id, token):
    """
    Check if a collection exists on Hugging Face Hub.
    Returns True if exists, False otherwise.
    """
    from huggingface_hub import get_collection
    
    try:
        get_collection(collection_id, token=token)
        return True
    except Exception:
        return False

def add_model_to_collection(collection_id, repo_id, token):
    """
    Add a model to an existing collection.
    Only adds if collection exists.
    """
    from huggingface_hub import add_collection_item
    
    if not collection_exists(collection_id, token):
        print(f"Collection {collection_id} does not exist. Skipping collection add.")
        return False
    
    try:
        add_collection_item(collection_id, item_id=repo_id, item_type=repo_type, token=token, exists_ok=True)
        print(f"Model {repo_id} added to collection {collection_id}")
        return True
    except Exception as e:
        print(f"Error adding model to collection: {e}")
        return False

def description_text(name, description, language, tags, task, framework):
    """
    Template for model card - metadata and minimal documentation
    """
    metadata = f"""---
language: {language}
task_ids:
  - {task}
tags:
  - {', '.join(tags) if tags else 'model'}
---

## Model Card for {" ".join(name.split("_")).title()}

### Description

{description}

### Model Details

- **Language**: {language.upper()}
- **Task**: {task}
- **Framework**: {framework}

### How to Use

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "DT4H/{name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

```

### Acknowledgments
This work received funding from the European Union’s Horizon Europe research and innovation programme under Grant Agreement No. 101057849 (DataTools4Heart project).
"""
    return metadata