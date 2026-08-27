import argparse
from huggingface_hub import HfApi, ModelCard, errors
import os

import config

# Example usage:
# python hf_model.py --organization "DT4H" --name "my-model" \
# --model_path . --description 'Medical NER model' \
# --language en --token YOUR_TOKEN --task token-classification --framework pytorch


def create_model_card(name, description, language, tags, task, framework): 
    """
    Gets main information and creates a model card using the template in config.py
    """ 
    text = config.description_text(name, description, language, tags, task, framework)
    # Using the Template
    card = ModelCard(content=text)

    return card

def push_to_huggingface(repo_id, model_path, card, token, private):
    api = HfApi()

    try:
        # Check if repository exists by trying to fetch its info
        api.repo_info(repo_id=repo_id, repo_type=config.repo_type)
        print(f"Repository '{repo_id}' already exists.")
    except errors.RepositoryNotFoundError:
        # If repository does not exist, create it
        print(f"Repository '{repo_id}' does not exist. Creating...")
        api.create_repo(token=token, repo_id=repo_id, private=private, repo_type=config.repo_type)
    except Exception as e:
        print(f"Error checking repository: {e}")
        raise e

    # Upload model files (all files from model_path directory)
    for root, _, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, model_path)
            
            # Skip common non-model files
            if file.startswith('.'):
                continue
                
            print(f"Uploading {relative_path}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=relative_path,
                repo_id=repo_id,
                repo_type=config.repo_type,
                token=token,
            )

    # Push model card
    card.push_to_hub(
        repo_id,
        token=token,
        repo_type=config.repo_type,
    )
    
def main():

    parser = argparse.ArgumentParser(description="Push model and model card to Hugging Face")
    parser.add_argument("--organization", default="DT4H", help="Organization to push the model to")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument("--model_path", required=True, help="Path to the model files")
    parser.add_argument("--name", required=True, help="Name of the model")
    parser.add_argument("--description", required=True, help="Description of the model")
    parser.add_argument("--language", required=True, help="Language of the model")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument("--task", default="token-classification", choices=config.model_tasks, help="Task type of the model")
    parser.add_argument("--framework", default="pytorch", help="Framework used (e.g., pytorch, tensorflow)")
    parser.add_argument("--tags", nargs="+", default=[], help="Tags for the model")
    parser.add_argument("--collection_name", help="Name of existing collection to add model to (optional)")

    args = parser.parse_args()

    repo_id = args.name.replace(" ", "-").lower()
    repo_id = f"{args.organization}/{repo_id}" if args.organization else repo_id

    # Create model card
    card = create_model_card(args.name, args.description, args.language, args.tags, args.task, args.framework)

    # Push model and card to Hugging Face
    push_to_huggingface(repo_id, args.model_path, card, args.token, private=args.private)

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"Model and card successfully pushed to {repo_url}")

    # Handle collection (optional)
    if args.collection_name:
        collection_slug = f"{args.organization}/{args.collection_name.replace(' ', '-').lower()}"
        
        # Check if collection exists and add model if it does
        if config.collection_exists(collection_slug, args.token):
            config.add_model_to_collection(collection_slug, repo_id, args.token)
        else:
            print(f"Note: Collection '{collection_slug}' does not exist. You can:")
            print(f"  1. Create it manually at https://huggingface.co/collections/new")
            print(f"  2. Run this script again with the same model name to add it to the collection")

if __name__ == "__main__":
    main()