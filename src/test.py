import os
import shutil

cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
print("Clearing cache at:", cache_dir)
shutil.rmtree(cache_dir, ignore_errors=True)  # Deletes everything—safe, will re-download
