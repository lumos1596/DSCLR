#!/usr/bin/env python3
"""
Upload dual_queries files to Hugging Face dataset repository.

Usage:
    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python scripts/upload_dual_queries_to_hf.py
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file

REPO_ID = "lumos2548/DSCLR-dual-queries"
DATASET_ROOT = Path("/home/luwa/Documents/DSCLR/dataset")

def get_all_dual_queries_files():
    """Find all dual_queries JSONL files."""
    files = []
    for p in DATASET_ROOT.rglob("*.jsonl"):
        if "dual_queries" in str(p).lower() or "dual_queries" in p.parent.name:
            files.append(p)
    return sorted(files)

def main():
    # Check if logged in
    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Not logged in to Hugging Face. Please run:")
        print(f"   /home/luwa/.conda/envs/dsclr/bin/huggingface-cli login")
        sys.exit(1)

    # Create repo if not exists
    try:
        create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"✅ Repository created/verified: {REPO_ID}")
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        sys.exit(1)

    # Get all files
    files = get_all_dual_queries_files()
    print(f"📊 Found {len(files)} dual_queries files")

    # Upload each file
    for i, file_path in enumerate(files, 1):
        # Compute relative path from dataset root
        rel_path = file_path.relative_to(DATASET_ROOT)
        # Use path as repo path (preserve directory structure)
        repo_path = str(rel_path)

        print(f"[{i}/{len(files)}] Uploading: {repo_path}")
        try:
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"❌ Failed to upload {repo_path}: {e}")
            continue

    print(f"\n✅ Upload complete! View at: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()