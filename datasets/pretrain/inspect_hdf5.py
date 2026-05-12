import h5py
import os
import glob
import random

# --- CONFIG ---
# Change this to the folder you want to check (Baseline or Reasoning)
DATA_ROOT = "/home/ubuntu/human-policy/data/recordings/processed_reasoning"
# ----------------

def inspect_random_file():
    # Find all hdf5 files
    search_path = os.path.join(DATA_ROOT, "**", "*.hdf5")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print(f"❌ No files found in {DATA_ROOT}")
        return

    # Pick one at random
    file_path = random.choice(files)
    print(f"🔍 Inspecting: {os.path.basename(file_path)}")
    print(f"📂 Path: {file_path}")
    print("-" * 40)

    try:
        with h5py.File(file_path, 'r') as f:
            # 1. Check Attributes (where text lives)
            print("【Attributes】")
            if 'language_instruction' in f.attrs:
                text = f.attrs['language_instruction']
                print(f"   ✅ language_instruction: \"{text}\"")
            else:
                print("   ❌ language_instruction: NOT FOUND")
            
            # Print other attributes just for info
            for k, v in f.attrs.items():
                if k != 'language_instruction':
                    print(f"   ℹ️  {k}: {v}")

            # 2. Check Datasets (images, actions)
            print("\n【Datasets】")
            print(f"   Keys found: {list(f.keys())}")
            if 'actions_48d' in f:
                print(f"   ✅ actions_48d shape: {f['actions_48d'].shape}")
            
    except Exception as e:
        print(f"❌ Error opening file: {e}")

if __name__ == "__main__":
    inspect_random_file()