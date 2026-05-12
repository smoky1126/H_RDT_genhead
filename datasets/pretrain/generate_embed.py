import os
import h5py
import torch
import glob
import argparse
from tqdm import tqdm
from transformers import T5EncoderModel, AutoTokenizer

# --- DEFAULT SETTINGS ---
# Local path to T5 model
DEFAULT_T5_PATH = "/home/ubuntu/H_RDT/models/t5-v1_1-xxl"
# ------------------------

def get_t5_embedding(text, tokenizer, model, device):
    # Tokenize and run model
    tokens = tokenizer(
        text, 
        return_tensors="pt", 
        padding="longest", 
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        output = model(**tokens)
    
    # Extract last hidden state (Batch, Seq, Dim)
    embedding = output.last_hidden_state.detach().cpu()
    return embedding

def main():
    parser = argparse.ArgumentParser(description="Generate .pt embeddings from HDF5 attributes.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the processed data folder")
    parser.add_argument("--t5_path", type=str, default=DEFAULT_T5_PATH, help="Path to local T5 model")
    args = parser.parse_args()

    # 1. Load Model
    print(f"🚀 Loading T5 Model from: {args.t5_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.t5_path)
        model = T5EncoderModel.from_pretrained(args.t5_path).to(device)
    except Exception as e:
        print(f"❌ Failed to load T5 model: {e}")
        return
    model.eval()

    # 2. Find Files
    search_path = os.path.join(args.data_root, "**", "*.hdf5")
    files = glob.glob(search_path, recursive=True)
    print(f"📂 Found {len(files)} HDF5 episodes in {args.data_root}")

    if len(files) == 0:
        print("⚠️ No files found! Check your path.")
        return

    # 3. Process Loop
    success_count = 0
    for file_path in tqdm(files, desc="Encoding"):
        try:
            # Read text from HDF5
            with h5py.File(file_path, 'r') as f:
                if 'language_instruction' not in f.attrs:
                    # Skip if missing (or use a fallback if you really want)
                    continue
                
                raw_text = f.attrs['language_instruction']
                
                # Handle decoding if stored as bytes
                if isinstance(raw_text, bytes):
                    text = raw_text.decode('utf-8')
                else:
                    text = str(raw_text)

            # Generate .pt file
            save_path = file_path.replace('.hdf5', '.pt')
            
            # Optimization: Skip if .pt already exists? 
            # (Uncomment next 2 lines to resume interrupted jobs)
            # if os.path.exists(save_path):
            #     continue

            embed = get_t5_embedding(text, tokenizer, model, device)
            # Also save token IDs for reasoning auxiliary loss
            tokens = tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=150
            )
            token_ids = tokens.input_ids.squeeze(0)  # (seq_len,)
            attn_mask = tokens.attention_mask.squeeze(0)  # (seq_len,)
            torch.save({
                "instruction": text,
                "embeddings": embed.squeeze(0),
                "task_name": text,
                "token_ids": token_ids,
                "attention_mask": attn_mask
            }, save_path)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")

    print(f"✅ Completed! Generated {success_count}/{len(files)} embeddings.")

if __name__ == "__main__":
    main()