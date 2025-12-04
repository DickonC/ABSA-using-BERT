#In this script we are taking the intitial large dataset and turning it into a new smaller dataset to train the BERT head on
#the output database will be the review tokens and the token labels (0,1,2 for Outside, B-ASP, I-ASP respectively)
#this will train our head to identify aspects from reviews. This will create a set numbered new dataset (1500)
# this is also trained for multi word aspects such as shoulder straps




import json
import pandas as pd
from transformers import AutoTokenizer

# 1. Setup
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define your "Weak Supervision" Keywords (The aspects we care about)
# Order matters slightly: put longer phrases first to avoid partial matching
aspect_keywords = [
    "bra strap", "shoulder straps", "arm holes", # Multi-word first
    "sleeves", "waist", "chest", "bust", "length", "shoulders", "arms", "hips", "thighs", "fit"
]

# 2. Function to auto-generate BIO tags
def create_bio_labels(text, keywords):
    # Tokenize the review text (return_offsets_mapping helps, but we'll do direct token matching for clarity)
    encoding = tokenizer(text, add_special_tokens=True)
    input_ids = encoding["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Initialize all labels to 0 (Outside)
    labels = [0] * len(tokens)
    
    # Check for each keyword
    for keyword in keywords:
        # Tokenize the keyword itself to see what it looks like to BERT
        # (add_special_tokens=False because keywords don't need [CLS]/[SEP])
        kw_tokens = tokenizer.tokenize(keyword)
        kw_len = len(kw_tokens)
        
        if kw_len == 0: continue
            
        # Scan the sentence tokens to find a match
        # We start at index 1 to skip [CLS] and end before [SEP]
        for i in range(1, len(tokens) - kw_len):
            # Create a slice of the sentence tokens to compare
            # We compare the actual token strings (e.g., ['bra', 'strap'])
            sentence_slice = tokens[i : i + kw_len]
            
            if sentence_slice == kw_tokens:
                # MATCH FOUND!
                # 1. Tag the first token as 1 (B-ASP)
                labels[i] = 1
                
                # 2. Tag the rest as 2 (I-ASP)
                for j in range(1, kw_len):
                    labels[i + j] = 2
                    
                # Optimization: In a real script, you might want to mark these as 'seen' 
                # to prevent overlapping tags, but for 1500 simple examples this is fine.

    # Only return if we actually found an aspect (sum > 0)
    if sum(labels) > 0:
        return {
            "tokens": tokens,
            "input_ids": input_ids,
            "ner_tags": labels # This is your 0, 1, 2 list
        }
    return None

# 3. Main Loop
input_file = r"O:\Dissertation\renttherunway_final_data.json" #Dataset file path
output_data = []
target_count = 1500

print("Starting Data Generation...")

# Open the big dataset and read line by line
with open(input_file, 'r') as f:
    for line in f:
        if len(output_data) >= target_count:
            break
            
        data = json.loads(line)
        text = data.get("review_text", "")
        
        # Skip empty reviews
        if not text:
            continue
            
        # Attempt to create labels
        processed_example = create_bio_labels(text, aspect_keywords)
        
        # If we found keywords, add to our new dataset
        if processed_example:
            output_data.append(processed_example)

# 4. Save the new "Silver" Dataset
output_file = "absa_training_data_1500.json"
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Success! Generated {len(output_data)} labeled examples.")
print(f"Saved to {output_file}")

# --- Show the User what one entry looks like ---
if len(output_data) > 0:
    example = output_data[0]
    print("\n--- Example Data Point ---")
    print(f"Tokens: {example['tokens']}")
    print(f"Labels: {example['ner_tags']}")
    # You will see: [0, 0, 1, 0, ...] where 1 is the aspect