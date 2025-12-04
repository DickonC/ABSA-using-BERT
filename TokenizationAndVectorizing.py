from transformers import AutoTokenizer, AutoModel
import torch

# 1. Load the pre-trained BERT tokenizer and model
# 'bert-base-uncased' is a standard model
#base = 12 hidden layers (stacks of transformers, layer 1 passes to layer 2 et.c),
# 768  hidden units (vector size - not same as hidden layers),
#uncase means converts to lowercase

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Your Input Sentence
text = "The sleeves are way too baggy, and its quite tight around the chest."

# 3. Tokenization - lookup token ids
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 4. Vectorization (Getting the numbers) - also a lookup for the IDs in the vocabulary
with torch.no_grad():  #not training just inference therefore no_grad = inference
    outputs = model(**inputs)
    # The 'last_hidden_state' is the contextual vector for every word
    vectors = outputs.last_hidden_state #passed through layers of attention (contextualized)

# --- Results ---
print(f"Original Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
print(f"Vector Shape: {vectors.shape}")