#Token Classification, we want to turn the token "the, sleeves, are, baggy"
#to "O, B-ASP, O ,O " where O means outside the aspect and B-ASP means
#beginning of aspect, I-ASP = Inside aspect

#we want to add a HEAD (Simple Neural Network) on top of BERT
#the input to the head is the each token vector, this vector is multiplied by a weight matrix
# + a bias and turns the large vector space into a 2-dimentional vector, Score for yes its and aspect or no its not
#these can be turned with softmax into a probability so we get a final vector that looks like this:
# [0.05,0.95] saying 95% its an aspect, 5% its not

''' the head we added to the top is initialized with random noise parameters (Guassian Distribution)
so we need to fine tune this final head (train only/mostly this final head how to interpret the BERT body into our results)'''

from transformers import AutoTokenizer, BertForTokenClassification
import torch

# Load a model with a "Classification Head" on top
# We use 'bert-base-uncased' again.
# num_labels=3 corresponds to our tags: 0=O (Outside), 1=B-ASP (Begin), 2=I-ASP (Inside)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=3)

# Input Data
text = "The sleeves are way too baggy."
#tokenize the text
inputs = tokenizer(text, return_tensors="pt")

#The Forward Pass
with torch.no_grad():
#contextualizes the vectors
    outputs = model(**inputs)

#Extracting the Logits (The raw scores from the Linear Layer)
# Shape will be: [1 (Batch), 9 (Tokens), 3 (Labels (outside, beginning, inside))]
#logits are the raw scores for these without softmaxing
logits = outputs.logits

#Converting to Probabilities (The Softmax function)
#the dim = -1 just meand apply the softmax to the final dimension of logits (the 3 label scores)
predictions = torch.argmax(logits, dim=-1)

# --- Visualizing the Result ---
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"Original Sentence: {text}")
print(f"{'TOKEN':<12} | {'PREDICTION (0=O, 1=B, 2=I)':<25}")
print("-" * 40)

for token, prediction in zip(tokens, predictions[0].tolist()):
    print(f"{token:<12} | {prediction}")

# Since this model is NOT fine-tuned yet, the predictions will be garbage (random).
# I need to TRAIN this model to output '1' for 'sleeves' for example.

'''
once we have this trained model head outputting the aspect properly we will have built the EXTRACTOR

then we move onto the CLASSIFIER, this is a model given the aspect and asked given the context of the sentence and the aspect
"what is the sentiment" -  or in our case what is the "fit" on a scale of -2 to 2 maybe e.g. +2 = very loose

'''