# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:13:29 2019

@author: 群青雨
"""

# Import required libraries
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
text = "How to use the API"
predicted_text = text
while predicted_text[-1] not in {'.','?','!','~','\n'}:  
    text = predicted_text
    indexed_tokens = tokenizer.encode(text)
    
    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    
    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Set the model in evaluation mode to deactivate the DropOut modules
    model.eval()
    
    # If you have a GPU, put everything on cuda
    #tokens_tensor = tokens_tensor.to('cuda')
    #model.to('cuda')
    
    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    
    # Get the predicted next sub-word
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

# Print the predicted word
print(predicted_text)