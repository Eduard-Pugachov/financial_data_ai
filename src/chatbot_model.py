import random
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
import numpy as np


# Download required NLTK data
nltk.download('punkt')

# Text preprocessing (NLP foundation)
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return PorterStemmer().stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = [1 if w in sentence_words else 0 for w in words]
    return bag

# Neural network model architecture
class EquityChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EquityChatbotModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


class EquityChatbot:
    def __init__(self, model_path='chatbot_model.pth'):
        data = torch.load(model_path)

        self.input_size = data['input_size']
        self.hidden_size = data['hidden_size']
        self.output_size = data['output_size']
        self.all_words = data['all_words']
        self.tags = data['tags']
        self.intents = data['intents']

        self.model = EquityChatbotModel(self.input_size,self.hidden_size,self.output_size)
        self.model.load_state_dict(data['model_state'])
        self.model.eval()

    def get_response(self, user_input):
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, self.all_words)
        X = torch.from_numpy(np.array(X)).float()
        X = X.view(1, -1)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim = 1)
        confidence = probs[0][predicted.item()]

        if confidence > 0.75:
            for intent in self.intents['intents']:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])
        
        return "I'm sorry, I am not sure I understand. Can you rephrase that?"

