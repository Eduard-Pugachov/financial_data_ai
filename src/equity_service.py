from src.chatbot_model import EquityChatbot
from src.data_utils import (
    max_price, min_price, avg_price, total_volume, load_preprocess_data
)
from src.chatbot_model import tokenize, bag_of_words
import pandas as pd
import torch
import numpy as np

df = load_preprocess_data('stocks.csv')
bot = EquityChatbot()

def handle_max(company):
    value = max_price(company, df)
    return f"The maximum price for {company} is {value:.2f}"

def handle_min(company):
    value = min_price(company, df)
    return f"The minimum price for {company} is {value:.2f}"

def handle_avg(company):
    value = avg_price(company, df)
    return f"The average price for {company} is {value:.2f}"

def handle_volume(company):
    value = total_volume(company, df)
    return f"The total volume for {company} is {value:.2f}"

intent_handlers = {
    "maximum": handle_max,
    "minimum": handle_min,
    "average": handle_avg,
    "volume": handle_volume
}

def infer_company(user_message):
    companies = df["name"].unique().tolist()
    return next((c for c in companies if c.lower() in user_message.lower()), None)

def basic_single_response(user_message):
    tag, confidence = bot.predict_intent(user_message)
    company = infer_company(user_message)

    if confidence < 0.75 or company is None:
        return "I'm sorry, I couldn't understand your request. Please specify a valid company and query."
    
    handler = intent_handlers.get(tag)
    if handler:
        return handler(company)
    else:
        return "I'm sorry, I couldn't process your request."
    
def predict_intent(self, user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, self.all_words)
    X = torch.from_numpy(np.array(X)).float().view(1, -1)
    
    output = self.model(X)
    probs = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    
    tag = self.tags[predicted.item()]
    return tag, confidence.item()