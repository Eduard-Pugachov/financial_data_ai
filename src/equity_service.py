from src.chatbot_model import EquityChatbot
from src.data_utils import (
    max_price, min_price, avg_price, total_volume, load_preprocess_data
)
from src.chatbot_model import tokenize, bag_of_words
import pandas as pd
import torch
import numpy as np
from typing import Dict, Optional, Callable
from datetime import datetime
class EquityService:
    def __init__(self, data_path: str, model_path: str):
        self.df = self._load_data(data_path)
        self.bot = EquityChatbot(model_path=model_path)
        self.handlers: Dict[str, Callable] = {
            "maximum": self._handle_maximum,
            "minimum": self._handle_minimum,
            "average": self._handle_average,
        }
    def _load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def _extract_entity(self, message: str) -> Optional[str]:
        message_lower = message.lower()
        for company in self.df['name'].unique():
            if company.lower() in message_lower:
                return company
        return None
    
    def _get_company_data(self, company: str) -> pd.DataFrame:
        return self.df[self.df['name'] == company].copy()
    
    def _format_currency(self, value: float) -> str:
        return f"${value:,.2f}"
    
    def _format_date(self, dt: datetime) -> str:
        return dt.strftime('%b %d, %Y')
    
    def _format_percentage(self, value: float) -> str:
        return f"{abs(value):.1f}"
    
    def _format_percentage(self, value: float) -> str:
        return f"{abs(value):.1f}%"
    
    def _handle_maximum(self, company: str) -> str:
        company_df = self._get_company_data(company)
        max_row = company_df.loc[company_df['high'].idxmax()]
        latest_row = company_df.iloc[-1]

        max_price = max_row['high']
        max_date = max_row['timestamp']
        latest_price = latest_row['last']

        diff_pct = ((latest_price - max_price) / max_price) * 100

        if diff_pct >= -1:
            context = f"which is near its current level of {self._format_currency(latest_price)}"
        elif diff_pct >= -10:
            context = f"{self._format_percentage(diff_pct)} below its current price of {self._format_currency(latest_price)}"
        else:
            context = f"significantly above today's {self._format_currency(latest_price)} (down {self._format_percentage(diff_pct)})"
        
        return (f"**{company}** peaked at **{self._format_currency(max_price)}** on {self._format_date(max_date)}, "
                f"{context}.")
    
    def _handle_minimum(self, company: str) -> str:
        company_df = self._get_company_data(company)
        min_row =  company_df.loc[company_df['low'].idxmin()]
        latest_row = company_df.iloc[-1]

        min_price = min_row['low']
        min_date = min_row['timestamp']
        latest_price = latest_row['last']

        recovery_pct = ((latest_price - min_price) / min_price) * 100

        if recovery_pct <= 1:
            context = f"roughly where it still trades at {self._format_currency(latest_price)}"
        elif recovery_pct <= 20:
            context = f"and has since recovered {self._format_percentage(recovery_pct)} to {self._format_currency(latest_price)}"
        else:
            context = f"and has surged {self._format_percentage(recovery_pct)} to today's {self._format_currency(latest_price)}"
        
        return (f"**{company}** bottomed at **{self._format_currency(min_price)}** on {self._format_date(min_date)}, "
                f"{context}.")
    
    def _handle_average(self, company: str) -> str:
        company_df = self._get_company_data(company)
        
        avg_price = company_df['last'].mean()
        max_price = company_df['high'].max()
        min_price = company_df['low'].min()
        latest_price = company_df.iloc[-1]['last']
        
        price_range = max_price - min_price
        range_pct = (price_range / avg_price) * 100
        vs_avg_pct = ((latest_price - avg_price) / avg_price) * 100
        
        if range_pct < 5:
            vol_note = "a tight trading range"
        elif range_pct < 15:
            vol_note = "moderate swings"
        else:
            vol_note = "high volatility"
        
        if abs(vs_avg_pct) < 2:
            position = f"Currently trading right at the average ({self._format_currency(latest_price)})"
        elif vs_avg_pct > 0:
            position = f"Currently {self._format_percentage(vs_avg_pct)} above average at {self._format_currency(latest_price)}"
        else:
            position = f"Currently {self._format_percentage(vs_avg_pct)} below average at {self._format_currency(latest_price)}"
        
        return (f"**{company}** averages **{self._format_currency(avg_price)}** with {vol_note} "
                f"({self._format_currency(min_price)}–{self._format_currency(max_price)}). {position}.")
    
    def process_query(self, message: str) -> str:
        tag, confidence = self.bot.predict_intent(message)
        company = self._extract_entity(message)
        
        if company is None:
            if tag in ["greeting", "goodbye", "thanks", "help"]:
                return self.bot.get_response(message)
            return "Please mention a company name like Microsoft, Apple, NVIDIA, or JPMorgan."
        
        if confidence < 0.75:
            return "I'm not sure what you're asking. Try questions like 'max Microsoft' or 'average Apple'."
        
        handler = self.handlers.get(tag)
        if handler:
            return handler(company)
        
        return self.bot.get_response(message)


_service_instance = None

def get_service() -> EquityService:
    global _service_instance
    if _service_instance is None:
        _service_instance = EquityService(
            data_path='stocks.csv',
            model_path='src/chatbot_model.pth'
        )
    return _service_instance

def basic_single_response(message: str) -> str:
    return get_service().process_query(message)