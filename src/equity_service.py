from src.chatbot_model import EquityChatbot
from src.data_utils import (
    max_price, min_price, avg_price, total_volume, load_preprocess_data, parse_volume
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
            "volume_query": self._handle_volume,
            "comparison": self._handle_comparison,
            "growth_query": self._handle_growth

        }
    def _load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['vol_'] = df['vol_'].apply(parse_volume)
        df['chg_%'] = df['chg_%'].str.replace('%', '').str.replace('+', '').astype(float)
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
    
    def _handle_volume(self, company: str) -> str:
        company_df = self._get_company_data(company)

        total_volume = company_df['vol_'].sum()
        avg_volume = company_df['vol_'].mean()
        max_volume = company_df['vol_'].max()
        latest_volume = company_df.iloc[-1]['vol_']
        
        max_vol_row = company_df.loc[company_df['vol_'].idxmax()]
        max_vol_date = max_vol_row['timestamp']
        
        total_million = total_volume / 1_000_000
        avg_million = avg_volume / 1_000_000
        latest_million = latest_volume / 1_000_000
        max_million = max_volume / 1_000_000
        
        volume_ratio = latest_volume / avg_volume
        
        if volume_ratio > 1.5:
            activity = "significantly above average"
        elif volume_ratio > 1.1:
            activity = "moderately elevated"
        elif volume_ratio < 0.7:
            activity = "below average"
        else:
            activity = "near average"
        
        lines = [
            f"{company} Trading Volume",
            f"",
            f"• Total volume: {total_million:.1f}M shares",
            f"• Daily average: {avg_million:.1f}M shares",
            f"• Peak volume: {max_million:.1f}M on {self._format_date(max_vol_date)}",
            f"• Latest activity: {latest_million:.1f}M shares ({activity})"
        ]
        
        return "\n".join(lines)
    
    def _handle_comparison(self, company: str) -> str:
        if company is None:
            return self._compare_all_companies()
        return self._compare_single_company(company)
    
    def _compare_all_companies(self) -> str:
        all_companies = self.df.groupby('name').agg({
            'chg_%': 'last',
            'vol_': 'sum',
            ''
            'last': 'last'
        }).reset_index()

        all_companies = all_companies.sort_values('chg_%', ascending=False)

        top_3 = all_companies.head(3)
        bottom_3 = all_companies.tail(3)

        top_performers = []
        for _, row in top_3.iterrows():
            top_performers.append(f"{row['name']} ({row['chg_%']:+.2f}%)")
    
        bottom_performers = []
        for _, row in bottom_3.iterrows():
            bottom_performers.append(f"{row['name']} ({row['chg_%']:+.2f}%)")
        
        top_names = ", ".join(top_performers)
        bottom_names = ", ".join(bottom_performers)
        
        response = (
            f"Top performers: {top_names}\n\n"
            f"Bottom performers: {bottom_names}"
        )
        
        return response

    def _compare_single_company(self, company: str) -> str:
        company_df = self._get_company_data(company)
        latest_row = company_df.iloc[-1]

        latest_price = latest_row['last']
        latest_chg_pct = latest_row['chg_%']
        avg_price = company_df['last'].mean()
        total_volume = company_df['vol_'].sum()

        all_latest = self.df.groupby('name').last().reset_index()
        total_companies = len(all_latest)
        
        rank_by_price = (all_latest['last'] > latest_price).sum() + 1
        rank_by_change = (all_latest['chg_%'] > latest_chg_pct).sum() + 1
        
        volume_million = total_volume / 1_000_000
        
        response = (
            f"{company} ranks #{rank_by_price} by price "
            f"({self._format_currency(latest_price)}) and #{rank_by_change} by change "
            f"({latest_chg_pct:+.2f}%) among {total_companies} companies. "
            f"Average price is {self._format_currency(avg_price)} with total volume of "
            f"{volume_million:.1f}M shares."
        )
        
        return response

    def _handle_growth(self, company: str) -> str:
        if company is None:
            return self._compare_all_companies()
        
        company_df = self._get_company_data(company)
        latest_row = company_df.iloc[-1]
        first_row = company_df.iloc[0]
        
        start_price = first_row['last']
        end_price = latest_row['last']
        total_growth = ((end_price - start_price) / start_price) * 100
        latest_chg_pct = latest_row['chg_%']
        
        all_latest = self.df.groupby('name').last().reset_index()
        all_latest = all_latest.sort_values('chg_%', ascending=False)
        
        rank = (all_latest['chg_%'] > latest_chg_pct).sum() + 1
        total_companies = len(all_latest)
        
        if total_growth > 10:
            performance = "strong gains"
        elif total_growth > 5:
            performance = "moderate growth"
        elif total_growth > 0:
            performance = "slight gains"
        elif total_growth > -5:
            performance = "slight decline"
        else:
            performance = "significant losses"
        
        lines = [
            f"{company} Growth Analysis",
            f"",
            f"• Period growth: {total_growth:+.2f}%",
            f"• Latest change: {latest_chg_pct:+.2f}%",
            f"• Performance rank: #{rank} of {total_companies} ({performance})",
            f"• Start price: {self._format_currency(start_price)}",
            f"• Current price: {self._format_currency(end_price)}"
        ]
        
        return "\n".join(lines)
    def process_query(self, message: str) -> str:
        tag, confidence = self.bot.predict_intent(message)
        company = self._extract_entity(message)
        
        if company is None and tag in ["comparison", "growth_query"]:
            handler = self.handlers.get(tag)
            if handler:
                return handler(None)
        
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