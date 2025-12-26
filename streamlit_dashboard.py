import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import time
import streamlit.components.v1 as components
import plotly.graph_objects as go
from src.data_utils import (
    load_preprocess_data,
    get_companies,
    get_company_stats,
    max_price,
    min_price,
    avg_price,
    total_volume
)

st.set_page_config(
    page_title="Equity - Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_data
def load_data():
    return load_preprocess_data('stocks.csv')

df = load_data()

st.title("📈 Equity - Top Stock Analysis Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Companies", len(get_companies(df)))

with col2:
    st.metric("Total Records", f"{len(df):,}")

with col3:
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    date_range = f"{min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}"
    st.metric("Date Range", date_range)

with col4:
    st.metric("Avg Volume", f"{df['vol_'].mean():,.2f}")

st.markdown("---")

col_left, col_right = st.columns([2,1])

with col_left:
    st.header("Stock Price Trends")

    companies = get_companies(df)
    selected_companies = st.multiselect(
        "Select Companies to display",
        options=companies,
        default=companies[:5] if len(companies) >= 5 else companies[:1]
    )

    if selected_companies:
        filtered_df = df[df['name'].isin(selected_companies)]

        fig_line = px.line(
            filtered_df,
            x='timestamp',
            y='last',
            color='name',
            title='Stock Prices Over Time',
            labels={'last': 'Stock Price', 'timestamp': 'Date', 'name': 'Company'}
        )
        fig_line.update_layout(hovermode='x unified')
        st.plotly_chart(fig_line, use_container_width=True)

        fig_volume = px.bar(
            filtered_df,
            x='timestamp',
            y='vol_',
            color='name',
            title='Trading Volume Over Time',
            labels={'vol_': 'Volume', 'timestamp': 'Date', 'name': 'Company'},
            height = 300
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    else:
        st.info("Select at least one company to see the charts")

with col_right:
    st.header("Company Statistics")

    selected_company = st.selectbox(
        "Select a company for detailed stats:",
        options=companies
    )
    
    if selected_company:
        max_p, min_p, avg_p, total_vol = get_company_stats(df, selected_company)
        
        st.subheader(f"{selected_company}")
        
        st.metric("Maximum Price", f"${max_p:,.2f}")
        st.metric("Minimum Price", f"${min_p:,.2f}")
        st.metric("Average Price", f"${avg_p:,.2f}")
        st.metric("Total Volume", f"{total_vol:,.0f}")
        
        company_data = df[df['name'] == selected_company]
        fig_dist = px.histogram(
            company_data,
            x='last',
            title=f'{selected_company} Price Distribution',
            labels={'last': 'Price ($)'},
            height=250
        )
        st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

st.header("Company Comparison")

compare_companies = st.multiselect(
    "Select companies to compare side-by-side:",
    options=companies,
    default=companies[:3] if len(companies) >= 3 else companies[:2]
)

if compare_companies:
    comparison_data = []
    for company in compare_companies:
        max_p, min_p, avg_p, total_vol = get_company_stats(df, company)
        comparison_data.append({
            'Company': company,
            'Max Price ($)': f"${max_p:,.2f}",
            'Min Price ($)': f"${min_p:,.2f}",
            'Avg Price ($)': f"${avg_p:,.2f}",
            'Total Volume': f"{total_vol:,.0f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    comparison_chart_data = []
    for company in compare_companies:
        max_p, min_p, avg_p, total_vol = get_company_stats(df, company)
        comparison_chart_data.append({
            'Company': company,
            'Max Price': max_p,
            'Min Price': min_p,
            'Avg Price': avg_p
        })
    
    comparison_chart_df = pd.DataFrame(comparison_chart_data)
    
    fig_comparison = px.bar(
        comparison_chart_df,
        x='Company',
        y=['Max Price', 'Min Price', 'Avg Price'],
        title='Price Comparison Across Companies',
        labels={'value': 'Price ($)', 'variable': 'Metric'},
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
else:
    st.info("Select companies to see comparison")



# Chatbot integration
@st.cache_resource
def start_gradio():
    proc = subprocess.Popen(
        ["python", "gradio_chat.py"])
    time.sleep(3)
    return proc

chatbot_process = start_gradio()

gradio_url = "http://127.0.0.1:7860"
iframe_html = f"""
<style>
    .gradio-container {{
        border: none !important;
    }}
</style>
<div style="width: 100%; height: 650px; border-radius: 12px; overflow: hidden; border: none;">
    <iframe 
        src="{gradio_url}" 
        width="100%" 
        height="650px" 
        frameborder="0"
        style="border: none; border-radius: 12px;">
    </iframe>
</div>
"""

components.html(iframe_html, height=670)

st.markdown("---")
st.caption("Equity Dashboard - Built with Streamlit, Plotly, and Gradio")