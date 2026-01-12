# Top Companies' Stock Data AI Chatbot - "Equity"
An interactive dashboard and chatbot assistant (Equity) to analyze and predict the top ~32 company stocks.

## 🎥 Demo Video
Short video with a few of the capabilities shown:
https://youtu.be/__HLwR6N6S0

## Overview

NOTE --- PREDICTION NOT YET IMPLEMENTED

This project combines data analysis, machine learning, and conversational AI to deliver real-time insights on financial markets. It showcases the full ML pipeline, from data preprocessing and model training to interactive web interfaces using Streamlit and Gradio.

## Features

- **AI Chatbot**: Natural language interface for querying stock data
- **Interactive Dashboard**: Real-time visualizations and comparisons of company stock prices
- **Multi-company Analysis**: Comparative analytics across major equities assets

## Tech Stack

- **Languages**: Python, SQL
- **ML Frameworks**: scikit-learn, PyTorch
- **Data Analysis**: pandas, NumPy, matplotlib, seaborn
- **NLP**: NLTK
- **Deployment**: Streamlit, Gradio
- **Environment**: Jupyter Notebook, Kaggle datasets

## Project Structure

```
financial_data_ai/
├── notebooks/          # Exploratory data analysis and model experimentation
├── src/               # Core modules and utilities
├── gradio_chat.py     # Gradio-based chatbot interface
├── streamlit_dashboard.py  # Streamlit visualization dashboard
├── train_chatbot.py   # Model training pipeline
├── stocks.csv         # Stock market dataset
└── cryptocurrency.csv # Cryptocurrency dataset
```

## Key Learning Goals / Motivation

This project was built to understand:
- Architecture and workflow of AI chatbot systems
- Integration of ML models with web frameworks (Streamlit, Gradio)
- End-to-end data science pipeline from raw data to deployment
- Real-world application of NLP and time-series forecasting

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Eduard-Pugachov/financial_data_ai.git
cd financial_data_ai

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard (Gradio chatbot included)
streamlit run streamlit_dashboard.py

```

## Skills Demonstrated

- Machine learning model development and evaluation
- Natural language processing for conversational AI
- Data visualization and dashboard design
- Full-stack Python development
- Version control and project organization

## Future Enhancements

- Integration with real-time market APIs
- Expanded model architectures
- Multi-language support
- Enhanced prediction accuracy with additional features

## License

This project is open source and available for educational and portfolio purposes.
