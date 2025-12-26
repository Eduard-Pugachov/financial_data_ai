import gradio as gr
from src.equity_service import basic_single_response
def chat_fn(message, history):
    reply = basic_single_response(message)
    return reply

theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    body_background_fill="#0E1117",
    body_background_fill_dark="#0E1117",
    
    body_text_color="#FAFAFA",
    body_text_color_dark="#FAFAFA",
    
    block_background_fill="#0E1117",
    block_background_fill_dark="#0E1117",
    
    panel_background_fill="#0E1117",
    panel_background_fill_dark="#0E1117",
    
    input_background_fill="#262626",
    input_background_fill_dark="#262626",
    
    button_primary_background_fill="#1f77b4",
    button_primary_background_fill_dark="#1f77b4",
    
    border_color_primary="#262626",
)

demo = gr.ChatInterface(
    fn = chat_fn,
    title = "Equity - Stock Analysis Assistant",
    description="Ask about max/min/avg prices for companies such as Microsoft, Apple, NVidia",
    theme = theme
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share= False)

