import gradio as gr
from src.equity_service import basic_single_response
def chat_fn(message, history):
    reply = basic_single_response(message)
    return reply

demo = gr.ChatInterface(
    fn = chat_fn,
    title = "Equity - Stock Analysis Assistant",
    description="Ask about max/min/avg prices for companies such as Microsoft, Apple, NVidia"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share= False)

