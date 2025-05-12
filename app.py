import gradio as gr
from transformers import pipeline

# Load a pre-trained sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)[0]
    return f"{result['label']} ({round(result['score'] * 100, 2)}%)"

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence..."),
    outputs="text",
    title="Sentiment Analysis App",
    description="Enter text to analyze its sentiment using a Hugging Face model."
)

if __name__ == "__main__":
    iface.launch()

