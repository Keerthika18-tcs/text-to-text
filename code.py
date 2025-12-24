!pip install transformers torch sentencepiece gradio -q
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
def generate_headline(article_text, max_length=20):
    if not article_text.strip():
        return "⚠️ Please enter some text!"
    input_text = "summarize: " + article_text
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
sample_article = """The Indian Space Research Organisation (ISRO) successfully launched its latest satellite
into orbit today. The satellite aims to improve communication services and provide better
connectivity in rural areas of India. The mission is considered a major milestone
for the country's space program."""
print("Sample Headline:", generate_headline(sample_article))
with gr.Blocks() as demo:
    gr.Markdown("## 📰 AI Headline Generator")
    gr.Markdown("Paste a news article or paragraph, and get a short headline!")

    with gr.Row():
        article_input = gr.Textbox(label="Enter News Article", lines=7, placeholder="Paste your news content here...")
    with gr.Row():
        headline_output = gr.Textbox(label="Generated Headline")

    submit_btn = gr.Button("Generate Headline")
    submit_btn.click(fn=generate_headline, inputs=article_input, outputs=headline_output)

demo.launch()
