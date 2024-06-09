from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("./results")


def generate_summary(
    model, tokenizer, text, max_input_length=512, max_output_length=256
):
    # Tokenize đầu vào
    inputs = tokenizer(
        text,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Tạo tóm tắt
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True,
    )

    # Giải mã tóm tắt thành chuỗi văn bản
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


import gradio as gr


def get_summary(initial_text, max_iterations=25):
    global model, tokenizer
    return generate_summary(model, tokenizer, initial_text)


# Create Gradio interface
iface = gr.Interface(
    fn=get_summary,
    inputs=gr.Textbox(lines=2, label="Enter the text you want to summarize:"),
    outputs=gr.Textbox(label="Summarized text:"),
    title="Simple Summarizer",
    description="Enter a text and get a summary.",
)

# Launch the interface
iface.launch()
