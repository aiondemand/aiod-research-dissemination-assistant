from transformers import BartTokenizer, BartForConditionalGeneration
import torch


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def chunk_text(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        trial_text = ' '.join(current_chunk + [word])
        token_count = len(tokenizer.tokenize(trial_text))
        if token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def summarize_text(text):
    chunks = chunk_text(text, max_tokens=1024)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return ' '.join(summaries)
