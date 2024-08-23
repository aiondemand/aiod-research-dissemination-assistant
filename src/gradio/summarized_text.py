import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

reset_flag = None

def chunk_text(text, max_tokens, overlap=50):
    words = text.split()
    chunks = []
    current_chunk = []
    current_count = 0
    last_overlap_index = 0

    for index, word in enumerate(words):
        if reset_flag and reset_flag.is_set():
            return []

        word_tokens = tokenizer.tokenize(word)
        word_token_count = len(word_tokens)

        if current_count + word_token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = words[last_overlap_index:index]
            current_count = len(tokenizer.tokenize(' '.join(current_chunk)))
            last_overlap_index = max(index - overlap, 0)

        else:
            current_chunk.append(word)
            current_count += word_token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize_text(text, max_tokens=512, overlap=50):
    if reset_flag and reset_flag.is_set():
        return "Summarization process interrupted by user."

    chunks = chunk_text(text, max_tokens, overlap)
    if not chunks:
        return "Summarization process interrupted by user."

    summaries = []

    for chunk in chunks:
        if reset_flag and reset_flag.is_set():
            return "Summarization process interrupted by user."

        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=max_tokens)
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return ' '.join(summaries)
