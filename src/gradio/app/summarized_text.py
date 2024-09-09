import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from .settings import settings

tokenizer = AutoTokenizer.from_pretrained(settings.summarization_model)
model = AutoModelForSeq2SeqLM.from_pretrained(settings.summarization_model)


def chunk_text(text, max_tokens, overlap=50):
    words = text.split()
    chunks = []
    current_chunk = []
    current_count = 0

    for index, word in enumerate(words):
        word_tokens = tokenizer.tokenize(word)
        word_token_count = len(word_tokens)

        if current_count + word_token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = words[max(index - overlap, 0) : index - 1]
            current_count = len(tokenizer.tokenize(" ".join(current_chunk)))

        else:
            current_chunk.append(word)
            current_count += word_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize_text(text, max_tokens=500, overlap=50):
    chunks = chunk_text(text, max_tokens, overlap)

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    summaries = summarizer(
        chunks,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return "/n".join(summary["summary_text"] for summary in summaries)
