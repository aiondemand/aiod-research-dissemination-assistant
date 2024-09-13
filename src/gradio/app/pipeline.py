import asyncio
import logging

import pymupdf
from fastapi import HTTPException
from langchain_community.llms import Ollama
from starlette import status

from .settings import settings
from .summarized_text import summarize_text

llm_tasks = {}
summary_tasks = {}


def validate_pdf_file(file_content):
    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file was uploaded."
        )
    if not getattr(file_content, "name", "").lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file is not a PDF.",
        )


def extract_text_from_pdf(file_content):
    try:
        pdf_document = pymupdf.open(file_content, filetype="pdf")
        text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text += page.get_text("text")
        pdf_document.close()
        return text
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing the PDF file: {str(e)}",
        )


async def process_pdf_and_summarize(file_content, session_id) -> str:
    logging.info(f"Starting PDF processing for session {session_id}...")

    validate_pdf_file(file_content)  # validation input

    text = extract_text_from_pdf(file_content)  # extract text from pdf

    try:
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, summarize_text, text)
        summary_tasks[session_id] = task
        output_text = await task
        logging.info("Summarization completed.")
        return output_text
    except Exception as e:
        logging.error(f"Failed during processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {str(e)}",
        )


def prepare_post_text(
    summary,
    audience,
    english_level,
    length,
    hashtag_preference,
    perspective,
    emoji_usage,
    question_option,
    paper_url,
    custom_requirements,
):
    post_text = (
        "Can you create a LinkedIn post, summarize the text: "
        + summary
        + "; In the post use these parameters:"
    )

    if english_level:
        post_text += f" Use {english_level} English."
    if length:
        post_text += f" Length of post: {length}."
    if emoji_usage:
        post_text += f" {emoji_usage}."
    if audience:
        post_text += f" Targeted at {audience}."
    if hashtag_preference:
        post_text += f" {hashtag_preference}."
    if perspective:
        post_text += f" Written from a {perspective} perspective."
    if question_option:
        post_text += f" {question_option}"
    if paper_url:
        post_text += f" URL to original paper: {paper_url}"
    if custom_requirements:
        post_text += f" Custom requirements: {custom_requirements}"

    post_text += "\n Skip introduction about audience, do not greet audience, finish with call to action"

    return post_text


async def generate_post_async(
    summary,
    audience,
    english_level,
    length,
    hashtag_preference,
    perspective,
    emoji_usage,
    question_option,
    paper_url,
    custom_requirements,
    session_id,
):
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Summarization not completed",
        )
    else:
        post_text = prepare_post_text(
            summary,
            audience,
            english_level,
            length,
            hashtag_preference,
            perspective,
            emoji_usage,
            question_option,
            paper_url,
            custom_requirements,
        )

        llm = Ollama(
            model=settings.generation_ollama_model,
            base_url=settings.ollama_url,
        )

        try:
            loop = asyncio.get_running_loop()
            llm_task = loop.run_in_executor(None, llm.invoke, post_text)
            llm_tasks[session_id] = llm_task
            generated_post = await llm_task
            return generated_post
        except Exception as e:
            logging.error(f"Failed post generation process: {str(e)}.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed post generation process.",
            )


def stop_llm(session_id):
    task = llm_tasks.get(session_id)

    if task is not None:
        llm_tasks.pop(session_id, None)
        logging.info(
            f"The post generation process for session {session_id} was cancelled."
        )
    else:
        logging.info(
            f"No post generation process was found running for session {session_id}."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No post generation running"
        )


def reset_summarization(session_id):
    task = summary_tasks.get(session_id)

    if task is not None:
        summary_tasks.pop(session_id, None)
        logging.info(f"Summarization task for session {session_id} was cancelled.")
    else:
        logging.info(
            f"No summarization process was found running for session {session_id}."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No summarization running"
        )
