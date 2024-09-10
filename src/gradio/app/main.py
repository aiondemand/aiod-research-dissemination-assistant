import asyncio
import logging
import uuid

import pymupdf
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_community.llms import Ollama

import gradio as gr

from .settings import settings
from .summarized_text import summarize_text

llm_tasks = {}
summary_tasks = {}

logging.basicConfig(level=logging.INFO)

app = FastAPI()


def validate_pdf_file(file_content):
    if not file_content:
        raise HTTPException(status_code=400, detail="No file was uploaded.")
    if not getattr(file_content, "name", "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF.")


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
            status_code=500, detail=f"Error processing the PDF file: {str(e)}"
        )


async def process_pdf_and_summarize(file_content, session_id) -> str:
    global summary_tasks
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
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    except asyncio.CancelledError:
        logging.warning(f"Summarization was cancelled for session {session_id}.")
        raise HTTPException(
            status_code=400, detail="Summarization was cancelled"
        )  # gr.info()


# delete input from UI
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
    global llm_tasks

    if not summary:
        raise HTTPException(status_code=400, detail="Summarization not completed")
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
            logging.error(f"Failed during processing: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process PDF: {str(e)}"
            )
        except asyncio.CancelledError:
            logging.warning(f"Summarization was cancelled for session {session_id}.")
            raise HTTPException(status_code=400, detail="Summarization was cancelled")


def stop_llm(session_id):
    global llm_tasks
    task = llm_tasks.get(session_id)
    if task is not None:
        logging.info(f"Summarization task for session {session_id} was cancelled.")
        summary_tasks.pop(session_id, None)
        raise HTTPException(status_code=400, detail="Summarization was cancelled")
    else:
        logging.info(
            f"No summarization process was found running for session {session_id}."
        )
        raise HTTPException(status_code=400, detail="No summarization running")


async def reset_summarization(session_id):
    global summary_tasks
    logging.info(summary_tasks)
    task = summary_tasks.get(session_id)

    if task is not None:
        logging.info(f"Summarization task for session {session_id} was cancelled.")
        summary_tasks.pop(session_id, None)
        raise HTTPException(status_code=400, detail="Summarization was cancelled")
    else:
        logging.info(
            f"No summarization process was found running for session {session_id}."
        )
        raise HTTPException(status_code=400, detail="No summarization running")


with gr.Blocks(title="Research dissemination assistant") as demo:
    session_id = gr.State(lambda: uuid.uuid4().hex)
    logging.info(session_id)

    gr.HTML(
        """
        <div style="display: flex; align-items: center;">
            <img src='file=Main_logo_RGB_colors.png' style='height: 100px; width: auto; alt='AI4EUROPE_logo'; margin-right: 20px;'/>
            <h1 style="margin: 0; font-size: 24px;">Generate social media post</h1>
        </div>
        """
    )
    first_part = gr.Group()
    second_part = gr.Group()

    with first_part:
        gr.Markdown("## Document Summarization")
        pdf_input = gr.File(label="Upload your PDF Document")
        summary_output = gr.Textbox(label="Summary of the PDF", lines=6)
        start_summarization_button = gr.Button("Start summarization")
        reset_button = gr.Button("Reset")

    with second_part:
        gr.Markdown("## Social Media Post Panel")

        audience_input = gr.Dropdown(
            [
                "",
                "High school student",
                "University teacher",
                "Researcher",
                "Business executive",
                "IT professional",
                "Non-profit organization member",
                "General public",
            ],
            label="Select your audience:",
        )
        english_level_input = gr.Dropdown(
            ["Beginner", "Intermediate"], label="Select the English level:"
        )
        length_input = gr.Dropdown(
            ["Long", "Short", "Very short"], label="Select the length of the post:"
        )
        hashtag_input = gr.Dropdown(
            [
                "No use hashtags",
                "Use hashtags",
                "Use 3 hashtags that are already existing, popular and relevant",
            ],
            label="Hashtag usage:",
        )
        perspective_input = gr.Dropdown(
            ["First person singular", "First person plural", "Second person"],
            label="Choose the narrative perspective:",
        )
        emoji_input = gr.Dropdown(
            [
                "Use emoticons",
                "Use emoticons instead of bullet points",
                "Do not use emoticons",
            ],
            label="Emoji usage:",
        )
        url_input = gr.Textbox(label="Paste the URL to your paper here:")
        question_input = gr.Checkbox(label="Begin with Thought-Provoking Question.")
        custom_requirements_input = gr.Textbox(
            label="Enter your custom requirements here:", lines=4
        )

        submit_button = gr.Button("Generate post")
        stop_button = gr.Button("Stop")
        post_output = gr.Textbox(label="Generated LinkedIn Post", lines=8)

    click_event = start_summarization_button.click(
        fn=process_pdf_and_summarize,
        inputs=[pdf_input, session_id],
        outputs=[summary_output],
        queue=True,
    )

    click_event_post = submit_button.click(
        fn=generate_post_async,
        inputs=[
            summary_output,
            audience_input,
            english_level_input,
            length_input,
            hashtag_input,
            perspective_input,
            emoji_input,
            question_input,
            url_input,
            custom_requirements_input,
            session_id,
        ],
        outputs=post_output,
        api_name="generate_post",
        queue=True,
    )

    stop_button.click(
        fn=lambda session_id: stop_llm(session_id),
        inputs=[session_id],
        outputs=post_output,
        queue=False,
        cancels=[click_event_post],
    )

    reset_button.click(
        fn=reset_summarization,
        inputs=[session_id],
        outputs=[summary_output],
        queue=False,
        cancels=[click_event],
    )

    pdf_input.clear(
        fn=reset_summarization,
        inputs=[session_id],
        outputs=[summary_output],
        queue=False,
        cancels=[click_event],
    )

app = gr.mount_gradio_app(app, demo, path="/", allowed_paths=["./"])

# Run the interface
if __name__ == "__main__":
    uvicorn.run(app, port=7860, log_level="debug")
