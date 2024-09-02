import asyncio
import logging
import uuid

import pymupdf
from fastapi import FastAPI
from langchain_community.llms import Ollama
from summarized_text import summarize_text

import gradio as gr

llm_tasks = {}
summary_tasks = {}

logging.basicConfig(level=logging.INFO)

app = FastAPI()


async def process_pdf_and_summarize(file_content, session_id):
    logging.info(f"Starting PDF processing for session {session_id}...")

    if not (
        file_content and getattr(file_content, "name", "").lower().endswith(".pdf")
    ):
        logging.error("Uploaded file is not a PDF.")
        return "Error: The uploaded file is not a PDF or no file was uploaded.", None

    else:
        pdf_document = pymupdf.open(file_content, filetype="pdf")
        text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text += page.get_text("text")
        pdf_document.close()

        try:
            loop = asyncio.get_running_loop()
            summary_task = loop.run_in_executor(None, summarize_text, text)
            summary_tasks[session_id] = summary_task
            output_text = await summary_task
            logging.info("Summarization completed.")
            return output_text, text
        except asyncio.CancelledError:
            logging.warning(f"Summarization was cancelled for session {session_id}.")
            return "Summarization processing was stopped by the user.", None
        finally:
            summary_tasks.pop(session_id, None)


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

    post_text += "skip introduction about audience, do not greet audience, finish with call to action"

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
        return "Summarization is still in progress. Please wait and try again."

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
        model="llama3",
        # TODO: Remove hardcoded url
        base_url="http://ollama:11434",
    )

    try:
        loop = asyncio.get_running_loop()
        llm_task = loop.run_in_executor(None, llm.invoke, post_text)
        llm_tasks[session_id] = llm_task
        generated_post = await llm_task
        return generated_post
    except asyncio.CancelledError:
        logging.warning(f"LLM processing was cancelled for session {session_id}.")
        return "LLM processing was stopped by the user."
    finally:
        llm_tasks.pop(session_id, None)


def stop_llm(session_id):
    global llm_tasks
    task = llm_tasks.get(session_id)
    if task is not None:
        task.cancel()
        return "LLM processing has been stopped."
    return "No LLM processing is currently running."


async def reset_summarization(session_id):
    global summary_tasks
    logging.info(summary_tasks)
    task = summary_tasks.get(session_id)
    if task is not None:
        task.cancel()
        logging.info(f"Task for session {session_id} was cancelled.")
        return "Summarization process was stopped by the user."
    else:
        logging.info(
            f"No summarization process was found running for session {session_id}."
        )
        return "No active summarization process to stop."


async def summarize_and_store(file_content, session_id):
    summary, raw_text = await process_pdf_and_summarize(file_content, session_id)
    return summary, raw_text


with gr.Blocks() as demo:
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

    pdf_input = gr.File(label="Upload your PDF Document")
    summary_output = gr.Textbox(label="Summary of the PDF", lines=6)
    summary_state = gr.State()
    reset_button = gr.Button("Reset")

    audience_input = gr.Dropdown(
        [
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

    click_event = pdf_input.change(
        summarize_and_store,
        inputs=[pdf_input, session_id],
        outputs=[summary_output, summary_state],
        queue=True,
    )

    submit_button.click(
        fn=generate_post_async,
        inputs=[
            summary_state,
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
    )

    reset_button.click(
        fn=lambda session_id: asyncio.run(reset_summarization(session_id)),
        inputs=[session_id],
        outputs=[summary_output],
        queue=False,
        cancels=[click_event],
    )

    stop_button = gr.Button("Stop")

app = gr.mount_gradio_app(app, demo, path="/")

# Run the interface
if __name__ == "__main__":
    demo.launch(allowed_paths=["./"])
