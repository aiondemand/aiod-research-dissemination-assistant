import asyncio

import gradio as gr
import pymupdf
from langchain_community.llms import Ollama

from summarized_text import summarize_text

# This will hold the task for the long-running LLM invoke operation
llm_task = None
summary_task = None

async def process_pdf_and_summarize(file_content):
    global summary_task

    if not file_content or not getattr(file_content, 'name', '').lower().endswith('.pdf'):
        return "Error: The uploaded file is not a PDF or no file was uploaded."

    pdf_document = pymupdf.open(file_content, filetype="pdf")
    text = ""
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        text += page.get_text("text")
    pdf_document.close()

    try:
        loop = asyncio.get_running_loop()
        summary_task = loop.run_in_executor(None, summarize_text, text)
        output_text = await summary_task
        return output_text
    except asyncio.CancelledError:
        return None, "Summarization processing was stopped by the user."
    finally:
        summary_task = None




def prepare_post_text(summary, audience, english_level, length, hashtag_preference, perspective,
                      emoji_usage, question_option, paper_url, custom_requirements):
    post_text = "Can you create a LinkedIn post, summarize the text: " + summary + "; In the post use these parameters:"

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


async def generate_post_async(summary, audience, english_level, length, hashtag_preference,
                              perspective, emoji_usage, question_option, paper_url,
                              custom_requirements):
    global llm_task

    if not summary:
        return "Summarization is still in progress. Please wait and try again."

    post_text = prepare_post_text(summary, audience, english_level, length, hashtag_preference,
                                  perspective, emoji_usage, question_option, paper_url,
                                  custom_requirements)

    llm = Ollama(model="llama3")

    try:
        loop = asyncio.get_running_loop()
        # Execute llm.invoke asynchronously
        llm_task = loop.run_in_executor(None, llm.invoke, post_text)
        generated_post = await llm_task
        return generated_post
    except asyncio.CancelledError:
        return "LLM processing was stopped by the user."
    finally:
        llm_task = None  # Reset the task when done


def stop_llm():
    global llm_task
    if llm_task is not None:
        llm_task.cancel()  # Cancel the task if it is running
        return "LLM processing has been stopped."
    return "No LLM processing is currently running."

def reset_summarization():
    global summary_task
    if summary_task is not None:
        summary_task.cancel()  # Cancel the task if it is running
        return "Summarization process not running.", None, None
    return "Summarization process not running.", None, None


with gr.Blocks() as demo:
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
    summary_state = gr.State()  # To store the summary
    reset_button = gr.Button("Reset")  # Add a reset button

    audience_input = gr.Dropdown(
        ["High school student", "University teacher",
         "Researcher", "Business executive", "IT professional",
         "Non-profit organization member", "General public"],
        label="Select your audience:"
    )
    english_level_input = gr.Dropdown(
        ["Beginner", "Intermediate"],
        label="Select the English level:"
    )
    length_input = gr.Dropdown(
        ["Long", "Short", "Very short"],
        label="Select the length of the post:"
    )
    hashtag_input = gr.Dropdown(
        ["No use hashtags", "Use hashtags",
         "Use 3 hashtags that are already existing, popular and relevant"],
        label="Hashtag usage:"
    )
    perspective_input = gr.Dropdown(
        ["First person singular", "First person plural", "Second person"],
        label="Choose the narrative perspective:"
    )
    emoji_input = gr.Dropdown(
        ["Use emoticons", "Use emoticons instead of bullet points", "Do not use emoticons"],
        label="Emoji usage:"
    )
    url_input = gr.Textbox(label="Paste the URL to your paper here:")
    question_input = gr.Checkbox(label="Begin with Thought-Provoking Question.")
    custom_requirements_input = gr.Textbox(label="Enter your custom requirements here:", lines=4)

    submit_button = gr.Button("Generate post")
    stop_button = gr.Button("Stop")
    post_output = gr.Textbox(label="Generated LinkedIn Post", lines=8)

    async def summarize_and_store(file_content):
        summary = await process_pdf_and_summarize(file_content)
        return summary, None


    # Trigger summarization immediately after PDF upload
    pdf_input.change(
        summarize_and_store,
        inputs=pdf_input,
        outputs=[summary_output, summary_state]
    )

    # Generate the post only when the submit button is clicked
    submit_button.click(
        fn=generate_post_async,
        inputs=[
            summary_state, audience_input, english_level_input, length_input, hashtag_input,
            perspective_input, emoji_input, question_input, url_input, custom_requirements_input
        ],
        outputs=post_output,
        api_name="generate_post"
    )

    stop_button.click(
        fn=stop_llm,
        inputs=None,
        outputs=post_output
    )

    reset_button.click(
        reset_summarization,
        inputs=None,
        outputs=[summary_output, summary_state, pdf_input]
    )

# Run the interface
if __name__ == "__main__":
    demo.launch()
