import uuid

import uvicorn
from fastapi import FastAPI

import gradio as gr

from .utils import (
    detailed_feedback,
    generate_post_async,
    logging,
    process_pdf_and_summarize,
    reset_summarization,
    simple_feedback,
    stop_llm,
)

logging.basicConfig(level=logging.INFO)

app = FastAPI()

js = """() => {
    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
}"""

with gr.Blocks(
    title="Research dissemination assistant", theme=gr.themes.Default(), js=js
) as demo:
    session_id = gr.State(lambda: uuid.uuid4().hex)

    gr.HTML(
        """
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <img src='file=Quickrepost_logo.png' style='height: 150px; width: auto; alt="QuickRePost_logo"; margin-right: 20px;'/>
            <img src='file=Main_logo_RGB_colors.png' style='height: 150px; width: auto; alt="AI4EUROPE_logo"; margin-left: 20px;'/>
        </div>
        """
    )

    gr.HTML(
        """
        <p style="color: #4a4a4a;">
            Generate a social media post quickly from your research paper with the Quick Research Post tool.
        </p>
    """
    )

    first_part = gr.Group()
    second_part = gr.Group()
    feedback_part = gr.Group()

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
            ["", "Beginner", "Intermediate"], label="Select the English level:"
        )
        length_input = gr.Dropdown(
            ["", "Long", "Short", "Very short"], label="Select the length of the post:"
        )
        hashtag_input = gr.Dropdown(
            [
                "",
                "No use hashtags",
                "Use hashtags",
                "Use 3 hashtags that are already existing, popular and relevant",
            ],
            label="Hashtag usage:",
        )
        perspective_input = gr.Dropdown(
            ["", "First person singular", "First person plural", "Second person"],
            label="Choose the narrative perspective:",
        )
        emoji_input = gr.Dropdown(
            [
                "",
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
        post_output = gr.Markdown(label="Generated LinkedIn Post")

    with feedback_part:
        gr.Markdown("## Feedback")
        with gr.Row():
            like_button = gr.Button("👍")
            dislike_button = gr.Button("👎")

        feedback_text = gr.Textbox(
            visible=False,
            label="Write your detailed feedback here:",
            lines=4,
            interactive=True,
        )
        submit_feedback_button = gr.Button(
            "Submit Feedback", visible=False, interactive=True
        )

        dark_mode_btn = gr.Button("Dark Mode", variant="primary", size="sm")

    like_button.click(
        fn=lambda session_id: simple_feedback("like", session_id, gr),
        inputs=[session_id],
        outputs=[feedback_text, submit_feedback_button],
    )

    dislike_button.click(
        fn=lambda session_id: simple_feedback("dislike", session_id, gr),
        inputs=[session_id],
        outputs=[feedback_text, submit_feedback_button],
    )

    submit_feedback_button.click(
        fn=lambda feedback_text, session_id: detailed_feedback(
            feedback_text, session_id, gr
        ),
        inputs=[feedback_text, session_id],
        outputs=[],
    )

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
