import gradio as gr
import pymupdf
from langchain_community.llms import Ollama




def process_pdf_and_summarize(file_content):
    # Check if the uploaded file is a PDF
    if not file_content.name.lower().endswith('.pdf'):
        return "Error: The uploaded file is not a PDF.", None

    try:
        # Attempt to open and process the PDF
        pdf_document = pymupdf.open(file_content, filetype="pdf")
        text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text += page.get_text("text")
        pdf_document.close()

        from summarized_text import summarize_text
        return summarize_text(text), text

    except Exception as e:
        return f"Error processing PDF: {str(e)}", None


def prepare_post_text(summary, audience, english_level, tone, length, hashtag_preference, perspective, emoji_usage,
                      question_option, paper_url, custom_requirements):
    post_text = "Can you create a LinkedIn post, summarize the text: " + summary + "; In the post use these parameters:"

    if tone:
        post_text += f" Use {tone} tone."
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


def generate_post(summary, audience, english_level, tone, length, hashtag_preference,
                              perspective, emoji_usage, question_option, paper_url, custom_requirements):

    if not summary:
        return "Summarization is still in progress. Please wait and try again."

    post_text = prepare_post_text(summary, audience, english_level, tone, length, hashtag_preference, perspective,
                                  emoji_usage, question_option, paper_url, custom_requirements)

    llm = Ollama(model="llama3")
    generated_post = llm.invoke(post_text)

    return generated_post


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
        ["No use hashtags", "Use hashtags", "Use 3 hashtags that are already existing, popular and relevant"],
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
    post_output = gr.Textbox(label="Generated LinkedIn Post", lines=8)


    def summarize_and_store(file_content):
        summary, raw_text = process_pdf_and_summarize(file_content)
        return summary, raw_text


    # Trigger summarization immediately after PDF upload
    pdf_input.change(
        summarize_and_store,
        inputs=pdf_input,
        outputs=[summary_output, summary_state]
    )

    # Generate the post only when the submit button is clicked
    submit_button.click(
        generate_post,
        inputs=[
            summary_state, audience_input, english_level_input, length_input,
            hashtag_input, perspective_input, emoji_input, question_input, url_input, custom_requirements_input
        ],
        outputs=post_output
    )

# Run the interface
if __name__ == "__main__":
    demo.launch(allowed_paths=["./"])
