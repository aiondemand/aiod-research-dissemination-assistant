import logging
import uuid

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import RedirectResponse
from keycloak import KeycloakAuthenticationError
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles

import gradio as gr

from .auth import get_user, keycloak_openid
from .pipeline import (
    generate_post_async,
    process_pdf_and_summarize,
    reset_summarization,
    stop_llm,
)
from .settings import settings

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    SessionMiddleware, secret_key=settings.aiod_keycloak.CLIENT_SECRET  # noqa
)


@app.get("/")
def home(user: dict = Depends(get_user)):
    if user:
        return RedirectResponse(url="/app")
    else:
        return RedirectResponse(url="/login-page")


@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth")
    auth_url = keycloak_openid.auth_url(redirect_uri)
    return RedirectResponse(url=auth_url)


@app.get("/auth")
async def auth(request: Request):
    code = request.query_params.get("code")
    redirect_uri = request.url_for("auth")

    try:
        token = keycloak_openid.token(
            grant_type="authorization_code",
            code=code,
            redirect_uri=redirect_uri,
        )
    except KeycloakAuthenticationError:
        return RedirectResponse(url=request.url_for("home"))

    request.session["refresh_token"] = token.get("refresh_token")
    request.session["user"] = keycloak_openid.introspect(token.get("access_token"))

    return RedirectResponse(url="/app")


@app.get("/logout")
async def logout(request: Request):
    keycloak_openid.logout(request.session["refresh_token"])
    request.session.pop("user", None)

    return RedirectResponse(url=request.url_for("home"))


with gr.Blocks() as login_demo:
    gr.HTML(
        """
        <div style="display: flex; align-items: center;">
            <img src='/static/Main_logo_RGB_colors.png' style='height: 100px; width: auto; alt='AI4EUROPE_logo'; margin-right: 20px;'/>
            <h1 style="margin: 0; font-size: 24px;">QuickRePost</h1>
        </div>
        """
    )
    gr.Button("Login", link="/login")


with gr.Blocks(title="Research dissemination assistant") as demo:
    session_id = gr.State(lambda: uuid.uuid4().hex)

    gr.HTML(
        """
        <div style="display: flex; align-items: center;">
            <img src='/static/Main_logo_RGB_colors.png' style='height: 100px; width: auto; alt='AI4EUROPE_logo'; margin-right: 20px;'/>
            <h1 style="margin: 0; font-size: 24px;">QuickRePost</h1>
        </div>
        """
    )
    auth_part = gr.Group()
    first_part = gr.Group()
    second_part = gr.Group()

    with auth_part:

        def greet(request: gr.Request):
            return f"## Welcome to QuickRePost, {request.username}"

        m = gr.Markdown("## Welcome to QuickRePost")
        gr.Button("Logout", link="/logout")
        demo.load(greet, None, m)

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
        fn=stop_llm,
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


app = gr.mount_gradio_app(app, login_demo, path="/login-page", root_path="/login-page")
app = gr.mount_gradio_app(
    app, demo, path="/app", auth_dependency=get_user, root_path="/app"
)

# Run the interface
if __name__ == "__main__":
    uvicorn.run(app, port=7860, log_level="debug")
