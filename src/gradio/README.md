# Gradio-based Social Media Post Generator

This project is a Gradio application that generates social media posts based on PDF content. It uses a summarization model to extract key points from the uploaded PDF and then generates a custom LinkedIn post based on user-selected parameters.

## Features

- **PDF Summarization**: Upload a PDF document, and the application will extract and summarize the content.
- **Custom Post Generation**: Customize the generated post by selecting parameters such as audience, tone, length, perspective, and more.

## Requirements

- Python 3.7+
- The following Python libraries:
  - `gradio`
  - `transformers`
  - `torch`
  - `pymupdf`
  - `langchain_community`

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/aiondemand/aiod-research-dissemination-assistant.git
    cd aiod-research-dissemination-assistant/src/gradio
    ```

2. **Install the required Python packages**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:

    ```bash
    python app.py
    ```

4. **Access the application**:

   Open your browser and go to `http://localhost:7860` to use the Gradio app.

## Usage

1. **Upload PDF**: Click on the "Upload your PDF Document" button to upload a PDF file.
2. **Select Parameters**: Choose the appropriate options for audience, tone, length, etc., to customize the generated LinkedIn post.
3. **Generate Post**: Click on the "Generate post" button to create the LinkedIn post based on the PDF content.

## How It Works

- **PDF Processing**: The application uses `pymupdf` to extract text from the uploaded PDF document.
- **Text Summarization**: The extracted text is then summarized using a pre-trained model from Hugging Face's Transformers library.
- **Post Generation**: The summarized text is formatted into a LinkedIn post using the selected parameters. The post is generated by invoking the `Ollama` language model from LangChain.