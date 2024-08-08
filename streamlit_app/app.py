import streamlit as st
import pymupdf
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms import Ollama

st. set_page_config(layout="wide")
st.title('Generate LinkedIn Post')

executor = ThreadPoolExecutor(max_workers=1)

def process_pdf_and_summarize(file_content):
    pdf_document = pymupdf.open(stream=file_content, filetype="pdf")
    text = ""
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        text += page.get_text("text")
    pdf_document.close()

    from summarized_text import summarize_text
    
    return summarize_text(text)

def async_process_pdf(file_content):
    if 'processing_future' in st.session_state:
        st.session_state.pop('processing_future')
    future = executor.submit(process_pdf_and_summarize, file_content)
    st.session_state['processing_future'] = future
    st.session_state['processing'] = True

def scroll_to_top():
    js = '''
    <script>
    var body = window.parent.document.querySelector(".main");
    body.scrollTo({
        top: 0,
        left: 0,
        behavior: 'smooth'
    });
    </script>
    '''
    st.components.v1.html(js, height=0)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("## Choose a PDF file of your research article")
    uploaded_file = st.file_uploader("", type="pdf")
    if uploaded_file is not None:
        file_info = uploaded_file.getvalue()
        if 'last_file_info' not in st.session_state or st.session_state['last_file_info'] != file_info:
            st.session_state['last_file_info'] = file_info
            st.session_state.pop('summary', None)
            async_process_pdf(file_info)

        if 'processing' in st.session_state:
            future = st.session_state['processing_future']
            if future.done():
                summary = future.result()
                st.session_state['processing'] = False
                st.session_state['summary'] = summary
                st.success("The summary of the text is completed.")
            else:
                st.info("Processing PDF, please wait...")

with col2:
    st.markdown("## Create your LinkedIn post preferences")
    if uploaded_file is not None:
    
        audience = st.selectbox("Select your audience:", ["Secondary school student", "High school student", "University teacher",
                                                          "Researcher", "Business executive", "IT professional", "Software developer",
                                                          "Entrepreneur", "Journalist", "Non-profit organization member", "General public", "Engineers"])
        english_level = st.selectbox("Select the English level:", ["Beginner", "Intermediate"])
        tone = st.selectbox("Select the tone of the post:", ["Informal", "Inspirational", "Persuasive", "Humorous", "Neutral", "Technical",
                                                             "Empathetic", "Authoritative", "Friendly", "Confident", "Playful", "Personal",
                                                             "Concise", "Relatable", "Captivating", "Enthusiastic", "Optimistic", "Respectful", "Engaging"])
        length = st.selectbox("Select the length of the post:", ["Long", "Short", "Very short"])
        hashtag_preference = st.selectbox("Hashtag usage:", ["No use hashtags", "Use hashtags", "Use 3 hashtags that are already existing, popular and relevant"])
        perspective = st.selectbox("Choose the narrative perspective:", ["First person singular", "First person plural", "Second person", "Third person objective"])
        emoji_usage = st.selectbox("Emoji usage:", ["Use emoticons", "Do not use emoticons"])
        
        st.write("Choose additional settings:")
        option_1 = "Begin with Thought-Provoking Question." if st.checkbox("Begin with Thought-Provoking Question.") else ""

        
        if st.button("Generate Post"):
            if 'summary' in st.session_state and not st.session_state['processing']:
                post_text = f"""
                Can you create a LinkedIn post: summarize the text: {st.session_state['summary']}; 
                In the post use these parameters: use {tone} tone, use {english_level} English, 
                use {length} length of post, {emoji_usage}, targeted at {audience}, {hashtag_preference}, 
                written from a {perspective} perspective. Do not use passive voice. Use popular terms. Use sentences with one or no conjunctions. Use words with one or two syllables where possible. {option_1}
                """
                st.session_state['post_text'] = post_text
            else:
                st.warning("Please wait until the summarization is complete before generating the post.")

with col3:
    st.markdown("## LinkedIn post output")
    if 'post_text' in st.session_state:
        llm = Ollama(model="llama3")
        generated_post = llm.invoke(st.session_state['post_text'])
        scroll_to_top()
        st.text_area("Generated LinkedIn Post", generated_post, height=700)
