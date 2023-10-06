import os, streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

st.image("./ucf2.png", width=400)


# create different page options on the left
page = st.sidebar.selectbox(
    "Choose a page", ["Home", "Rubric", "Grade", "Chatbot", "Text Summarizer"]
)


if page == "Home":
    st.title("H&P Assignment Grading")
    st.write("This app is for the H&P assignment grading")

    # options for different large language models

    # write the process for grading the history and physical write-up assignment
    st.header("Process")
    st.write(
        "The process for grading the history and physical write-up assignment is as follows:"
    )
    # step 1 review the rubric in the rubric page
    st.subheader("Step 1")
    st.write("Review the rubric in the rubric page")
    # step 2 provide feedback to the student
    st.subheader("Step 2")
    st.write("Upload or copy/paste the student's write-up and Grade the student")
    # step 3 provide feedback to the student
    st.subheader("Step 3")
    st.write("Provide feedback to the student")


# if the user selects the rubric page
if page == "Rubric":
    st.header("Clerkship History and Physical Write-up Assessment Rubric 2023")

    # add a selectbox to the sidebar
    rubric_type = st.sidebar.selectbox(
        "Choose a rubric type",
        ["Data Gathering", "Clinical Reasoning", "Diagnostic and Therapeutic Plan"],
    )

    # if the user selects the data gathering rubric
    if rubric_type == "Data Gathering":
        st.subheader("Data Gathering")
        # st.write("This is the data gathering rubric")
        with open("rubric.md", "r") as f:
            st.markdown(f.read())

    # if the user selects the clinical reasoning rubric
    if rubric_type == "Clinical Reasoning":
        st.subheader("Clinical Reasoning")
        with open("rubric2.md", "r") as f:
            st.markdown(f.read())

    # if the user selects the diagnostic and therapeutic plan rubric
    if rubric_type == "Diagnostic and Therapeutic Plan":
        st.subheader("Diagnostic and Therapeutic Plan")
        # st.write("This is the diagnostic and therapeutic plan rubric")
        with open("rubric3.md", "r") as f:
            st.markdown(f.read())

# if the user selects the Grade page
if page == "Grade":
    st.header("Grade")
    st.write("This is the Grade page")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    # selector for temperature
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    # explanatiom for temperature
    st.text("The higher the temperature the more random the output.")

    # create a tab for copy/paste and upload
    tab1, tab2 = st.tabs(["Copy/Paste", "Upload"])

    with tab1:
        # create a tab for copy/paste
        st.subheader("Copy/Paste")
        # create a text area for the student's write-up
        source_text = st.text_area("Source Text", height=200)
        # create a button to grade the student's write-up
        if st.button("Grade"):
            # Validate inputs
            if not openai_api_key.strip() or not source_text.strip():
                st.write(f"Please complete the missing fields.")
            else:
                try:
                    # Split the source text
                    text_splitter = CharacterTextSplitter()
                    texts = text_splitter.split_text(source_text)

                    # Create Document objects for the texts
                    docs = [Document(page_content=t) for t in texts[:3]]

                    # Initialize the OpenAI module, load and run the summarize chain
                    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                    chain = load_summarize_chain(llm, chain_type="map_reduce")
                    summary = chain.run(docs)

                    # Display summary
                    st.write(summary)
                except Exception as e:
                    st.write(f"An error occurred: {e}")

        with tab2:
            # create a tab for upload
            st.subheader("Upload")
            # create a file uploader for the student's write-up


# if the user selects the chatbot page
if page == "Chatbot":
    st.header("Chatbot")
    st.write("This is the chatbot page")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    # explanatiom for temperature
    st.text("The higher the temperature the more random the output.")


if page == "Text Summarizer":
    st.title("Text Summarizer")

    # Get OpenAI API key and source text input
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    # explanatiom for temperature
    st.text("The higher the temperature the more random the output.")
    source_text = st.text_area("Source Text", height=200)

    # Check if the 'Summarize' button is clicked
    if st.button("Summarize"):
        # Validate inputs
        if not openai_api_key.strip() or not source_text.strip():
            st.write(f"Please complete the missing fields.")
        else:
            try:
                # Split the source text
                text_splitter = CharacterTextSplitter()
                texts = text_splitter.split_text(source_text)

                # Create Document objects for the texts
                docs = [Document(page_content=t) for t in texts[:3]]

                # Initialize the OpenAI module, load and run the summarize chain
                llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = chain.run(docs)

                # Display summary
                st.write(summary)
            except Exception as e:
                st.write(f"An error occurred: {e}")
