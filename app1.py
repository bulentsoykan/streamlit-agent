import streamlit as st


# Library for Entailment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

text_classification_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large-mnli"
)


### Streamlit interface ###

st.title("Text Classification")

st.subheader("Entailment, neutral or contradiction?")

with st.form("submission_form", clear_on_submit=False):
    threshold = st.slider(
        "Threshold", min_value=0.0, max_value=1.0, step=0.1, value=0.7
    )

    sentence_1 = st.text_input("Sentence 1 input")

    sentence_2 = st.text_input("Sentence 2 input")

    submit_button_compare = st.form_submit_button("Compare Sentences")

# If submit_button_compare clicked
if submit_button_compare:
    print("Comparing sentences...")

    ### Text classification - entailment, neutral or contradiction ###

    raw_inputs = [f"{sentence_1}</s></s>{sentence_2}"]

    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

    # print(inputs)

    outputs = text_classification_model(**inputs)

    outputs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # print(outputs)

    # argmax_index = torch.argmax(outputs).item()

    print(
        text_classification_model.config.id2label[0],
        ":",
        round(outputs[0][0].item() * 100, 2),
        "%",
    )
    print(
        text_classification_model.config.id2label[1],
        ":",
        round(outputs[0][1].item() * 100, 2),
        "%",
    )
    print(
        text_classification_model.config.id2label[2],
        ":",
        round(outputs[0][2].item() * 100, 2),
        "%",
    )

    st.subheader("Text classification for both sentences:")

    st.write(
        text_classification_model.config.id2label[1],
        ":",
        round(outputs[0][1].item() * 100, 2),
        "%",
    )
    st.write(
        text_classification_model.config.id2label[0],
        ":",
        round(outputs[0][0].item() * 100, 2),
        "%",
    )
    st.write(
        text_classification_model.config.id2label[2],
        ":",
        round(outputs[0][2].item() * 100, 2),
        "%",
    )

    entailment_score = round(outputs[0][2].item(), 2)

    if entailment_score >= threshold:
        st.subheader("The statements are very similar!")
    else:
        st.subheader("The statements are not close enough")
