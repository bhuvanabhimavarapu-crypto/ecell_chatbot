import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


faq = pd.read_csv("chatbot_faq.csv")


model = SentenceTransformer("all-MiniLM-L6-v2")


question_embeddings = model.encode(
    faq["question"].tolist()
)


def get_answer(user_query):

    query_embedding = model.encode(
        [user_query]
    )

    similarities = cosine_similarity(
        query_embedding,
        question_embeddings
    )

    best_index = similarities.argmax()

    best_score = similarities[0][best_index]

    if best_score < 0.50:
        return (
            "Sorry, this information is not available in the JNTUH E-Cell knowledge base."
        )

    return faq.iloc[best_index]["answer"]


st.set_page_config(
    page_title="JNTUH E-Cell Chatbot",

)

st.title("JNTUH E-Cell Chatbot")

st.write(
    "Ask questions about entrepreneurship, startups, incubation, innovation, funding, and E-Cell activities."
)

question = st.text_input(
    "Ask your question:"
)

if st.button("Submit"):

    if question:

        answer = get_answer(question)

        st.success(answer)
