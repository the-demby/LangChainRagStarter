import streamlit as st
from rag_simple import get_compiled_rag

st.title("AI-Act RAG  App Simple")
st.session_state["rag_client"] = get_compiled_rag()
# Accept user input
if prompt := st.chat_input("Demandez moi quelquechose à propos de l'AI Act."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Réponse en cours de génération"):
            response = st.session_state["rag_client"].invoke({"question": prompt})
        st.write(response["reponse"])