import streamlit as st
from rag_chat import get_compiled_rag
from time import time

st.title("AI-Act RAG  App")
# Initialisation des variables de la session
if 'config' not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": str(time)}}
if 'rag_client' not in st.session_state:
    st.session_state.rag_client = get_compiled_rag()
if "messages" not in st.session_state:
    st.session_state.messages = []
# Affichage de l'historique de discussion
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Bouton pour réinitialiser l'historique et la mémoire du RAG
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.config = {"configurable": {"thread_id": str(time)}}

# Zone de question utilisateur
if prompt := st.chat_input("Demandez moi quelquechose à propos de l'AI Act."):
    # Ajouter les questions utilisateur à l'historique de conversation
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Montrer le message utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
    # Streamer la réponse du chatbot 
    with st.chat_message("assistant"):
        stream = (
            step[0].content for step in 
            st.session_state.rag_client.stream(
                {
                    "messages":[
                        {"role": "user", "content": prompt}
                    ]
                },
                stream_mode="messages",
                config=st.session_state.config
            )
            # Les seuls éléments à monter sont les réponses des états 'generate' et 'query_or_respond'. cf graph d'état du step_by_step_rag.ipynb  
            # 'generate' quand le RAG utilise le tool pour récupérer de l'information dans la base vectorielle.
            # 'query_or_respond' quand le RAG répond directement sans passer par la base vectorielle.
            if step[1].get("langgraph_node") in ("generate", "query_or_respond")
        )
        response = st.write_stream(stream)
    # Ajouter la réponse llm à l'historique de conversation
    st.session_state.messages.append({"role": "assistant", "content": response})