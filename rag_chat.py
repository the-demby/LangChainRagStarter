from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from globals import LLM, VECTOR_STORE, Search

def get_compiled_rag() -> CompiledStateGraph:
    """Fonction qui retourne un RAG prêt à être questionné."""
    # On initialise la mémoire pour pouvoir avoir des interactions de type chat avec le RAG
    memory = MemorySaver()
    # On définit la fonction d'extraction comme un 'tool' qui peut ensuite être donné au LLM. En faisant ainsi, les appels à la base vectorielle ne sont que faits si le llm estime en avoir besoin.
    @tool(response_format="content_and_artifact")
    def retrieve(question: str):
        """Fonction qui récupère de l'index vectoriel les documents pertinents vis-à-vis de la question."""
        # On applique l'analyse de la question utilisateur pour obtenir une meilleure requête de recherche.
        structured_llm = LLM.with_structured_output(Search)
        requete_index = structured_llm.invoke(question)
        # Avec requete_index["nature"] on peut réduire le champs de recherche en filtrant sur metadata/nature dans le similarity_search
        filter_str = "or".join([f"nature eq '{nature}'" for nature in requete_index["nature"]])
        retrieved_docs = VECTOR_STORE.similarity_search(
            requete_index["requete_indexe"], 
            k = 5, 
            filters=filter_str
        )
        serialized = "\n".join(
            (f"Source: {doc.metadata}" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(state: MessagesState):
        """Lance une requête à l'index ou réponds directement à la question utilisateur. Permet d'éviter de lancer des requêtes inutiles vers la base vectorielle."""
        llm_with_tools = LLM.bind_tools([retrieve])
        reponse = llm_with_tools.invoke(state["messages"])
        return {"messages": [reponse]}

    tools = ToolNode([retrieve])

    def generate(state: MessagesState):
        """Génère une réponse en fonction du contexte renvoyé depuis la base vectorielle."""
        appel_outil_precedent = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                appel_outil_precedent.append(message)
            else: 
                break
        recherches_indexe = appel_outil_precedent[::-1]

        context = "" \
        "".join(doc.content for doc in recherches_indexe)

        prompt_systeme = f"""
            Tu es un expert légal qui conseille l'utilisateur sur toute question en lien avec l'AI Act. 
            On interagira avec toi par questions-réponses.
            
            En te basant sur le contexte suivant et l'historique de la conversation, réponds à la question utilisateur.
            Les éléments de réponse doivent être suivis du URL vers la section de l'AI Act référencée (procurée dans les métadonnées des documents sous la clef 'url'). le format est le suivant:
            <élément de réponse>[titre](<url>). 
            Format la réponse en markdown.
            S'il n'y a pas de contexte fourni ou que son rapport à la question est marginal, précède ta réponse par la note suivante:

            "Ma base de connaissance semble limitée pour répondre à votre question, les éléments de réponse suivants doivent être pris avec des pincettes !"

            Context: {context}
            """
        message_conversationel = [
            message for message in state["messages"] 
            if message.type in ("human", "system") 
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(prompt_systeme)] + message_conversationel
        return {"messages": [LLM.invoke(prompt)]}

    # On définit la machine d'état du RAG
    graph = StateGraph(MessagesState)
    graph.add_node(query_or_respond)
    graph.add_node(tools)
    graph.add_node(generate)
    graph.set_entry_point("query_or_respond")
    graph.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph.add_edge("tools", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=memory)


