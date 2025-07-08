from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from globals import LLM, VECTOR_STORE, Search

def get_compiled_rag() -> CompiledStateGraph:
    memory = MemorySaver()
    # Plutot que de définir un états dédié à l'extraction de données de la base vectorielle, on peut definir la fonction 
    # d'éxtraction comme un tool qui peut ensuite être donné au llm. 
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Renvoie de l'information liée à la requéte."""
        # On applique directement l'analyse de question utilisateur dans l'outil de requete pour obtenir une meilleur requete de recherche.
        structured_llm = LLM.with_structured_output(Search)
        requete_index = structured_llm.invoke(query)
        # Avec requete_index["nature"] on peut réduire le champs de recherche en filtrant sur metadata/nature dans le similarity_search
        # print(requete_index["nature"])
        filter_str = f"nature/any(n:"+" "+" or " + [f"n eq {nature}" for nature in requete_index["nature"]]+")"
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
        """Lance une requete à l'index ou répond. Permet d'éviter de lancer des requéte inutile vers la base vectorielle."""
        llm_with_tools = LLM.bind_tools([retrieve])
        reponse = llm_with_tools.invoke(state["messages"])
        return {"messages": [reponse]}

    tools = ToolNode([retrieve])

    def generate(state: MessagesState):
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
            Tu est un expert legal qui conseil l'utilisateur sur toute question en lien avec l'AI Act. 
            On intéragira avec toi par questions réponses. 
            
            En te basant sur le context suivant et l'historique de la conversation, répond à la question utilisateur.
            Les éléments de réponse doivent être suivie du url vers la section de l'AI Act référencé (procuré dans les métadonnée des documents sous la clef 'url'). le format est le suivant:
            <élément de réponse>[titre](<url>). 
            Format la réponse en markdown.
            Si il n'y à pas de context fourni ou que sont rapport à la question est marginal, précède ta réponse par la note suivante:

            "Ma base de connaissance semble limité pour répondre à votre question, les éléments de réponse suivant doivent être pris avec des pincettes !"

            Context: {context}
            """
        message_conversationel = [
            message for message in state["messages"] 
            if message.type in ("human", "system") 
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(prompt_systeme)] + message_conversationel
        return {"messages": [LLM.invoke(prompt)]}

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


