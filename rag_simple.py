from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import List, TypedDict
from globals import PROMPT, LLM, VECTOR_STORE, Search, format_docs_into_citation

class State(TypedDict):
    question: str
    requete_indexe: Search
    context: List[Document]
    reponse: str

def get_compiled_rag(
        llm: AzureChatOpenAI=LLM, 
        setup_prompt: ChatPromptTemplate=PROMPT, 
        vector_store: AzureSearch=VECTOR_STORE
) -> CompiledStateGraph:
    """
    Fonction qui retourne un RAG pret a être questionner
    """
    def query_analysis(state: State):
        """Fonction qui reformule la question utilisateur en requete à envoyer vers l'index vectoriel."""
        structured_llm = llm.with_structured_output(Search)
        requete_index = structured_llm.invoke(state["question"])
        return {"requete_indexe": requete_index}

    def retrieve(state: State):
        """Fonction qui récupère de l'index vectoriel les documents pertinents."""
        requete_index = state["requete_indexe"]
        retrieved_docs = vector_store.similarity_search(
            requete_index["requete_indexe"],
        )
        return {"context": retrieved_docs}

    def generate(state: State):
        """Fonction qui génére la réponse du model."""
        docs_content = "\n".join(doc.page_content for doc in state["context"])
        messages = setup_prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        # Ajoutez les citations formatter en fin de réponse.
        response.content += format_docs_into_citation(state["context"])
        return {"reponse": response.content}
    
    graph_builder = StateGraph(State).add_sequence([query_analysis ,retrieve, generate])
    graph_builder.add_edge(START, "query_analysis")
    return graph_builder.compile()



