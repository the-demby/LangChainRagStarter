from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from langchain_core.prompts import HumanMessagePromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from typing_extensions import TypedDict, Annotated, Literal, List
from dotenv import load_dotenv
import os
load_dotenv(override=True)

rate_limiter = InMemoryRateLimiter(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=10
    )

LLM: AzureChatOpenAI = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    rate_limiter=rate_limiter,
    streaming=True
) 

EMBEDDINGS: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    retry_max_seconds=60,
    retry_min_seconds=10,
    max_retries=5,
    chunk_size=512,
    embedding_ctx_length=1000,
    check_embedding_ctx_length=True,
    skip_empty=True
)

VECTOR_STORE = AzureSearch(
    azure_search_endpoint=os.environ["SEARCH_ENDPOINT"],
    azure_search_key=os.environ["SEARCH_KEY"],
    index_name=os.environ["SEARCH_INDEX"],
    embedding_function=EMBEDDINGS.embed_query,
    additional_search_client_options={"retry_total": 4},
)

PROMPT = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'], 
                input_types={}, 
                partial_variables={}, 
                template="""
                Tu es un expert légal qui conseille l'utilisateur sur toute question en lien avec l'AI Act. 
                
                En te basant sur le contexte suivant, réponds à la question.  
                S'il n'y a pas de contexte fourni ou que son rapport à la question est marginal, précède ta réponse par la note suivante:

                "Ma base de connaissance semble limitée pour répondre à votre question, les éléments de réponse suivants doivent être pris avec des pincettes !" 
                
                \nQuestion: {question}
                \nContext: {context} 
                \Réponse:"""
            ), additional_kwargs={}
        )
    ]
)

class Search(TypedDict):
    """Format de sortie LLM pour transformer une question utilisateur en une requête pour une base vectorielle avec une liste de filtres."""
    requete_indexe: Annotated[str, ..., "Cherche la requête à envoyer à la base vectorielle à partir de cette question."]
    nature: Annotated[
        List[Literal["considerant", "article", "annexe"]],
        ...,
        "Nature des éléments parmi lesquels chercher."
    ]

def format_docs_into_citation(docs: list[Document]) -> str:
    """Formateur basique de citations pour compléter les réponses du RAG."""
    formatted = [
        f"Source-{i}: [{doc.metadata['titre']}]({doc.metadata['url']})"
        for i, doc in enumerate(docs)
    ]
    return "\n\n"+"\n\n".join(formatted)



