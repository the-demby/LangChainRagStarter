import tiktoken, glob, json, time, re, ast, os, random, logging
import pandas as pd
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticPrioritizedFields, SemanticSearch, SemanticField,
    AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters
)
from dotenv import load_dotenv
from tqdm import tqdm

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# === CREDENTIALS & PARAMS ===
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ED")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ED")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

DIR = "*.csv"
ACTIONNAIRE_CSV = "data/ACTIONNAIRES.csv"
CHUNK_TOK = 6000
CHUNK_OVER = 100
MIN_CHUNK_TOK = 200
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBED_BATCH = 8
BATCH_UPLOAD = 975
JSON_PATH = "chunked_docs_final.json"
ERROR_LOG_PATH = "failed_uploads.log"
EMBED_BEFORE_UPLOAD = os.getenv("EMBED_BEFORE_UPLOAD", "False").lower() == "true"

CONTENT_FIELDS = [
    "search_result_title", "search_result_snippet", "page_description", "page_content"
]
METADATA_FIELDS = [
    "search_result_link", "page_language"
]

# --- LOGGING SETUP ---
logging.basicConfig(
    filename=ERROR_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_failed_docs(docs, error_type="upload"):
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({"error_type": error_type, "doc_id": doc.get("id", None), "meta": doc}, ensure_ascii=False) + "\n")

# --- UTILITAIRES ---
def read_csv_flexible(filepath):
    for encoding in ["utf-8-sig", "utf-16"]:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except Exception:
            continue
    print(f"Impossible de lire {filepath}")
    return None

def clean_text(text):
    if not text:
        return ""
    try:
        text = BeautifulSoup(str(text), "html.parser").get_text()
    except Exception:
        text = str(text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_actionnaires_from_results(results_str):
    try:
        lst = ast.literal_eval(results_str)
        noms = []
        for d in lst:
            snippet = d.get("snippet", "")
            found = re.findall(r"actionnaire[s]? [:\-\–]?\s?([A-Za-z0-9,.\(\) ]+)", snippet, flags=re.I)
            for f in found:
                noms.append(f.strip())
        return "; ".join(set(noms)) if noms else "Non trouvé"
    except Exception:
        return "Non trouvé"

def load_actionnaires(actionnaire_csv):
    df = read_csv_flexible(actionnaire_csv)
    if df is None:
        return {}
    df["name"] = df["name"].astype(str).str.strip().str.upper()
    df["actionnaires"] = df["results"].apply(extract_actionnaires_from_results)
    return dict(zip(df["name"], df["actionnaires"]))

def prepare_content(row, content_fields):
    content = ""
    for field in content_fields:
        value = row.get(field, "")
        if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan":
            content += clean_text(value) + "\n"
    return content.strip()

def prepare_metadata(row, metadata_fields, actionnaires_dict=None):
    metadata = {}
    for field in metadata_fields:
        value = row.get(field, "")
        if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan":
            metadata[field] = value
    company_name = row.get("name", None)
    if actionnaires_dict and company_name:
        meta_name = str(company_name).strip().upper()
        metadata["actionnaires"] = actionnaires_dict.get(meta_name, "Non trouvé")
    else:
        metadata["actionnaires"] = "Non trouvé"
    return metadata

def load_and_prepare_documents(DIR, content_fields, metadata_fields, actionnaires_dict):
    all_docs = []
    files = glob.glob(DIR)
    for file in files:
        if ACTIONNAIRE_CSV.lower() in file.lower():
            continue
        df = read_csv_flexible(file)
        if df is None:
            continue
        for idx, row in df.iterrows():
            content = prepare_content(row, content_fields)
            metadata = prepare_metadata(row, metadata_fields, actionnaires_dict)
            if content:
                all_docs.append({"content": content, **metadata})
    print(f"Nombre de documents préparés : {len(all_docs)}")
    return all_docs

def chunk_documents_by_tokens(
    documents,
    chunk_size=CHUNK_TOK,
    overlap=CHUNK_OVER,
    min_tokens=MIN_CHUNK_TOK,
    max_bytes=32766
):
    encoding = tiktoken.get_encoding("cl100k_base")
    chunked_docs = []

    def split_chunk_tokens(tokens, doc, base_id, max_bytes):
        if not tokens:
            return
        chunk_text = encoding.decode(tokens)
        size = len(chunk_text.encode("utf-8"))
        if size <= max_bytes:
            new_doc = dict(doc)
            new_doc["content"] = chunk_text
            new_doc["id"] = base_id
            if EMBED_BEFORE_UPLOAD:
                if "embedding" not in new_doc:
                    new_doc["embedding"] = []
            chunked_docs.append(new_doc)
        else:
            mid = len(tokens) // 2
            # Recoupe les deux moitiés, avec nouvel id
            split_chunk_tokens(tokens[:mid], doc, base_id + "_a", max_bytes)
            split_chunk_tokens(tokens[mid:], doc, base_id + "_b", max_bytes)

    for idx, doc in enumerate(documents):
        tokens = encoding.encode(doc["content"])
        if not tokens:
            continue
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            if len(chunk_tokens) < min_tokens:
                continue
            base_id = f"doc_{idx}_{i}"
            split_chunk_tokens(chunk_tokens, doc, base_id, max_bytes)

    print(f"Nombre de chunks créés : {len(chunked_docs)}")
    return chunked_docs

# --- EXPONENTIAL BACKOFF ---
def exponential_backoff_retry(func, *args, max_retries=7, base_delay=2, error_type="upload", failed_docs=None, **kwargs):
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[Backoff] Erreur lors du call (tentative {attempt}/{max_retries}): {e}")
            if failed_docs is not None:
                log_failed_docs(failed_docs, error_type=error_type)
            if attempt == max_retries:
                print("Abandon du batch après plusieurs tentatives.")
                raise
            sleep_time = delay + random.uniform(0, 0.5 * delay)
            print(f"Pause {sleep_time:.1f}s avant retry...")
            time.sleep(sleep_time)
            delay *= 2

# --- EMBEDDING OPTIONNEL ---
def add_embeddings_to_chunks(chunked_docs, oaiclient, embed_model, batch=EMBED_BATCH):
    print("Démarrage embedding Azure OpenAI…")
    for i in tqdm(range(0, len(chunked_docs), batch), desc="Embedding batches"):
        batch_docs = chunked_docs[i:i+batch]
        contents = [doc["content"] for doc in batch_docs]
        def embed_call():
            resp = oaiclient.embeddings.create(input=contents, model=embed_model)
            for idxb, doc in enumerate(batch_docs):
                doc["embedding"] = resp.data[idxb].embedding
        try:
            exponential_backoff_retry(embed_call, max_retries=6, base_delay=5, error_type="embedding", failed_docs=batch_docs)
        except Exception as e:
            print(f"Batch embedding failed après backoff : {e}")
    print("Embeddings générés pour tous les chunks.")
    return chunked_docs

def save_chunks_to_json(docs, path=JSON_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

def create_or_reset_index():
    idx_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )
    # --- Suppression de l’ancien index
    try:
        if INDEX_NAME in [idx.name for idx in idx_client.list_indexes()]:
            idx_client.delete_index(INDEX_NAME)
            print(f"Index '{INDEX_NAME}' supprimé")
    except Exception as e:
        print(f"Erreur suppression index : {e}")

    # --- Champs de l’index
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="search_result_link", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="page_language", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="actionnaires", type=SearchFieldDataType.String, retrievable=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True, retrievable=False,
            vector_search_dimensions=1536,
            vector_search_profile_name="default"
        )
    ]

    # -- (3) Vectorizer (nom sans espace ni majuscule)
    vect_name = "openai_vectorizer"
    vectorizer = AzureOpenAIVectorizer(
        vectorizer_name=vect_name,
        parameters=AzureOpenAIVectorizerParameters(
            resource_url=AZURE_OPENAI_ENDPOINT,
            deployment_name=EMBEDDING_MODEL,
            model_name=EMBEDDING_MODEL,
            api_key=AZURE_OPENAI_KEY         
        )
    )

    vector_search = VectorSearch(
        vectorizers=[vectorizer],
        profiles=[VectorSearchProfile(
            name="default",
            algorithm_configuration_name="hnsw-config",
            vectorizer_name=vect_name
        )],
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")]
    )

    # --- Recherche sémantique
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="content"),
                    content_fields=[SemanticField(field_name="content")]
                )
            )
        ],
        default_configuration_name="default"
    )

    # --- Création de l’index
    idx = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        vectorizers=[vectorizer],
        semantic_search=semantic_search
    )
    idx_client.create_or_update_index(idx)
    print(f"Index '{INDEX_NAME}' (re)créé")

def get_json_size(obj):
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

def upload_batch_with_backoff(search_client, batch, batch_num, total_batches):
    def get_doc_size(doc):
        return len(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    def inner_upload():
        res = search_client.upload_documents(documents=batch)
        # map id -> result détail de l'erreur
        res_by_id = {r.key: r for r in res}
        failed = [r.key for r in res if not r.succeeded]
        if failed:
            print(f"Problèmes sur: {failed}")
            # Log détaillé
            for d in batch:
                if d["id"] in failed:
                    size = get_doc_size(d)
                    azure_error = getattr(res_by_id[d["id"]], "error_message", None)
                    print(f"    - Doc ID: {d['id']}")
                    print(f"      Taille: {size/1024:.2f} Ko")
                    print(f"      Champs: {list(d.keys())}")
                    print(f"      Début content: {repr(d.get('content', '')[:100])}")
                    print(f"      Présence embedding: {'embedding' in d}")
                    if azure_error:
                        print(f"      Azure error: {azure_error}")
                    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "error_type": "upload",
                            "doc_id": d["id"],
                            "size_bytes": size,
                            "fields": list(d.keys()),
                            "content_sample": d.get("content", "")[:200],
                            "has_embedding": "embedding" in d,
                            "azure_error": azure_error
                        }, ensure_ascii=False) + "\n")
        else:
            print(f"Batch {batch_num}/{total_batches} ok")
        time.sleep(0.5)
    exponential_backoff_retry(inner_upload, max_retries=6, base_delay=5, error_type="upload", failed_docs=batch)

def upload_json_to_index(json_path, search_client, start_batch=0):
    BATCH_SIZE = BATCH_UPLOAD
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"Upload de {len(docs)} docs dans AzureSearch..")
    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in tqdm(range(start_batch, total_batches), desc="Upload batches"):
        batch = docs[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
        batch_size_bytes = get_json_size(batch)
        if batch_size_bytes > 16*1024*1024 and len(batch) > 1:
            print(f"Batch {b+1} ({len(batch)}) fait {batch_size_bytes/1024/1024:.2f} Mo > 16Mo : On split")
            half = len(batch) // 2
            upload_batch_with_backoff(search_client, batch[:half], b+1, total_batches)
            upload_batch_with_backoff(search_client, batch[half:], b+1, total_batches)
        else:
            upload_batch_with_backoff(search_client, batch, b+1, total_batches)

def print_error_summary():
    if os.path.exists(ERROR_LOG_PATH):
        with open(ERROR_LOG_PATH, encoding="utf-8") as f:
            errors = f.readlines()
    else:
        print("Aucune erreur enregistrée dans le pipeline.")

# ===== MAIN PIPELINE =====
if __name__ == "__main__":
    actionnaires_dict = load_actionnaires(ACTIONNAIRE_CSV)
    docs = load_and_prepare_documents(
        DIR,
        CONTENT_FIELDS,
        METADATA_FIELDS,
        actionnaires_dict
    )
    chunked_docs = chunk_documents_by_tokens(
        docs,
        chunk_size=CHUNK_TOK,
        overlap=CHUNK_OVER,
        min_tokens=MIN_CHUNK_TOK
    )

    if EMBED_BEFORE_UPLOAD:
        oaiclient = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2024-10-21"
        )
        chunked_docs = add_embeddings_to_chunks(chunked_docs, oaiclient, EMBEDDING_MODEL)
    save_chunks_to_json(chunked_docs, JSON_PATH)

    create_or_reset_index()
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )
    upload_json_to_index(JSON_PATH, search_client)
    print_error_summary()
    print("\nCome on dudeeeee 🎉")
