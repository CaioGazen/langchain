import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import re

# --- Configuration ---
load_dotenv()


# ANSI escape codes for colored console output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = "intelligent-tourism-guide"
KNOWLEDGE_BASE_FILE = "knowledge_base.txt"

# --- Main Script ---


def print_header(message):
    """Prints a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}--- {message} ---{Colors.ENDC}")


def print_success(message):
    """Prints a success message."""
    print(f"{Colors.OKGREEN}✔ {message}{Colors.ENDC}")


def print_info(message):
    """Prints an informational message."""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def print_warning(message):
    """Prints a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def setup_pinecone_index():
    """Initializes Pinecone and creates the index if it doesn't exist."""
    print_header("INITIALIZING PINECONE")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not configured in the .env file.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print_info(f"Index '{INDEX_NAME}' not found. Creating it...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # For sentence-transformers/all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )
        # Wait for the index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print_info("Waiting for index initialization...")
            time.sleep(1)
        print_success(f"Index '{INDEX_NAME}' created successfully!")
    else:
        print_success(f"Index '{INDEX_NAME}' already exists.")

    return pc.Index(INDEX_NAME)


def load_knowledge_base(file_path: str = KNOWLEDGE_BASE_FILE) -> dict:
    """
    Loads the knowledge base from a text file and organizes it into a dictionary.

    Args:
        file_path: Path to the knowledge base text file.

    Returns:
        A dictionary with data organized by city and category.
    """
    print_header("LOADING KNOWLEDGE BASE")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        data = {}

        current_section_key = None
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("# TOURISM"):
                continue

            # Check if the line is a section header (e.g., "## ITINERARY - TOKYO")
            if line.startswith("##"):
                # Use regex to robustly extract category and city
                match = re.match(r"##\s*([\w-]+)\s*-\s*(\w+)", line)
                if match:
                    category, city = match.groups()
                    current_section_key = f"{city.lower()}_{category.lower()}"
                    if current_section_key not in data:
                        data[current_section_key] = []
                continue

            if current_section_key:
                data[current_section_key].append(line)

        print_success(f"Knowledge base loaded from '{file_path}'.")
        for category, docs in data.items():
            print(f"  - {category}: {len(docs)} documents found.")
        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while processing the knowledge base: {e}"
        )


def apply_chunking(text: str, max_chars: int = 250) -> list:
    """
    Splits text into smaller chunks based on sentence structure.

    Args:
        text: The text to be chunked.
        max_chars: The maximum character length for each chunk.

    Returns:
        A list of text chunks.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    sentences = text.split(". ")
    current_chunk = ""

    for sentence in sentences:
        # Re-add the period to the sentence
        if sentence and not sentence.endswith("."):
            sentence += "."

        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def create_and_upsert_vectors(index, tourism_data, embeddings_model):
    """
    Applies chunking, creates embeddings, and upserts vectors to Pinecone.
    """
    print_header("PROCESSING AND INDEXING DATA")

    chunked_data = {}
    total_chunks = 0

    print_info("Applying chunking strategy...")
    for category, documents in tourism_data.items():
        category_chunks = []
        for doc in documents:
            chunks = apply_chunking(doc)
            category_chunks.extend(chunks)
            total_chunks += len(chunks)
        chunked_data[category] = category_chunks

    print_success(f"Chunking complete. Total chunks created: {total_chunks}")

    print_info("Preparing vectors for upsert...")
    vectors_to_upsert = []
    for category, chunks in chunked_data.items():
        # Split key into city and type, accommodating types with hyphens like 'info-local'
        city, type = category.split("_", 1)
        for i, chunk in enumerate(chunks):
            embedding = embeddings_model.embed_query(chunk)
            vector_id = f"{category}_{i}"
            vector_data = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "category": category,
                    "city": city,
                    "type": type,
                },
            }
            vectors_to_upsert.append(vector_data)

    print_success(f"{len(vectors_to_upsert)} vectors prepared.")

    # Clear existing data for a fresh start can error on first run, just comment if needed
    print_warning("Clearing all existing vectors from the index...")
    index.delete(delete_all=True)
    print_success("Index cleared.")

    # Upsert data in batches
    print_info("Upserting vectors to Pinecone...")
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  - Upserted batch {i//batch_size + 1}")

    print_success("All vectors have been indexed successfully!")


if __name__ == "__main__":
    try:
        # 1. Initialize Pinecone
        pinecone_index = setup_pinecone_index()

        # 2. Load knowledge base from file
        tourism_knowledge = load_knowledge_base()

        # 3. Initialize embeddings model
        print_header("LOADING EMBEDDINGS MODEL")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        print_success("Embeddings model loaded: all-MiniLM-L6-v2")

        # 4. Process data and upsert to Pinecone
        create_and_upsert_vectors(pinecone_index, tourism_knowledge, embeddings)

        # 5. Final verification
        print_header("SETUP COMPLETE")

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"\n{Colors.FAIL}✖ An error occurred during setup: {e}{Colors.ENDC}")
