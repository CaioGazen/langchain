import os
import time
import argparse
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import Dict, List, Any

# --- Configuração ---
load_dotenv()


# Códigos de escape ANSI para saída de console colorida
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    USER_INPUT = "\033[93m"  # Amarelo para o usuário
    BOT_RESPONSE = "\033[96m"  # Ciano para o bot
    VERBOSE = "\033[90m"  # Cinza para verbose


# Variáveis de Ambiente
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "intelligent-tourism-guide"


# --- Funções de Utilitários ---
def print_header(message, color=Colors.HEADER):
    """Imprime um cabeçalho formatado."""
    print(f"\n{color}{Colors.BOLD}--- {message} ---{Colors.ENDC}")


def print_section(title, content, color=Colors.OKBLUE):
    """Imprime uma seção formatada."""
    print(f"{color}{Colors.BOLD}{title}:{Colors.ENDC} {content}")


def print_verbose(title, content):
    """Imprime conteúdo verbose em cinza."""
    print(f"{Colors.VERBOSE}\n--- {title} ---{Colors.ENDC}")
    print(f"{Colors.VERBOSE}{content}{Colors.ENDC}")


# --- Inicialização dos Componentes Principais ---
def initialize_components():
    """Carrega e inicializa todos os componentes necessários (Pinecone, LLMs, Embeddings)."""
    print_header("INICIALIZANDO COMPONENTES DO SISTEMA")

    # Validar variáveis de ambiente
    if not PINECONE_API_KEY or not GROQ_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY e GROQ_API_KEY devem ser definidos no arquivo .env."
        )

    # Inicializar Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    stats = pinecone_index.describe_index_stats()
    print_section(
        "Status do Pinecone",
        f"Conectado ao índice '{INDEX_NAME}' com {stats['total_vector_count']} vetores.",
        Colors.OKGREEN,
    )

    # Inicializar LLMs
    generation_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.2
    )
    router_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.0
    )
    print_section(
        "Status dos LLMs",
        "LLMs de Geração e Roteador (Llama 3.1 8B) carregados.",
        Colors.OKGREEN,
    )

    # Inicializar o Modelo de Embeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print_section(
        "Status dos Embeddings", "Modelo (all-MiniLM-L6-v2) carregado.", Colors.OKGREEN
    )

    return pinecone_index, generation_llm, router_llm, embeddings_model


# --- Definições de RAG e Cadeias ---
def fetch_context(query: str, index, embeddings, top_k: int = 5) -> str:
    """
    Busca o contexto relevante do Pinecone com base na consulta do usuário.
    """
    try:
        query_embedding = embeddings.embed_query(query)
        results = index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )

        documents = [match["metadata"]["text"] for match in results["matches"]]
        return (
            "\n".join(f"• {doc}" for doc in documents)
            if documents
            else "Nenhum contexto relevante encontrado."
        )
    except Exception as e:
        print(f"{Colors.FAIL}Erro durante a busca RAG: {e}{Colors.ENDC}")
        return "Erro ao buscar contexto."


def create_specialized_chains(llm: ChatGroq) -> Dict[str, RunnableSequence]:
    """
    Cria as cadeias de processamento especializadas para diferentes intenções do usuário.
    """
    templates = {
        "itinerary": """
        Você é um especialista em criar roteiros de viagem.
        Com base no contexto fornecido e na pergunta do usuário, crie uma sugestão de roteiro clara, organizada e inspiradora.
        Seja prático e dê dicas úteis.

        Contexto:
        {contexto}

        Pergunta: {query}

        Roteiro Sugerido:""",
        "logistics": """
        Você é um especialista em logística de viagens.
        Responda à pergunta do usuário com informações precisas e práticas sobre transporte, hospedagem e outras logísticas.
        Seja direto e foque em fornecer soluções.

        Contexto:
        {contexto}

        Pergunta: {query}

        Informação Logística:
        """,
        "info-local": """
        Você é um guia local experiente.
        Compartilhe informações sobre a cultura, costumes, gastronomia e dicas que só um morador local saberia.
        Sua resposta deve ser amigável e informativa.

        Contexto:
        {contexto}

        Pergunta: {query}

        Dica Local:
        """,
        "translation": """
        Você é um assistente de tradução para viajantes.
        Forneça a tradução solicitada e, se possível, inclua dicas de pronúncia ou frases alternativas úteis.

        Contexto (pode ser irrelevante para tradução direta):
        {contexto}

        Pergunta: {query}

        Tradução e Dicas:
        """,
    }

    chains = {}
    for name, template in templates.items():
        prompt = PromptTemplate(
            input_variables=["contexto", "query"], template=template
        )
        chains[name] = prompt | llm | StrOutputParser()

    return chains


def create_router_chain(llm: ChatGroq) -> RunnableSequence:
    """
    Cria a cadeia de roteador para classificar a intenção do usuário.
    """
    router_template = """
    Você é um classificador de intenções para um assistente de turismo.
    Analise a consulta do usuário e classifique-a em uma das seguintes categorias:

    - itinerary: Pedidos de roteiros, sugestões de o que fazer, atrações, pontos turísticos.
    - logistics: Perguntas sobre transporte, aeroportos, hotéis, como chegar a um lugar, locomoção.
    - info-local: Dúvidas sobre cultura, comida, costumes, segurança, dicas locais.
    - translation: Pedidos de tradução de frases ou palavras para outro idioma.

    CONSULTA: "{query}"

    Responda APENAS com o nome da categoria (itinerary, logistics, info-local, ou translation).
    """
    prompt = PromptTemplate(input_variables=["query"], template=router_template)
    return prompt | llm | StrOutputParser()


# --- Classe Principal da Aplicação ---
class IntelligentTourGuide:
    """
    A classe principal do sistema de guia de turismo inteligente.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.index, self.gen_llm, self.router_llm, self.embeddings = (
            initialize_components()
        )
        self.router_chain = create_router_chain(self.router_llm)
        self.specialized_chains = create_specialized_chains(self.gen_llm)

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processa a consulta de um usuário, da classificação à resposta final.
        """
        start_time = time.time()

        # 1. Classificar a intenção usando a cadeia de roteador
        intent = self.router_chain.invoke({"query": query}).strip().lower()
        if intent not in self.specialized_chains:
            intent = "itinerary"  # Retorno para uma cadeia padrão

        # 2. Buscar contexto relevante usando RAG
        context = fetch_context(query, self.index, self.embeddings)

        if self.verbose:
            print_verbose("CONTEXTO ENCONTRADO", context)

        # 3. Executar a cadeia especializada
        chain = self.specialized_chains[intent]
        response = chain.invoke({"contexto": context, "query": query})

        processing_time = (time.time() - start_time) * 1000  # em ms

        return {
            "response": response,
            "intent": intent,
            "context_found": context != "Nenhum contexto relevante encontrado.",
            "processing_time_ms": processing_time,
        }


# --- Demonstração e Modo Interativo ---
def run_demonstration(guide: IntelligentTourGuide):
    """
    Executa uma série de consultas predefinidas para demonstrar as capacidades do sistema.
    """
    print_header("DEMONSTRAÇÃO DO SISTEMA", color=Colors.OKGREEN)

    demo_queries = [
        "Qual o melhor roteiro para 3 dias em Tóquio, focado em cultura e tecnologia?",
        "Como funciona o transporte público em Sydney? Devo comprar um passe?",
        "Quais os pratos típicos que preciso provar em Roma e por que não devo pedir cappuccino depois do almoço?",
        "Qual a etiqueta para visitar templos no Japão?",
        "Como se diz 'com licença, onde fica a estação de trem?' em japonês?",
        "É verdade que a gorjeta é rude em Tóquio? E como funciona em Sydney?",
        "Vale a pena subir na Harbour Bridge em Sydney ou a vista do chão é suficiente?",
    ]

    for i, query in enumerate(demo_queries):
        print(f"\n{Colors.USER_INPUT}--- Consulta {i+1} ---{Colors.ENDC}")
        print(f"{Colors.USER_INPUT}Usuário:{Colors.ENDC} {query}")

        result = guide.process_query(query)

        print(f"{Colors.BOT_RESPONSE}Bot:{Colors.ENDC}")
        print(result["response"])
        print(
            f"{Colors.OKCYAN}(Intenção: {result['intent']} | RAG: {'Hit' if result['context_found'] else 'Miss'} | Tempo: {result['processing_time_ms']:.0f}ms){Colors.ENDC}"
        )


def run_interactive_mode(guide: IntelligentTourGuide):
    """
    Executa o sistema em um loop de prompt interativo.
    """
    print_header("MODO INTERATIVO", color=Colors.OKGREEN)
    print(
        f"{Colors.OKCYAN}Digite sua pergunta. Digite 'sair' para encerrar.{Colors.ENDC}"
    )
    while True:
        query = input(f"\n{Colors.USER_INPUT}Você:{Colors.ENDC} ").strip()
        if query.lower() == "sair":
            print(
                f"{Colors.OKCYAN}Encerrando o guia de turismo. Até a próxima!{Colors.ENDC}"
            )
            break

        if not query:
            continue

        result = guide.process_query(query)
        print(f"{Colors.BOT_RESPONSE}Bot:{Colors.ENDC}")
        print(result["response"])
        print(
            f"{Colors.OKCYAN}(Intenção: {result['intent']} | RAG: {'Hit' if result['context_found'] else 'Miss'} | Tempo: {result['processing_time_ms']:.0f}ms){Colors.ENDC}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Guia de turismo inteligente usando RAG."
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Ativa o modo interativo. Se não for usado, executa o modo de demonstração.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Ativa o modo verbose, mostrando o contexto recuperado do Pinecone.",
    )
    args = parser.parse_args()

    try:
        tour_guide = IntelligentTourGuide(verbose=args.verbose)
        if args.interactive:
            run_interactive_mode(tour_guide)
        else:
            run_demonstration(tour_guide)
    except Exception as e:
        print(f"\n{Colors.FAIL}✖ Ocorreu um erro crítico: {e}{Colors.ENDC}")
