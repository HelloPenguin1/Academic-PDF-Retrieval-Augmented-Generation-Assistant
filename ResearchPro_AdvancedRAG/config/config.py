from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import HypotheticalDocumentEmbedder
import torch
from groq import Groq

import os
from dotenv import load_dotenv
load_dotenv()


## for increased efficiency
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)


groq_api_key = os.getenv("GROQ_API_KEY")

##############################################################################################

hf_embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    encode_kwargs = {'normalize_embeddings':True},
)   

# Main LLM for final answer generation (large context window, best reasoning)
llm = ChatGroq(model="openai/gpt-oss-20b", 
               groq_api_key=groq_api_key,
               max_tokens=2048,      # Limit response to 2048 tokens
               temperature=0.1,      # Add consistency
               timeout=30            # 30 second timeout)
               )

# Lightweight LLM for query reformulation (reduces rate limit pressure)
llm_reformulate = ChatGroq(model="llama-3.1-8b-instant",
                           groq_api_key=groq_api_key,
                           max_tokens=256,       # Short reformulated queries
                           temperature=0.1,
                           timeout=10)

llm_summarize = ChatGroq(model="llama-3.3-70b-versatile", 
               groq_api_key=groq_api_key,
               max_tokens=512,       # Summaries should be brief (512 tokens ≈ 200 words)
               temperature=0.1,      # Lower temperature for factual summaries
               timeout=20            # 20 second timeout)
               )


vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

##############################################################################################


hf_reranker_encoder = "cross-encoder/ms-marco-MiniLM-L-6-v2"

##############################################################################################

hyde_base_embedding =  HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    encode_kwargs = {'normalize_embeddings':True},
)

hyde_embedding = HypotheticalDocumentEmbedder.from_llm(llm = llm, 
                                              base_embeddings = hyde_base_embedding,
                                              prompt_key="sci_fact")



vision_instruction = """
Role: You are a specialized research assistant expert in technical document analysis and data extraction.

Task: Provide a comprehensive, structured description of the attached image from a scientific paper. Your description will be used in a RAG (Retrieval-Augmented Generation) system, so focus on technical keywords and structural relationships.

Instructions:

Identify Category: State if this is a flowchart, architectural diagram, data plot (bar, line, scatter), table, or photographic figure.

Core Component Extraction:

Text & Labels: Transcribe all visible text, including axes, legends, node labels, and captions.

Data Points: If a graph, estimate key values or trends (e.g., "Accuracy peaks at 92% when X=50").

Connectivity: For diagrams, describe the flow (e.g., "Component A feeds into Component B via an embedding layer").

Contextual Significance: Based on the labels, explain the likely purpose of this figure (e.g., "This figure compares the latency vs. throughput of Llama-4 vs. previous generations").

Technical Granularity: Describe visual cues like line styles (dashed vs. solid), color coding, and mathematical symbols.

Constraint: Do not use flowery language. Be dense, technical, and objective. Use Markdown for clarity."""
