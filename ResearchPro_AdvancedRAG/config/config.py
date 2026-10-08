import os

import torch
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

load_dotenv()


## for increased efficiency
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)


groq_api_key = os.getenv("GROQ_API_KEY")

##############################################################################################

# Tokenizer for Docling's Hybrid Chunking


hf_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", local_files_only=False
)
chunking_tokenizer = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer,
)


hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

# Main LLM for final answer generation (large context window, best reasoning)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key,
    max_tokens=2048,  # Limit response to 2048 tokens
    temperature=0,  # Add consistency
    timeout=30,  # 30 second timeout)
)

# Lightweight LLM for query reformulation (reduces rate limit pressure)
llm_reformulate = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    max_tokens=256,  # Short reformulated queries
    temperature=0.1,
    timeout=10,
)

llm_summarize = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    max_tokens=512,  # Summaries should be brief (512 tokens ≈ 200 words)
    temperature=0.1,  # Lower temperature for factual summaries
    timeout=20,  # 20 second timeout)
)

hf_reranker_encoder = "cross-encoder/ms-marco-MiniLM-L-6-v2"
