"""
LangSmith + RAGAS Evaluation Script
This script runs RAGAS metrics and logs all traces, reasoning, and costs to LangSmith.
"""
import sys
import os
from pathlib import Path
import json
from datetime import datetime
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] =os.getenv("OPENAI_API_KEY")

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ragas import evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)

from openai import OpenAI
from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


def load_rag_results(results_path=None):
    """Load the pre-computed RAG results from JSON file."""
    if results_path is None:
        possible_paths = [
            "ResearchPro_AdvancedRAG/backend/app/evaluation/rag_results.json",
            os.path.join(os.path.dirname(__file__), "rag_results2.json")
        ]
        for p in possible_paths:
            if os.path.exists(p):
                results_path = p
                break
    
    if not results_path or not os.path.exists(results_path):
        raise FileNotFoundError("Results file not found. Run your RAG pipeline script first.")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Loaded {results['metadata']['num_questions']} questions from JSON.")
    return results['data']


def evaluate_with_langsmith(test_data):
    """
    Evaluate the RAG responses. Traces will be automatically sent to LangSmith.
    """
    print("\nStarting Evaluation. Traces will be logged to LangSmith...")
    
    openai_client = OpenAI()
    
    # max_tokens increased to prevent JSON truncation errors during evaluation
    evaluator_llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=10000)
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextRecall(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm)
    ]
    
    dataset = Dataset.from_dict(test_data)
    
    config = RunConfig(max_workers=4, timeout=60)
    
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=config
    )
    
    print("Evaluation complete. Check your LangSmith dashboard.")
    return result


def print_results(result):
    """Print formatted evaluation results to the terminal."""
    print("\nEvaluation Results:")
    
    if not result:
        print("No results to display.")
        return

    metrics_map = {
        'faithfulness': 'Faithfulness',
        'answer_relevancy': 'Answer Relevancy',
        'context_recall': 'Context Recall',
        'context_precision': 'Context Precision',
    }
    
    for key, name in metrics_map.items():
        if key in result:
            score = result[key]
            bar = "█" * int(score * 20)
            status = "Pass" if score >= 0.75 else "Warn" if score >= 0.6 else "Fail"
            print(f"{name:20s}: {score:.4f} {bar} [{status}]")


def main():
    try:
        test_data = load_rag_results()
        result = evaluate_with_langsmith(test_data)
        print_results(result)
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")


if __name__ == "__main__":
    main()