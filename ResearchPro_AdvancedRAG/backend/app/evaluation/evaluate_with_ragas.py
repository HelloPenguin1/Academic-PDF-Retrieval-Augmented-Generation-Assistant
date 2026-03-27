"""
RAGAS Evaluation Script
This script evaluates the RAG pipeline responses using RAGAS metrics.

Prerequisites:
1. Install RAGAS: pip install ragas langchain-openai
2. Run run_ragas_evaluation.py first to generate rag_results.json

Usage:
python -m backend.app.evaluation.evaluate_with_ragas
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root to path to allow importing from config
# Hierarchy: ResearchPro_AdvancedRAG/backend/app/evaluation/evaluate_with_ragas.py
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent # Points to ResearchPro_AdvancedRAG
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)
from datasets import Dataset
from config.config import llm_summarize, hf_embeddings


def load_rag_results(results_path=None):
    """Load the RAG results from JSON file."""
    if results_path is None:
        # Try a few common locations
        possible_paths = [
            "backend/app/evaluation/rag_results.json",
            "ResearchPro_AdvancedRAG/backend/app/evaluation/rag_results.json",
            os.path.join(os.path.dirname(__file__), "rag_results.json")
        ]
        for p in possible_paths:
            if os.path.exists(p):
                results_path = p
                break
    
    if not results_path or not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Results file not found. Checked: {possible_paths}\n"
            "Please run 'python -m backend.app.evaluation.run_ragas_evaluation' first."
        )
    
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✅ Loaded {results['metadata']['num_questions']} questions")
    return results['data']


def evaluate_with_ragas(test_data):
    """
    Evaluate the RAG responses using RAGAS metrics.
    
    Args:
        test_data: Dictionary with questions, answers, contexts, and ground_truth
        
    Returns:
        RAGAS evaluation results
    """
    print("\n" + "=" * 80)
    print("RUNNING RAGAS EVALUATION")
    print("=" * 80)
    
    # Initialize evaluator LLM
    # Using Llama 3.3 70B for better reasoning and higher token limits
    print("\n🤖 Initializing evaluator LLM (Groq Llama 3.3 70B)")
    evaluator_llm = llm_summarize
    evaluator_llm.max_tokens = 4096 # Increase for complex evaluation
    
    evaluator_embeddings = hf_embeddings
    print("✅ Evaluator ready")
    
    # Initialize Metrics with explicit LLM and Embeddings
    print("📏 Initializing metrics...")
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextRecall(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm)
    ]
    
    # Convert to HuggingFace Dataset format
    print("\n📊 Converting data to Dataset format...")
    dataset = Dataset.from_dict(test_data)
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Run evaluation
    print("\n⚡ Running RAGAS evaluation (this may take a few minutes)...")
    print("Metrics: Faithfulness, Answer Relevancy, Context Recall, Context Precision")
    
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    
    print("\n✅ Evaluation complete!")
    return result


def print_results(result):
    """Print formatted evaluation results."""
    print("\n" + "=" * 80)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 80)
    
    if not result:
        print("\n❌ No results to display. Evaluation may have failed.")
        return

    # Overall scores
    print("\n📊 Overall Scores:")
    print("─" * 80)
    
    metrics = {
        'faithfulness': ('Faithfulness', '🎯'),
        'answer_relevancy': ('Answer Relevancy', '🎪'),
        'context_recall': ('Context Recall', '📚'),
        'context_precision': ('Context Precision', '🔍'),
    }
    
    for key, (name, emoji) in metrics.items():
        if key in result:
            score = result[key]
            bar = "█" * int(score * 20)
            status = "✅" if score >= 0.75 else "⚠️" if score >= 0.6 else "❌"
            print(f"{emoji} {name:20s}: {score:.4f} {bar} {status}")
    
    print("\n" + "─" * 80)
    print("Score Interpretation:")
    print("  ✅ > 0.75: Good")
    print("  ⚠️  0.60-0.75: Needs improvement")
    print("  ❌ < 0.60: Poor")
    print("=" * 80)


def analyze_per_question(result):
    """Analyze results per question to identify weak spots."""
    print("\n" + "=" * 80)
    print("PER-QUESTION ANALYSIS")
    print("=" * 80)
    
    # Convert to pandas for easier analysis
    df = result.to_pandas()
    
    # Find questions with low scores
    print("\n⚠️  Questions with Low Faithfulness (< 0.7):")
    low_faith = df[df['faithfulness'] < 0.7]
    if len(low_faith) > 0:
        for idx, row in low_faith.iterrows():
            print(f"\n  Q{idx + 1}: {row['question'][:80]}...")
            print(f"       Faithfulness: {row['faithfulness']:.3f}")
    else:
        print("  ✅ All questions have good faithfulness!")
    
    print("\n⚠️  Questions with Low Answer Relevancy (< 0.7):")
    low_rel = df[df['answer_relevancy'] < 0.7]
    if len(low_rel) > 0:
        for idx, row in low_rel.iterrows():
            print(f"\n  Q{idx + 1}: {row['question'][:80]}...")
            print(f"       Answer Relevancy: {row['answer_relevancy']:.3f}")
    else:
        print("  ✅ All questions have good answer relevancy!")
    
    print("\n" + "=" * 80)


def save_evaluation_report(result, output_path="backend/app/evaluation/ragas_report.json"):
    """Save detailed evaluation report."""
    print(f"\n💾 Saving evaluation report to: {output_path}")
    
    # Convert to pandas and then to dict for JSON serialization
    df = result.to_pandas()
    
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(df),
            "paper": "Vision Transformer (ViT)"
        },
        "overall_scores": {
            "faithfulness": float(result.get('faithfulness', 0)),
            "answer_relevancy": float(result.get('answer_relevancy', 0)),
            "context_recall": float(result.get('context_recall', 0)),
            "context_precision": float(result.get('context_precision', 0)),
        },
        "per_question_scores": df.to_dict(orient='records')
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Report saved successfully")
    return output_path


def main():
    """Main execution function."""
    try:
        # Step 1: Load RAG results
        test_data = load_rag_results()
        
        # Step 2: Run RAGAS evaluation
        result = evaluate_with_ragas(test_data)
        
        # Step 3: Print results
        print_results(result)
        
        # Step 4: Analyze per question
        analyze_per_question(result)
        
        # Step 5: Save report
        report_path = save_evaluation_report(result)
        
        print("\n✅ EVALUATION COMPLETE!")
        print(f"\nDetailed report saved to: {report_path}")
        print("\nNext steps:")
        print("1. Review the scores and identify areas for improvement")
        print("2. Adjust your RAG pipeline (prompts, retrieval, reranking)")
        print("3. Re-run evaluation to measure improvements")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
