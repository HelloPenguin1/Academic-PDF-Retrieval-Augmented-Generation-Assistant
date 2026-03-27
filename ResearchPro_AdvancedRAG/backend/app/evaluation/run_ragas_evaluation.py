"""
RAGAS Evaluation Script for RAG Pipeline
This script runs your RAG pipeline on test questions and evaluates performance using RAGAS metrics.

Usage:
1. Make sure your PDF is uploaded and the RAG pipeline is initialized
2. Run: python -m backend.app.evaluation.run_ragas_evaluation
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import llm, hf_reranker_encoder
from backend.app.services.document_service import DocumentProcessor
from backend.app.services.rag_service import RAG_Pipeline
from backend.app.services.reranker import ReRanker_Model
from backend.utils.session_manager import SessionManager
import json
from datetime import datetime


# Test data for Vision Transformer paper
test_data = {
    "question": [
        "What is the fundamental architectural difference between the Vision Transformer (ViT) and traditional Convolutional Neural Networks (CNNs)?",
        "How does the Vision Transformer process a 2D image into an input suitable for the Transformer encoder?",
        "What specific inductive biases are present in CNNs but lacking in the Vision Transformer?",
        "How does dataset size influence the performance comparison between ViT and ResNets (BiT)?",
        "What top-1 accuracy did the best ViT model (ViT-H/14) achieve on the ImageNet dataset?",
        "What is the function of the [class] token in the ViT architecture?",
        "How does the pre-training computational cost of ViT compare to state-of-the-art ResNets like BiT?",
        "What conclusion did the authors reach regarding the use of 1D versus 2D positional embeddings?",
        "Describe the 'Hybrid Architecture' mentioned in the paper.",
        "What performance did the ViT-B/16 model achieve using self-supervised pre-training on ImageNet?"
    ],
    "answer": [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    ],
    "contexts": [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ],
    "ground_truth": [
        "ViT is a pure Transformer applied directly to sequences of image patches, reducing reliance on CNNs which use convolution layers with baked-in inductive biases.",
        "The image is split into fixed-size patches which are flattened and linearly projected into embeddings, then position embeddings are added before feeding the sequence to the encoder.",
        "The Vision Transformer lacks image-specific inductive biases such as translation equivariance and locality (except in MLP layers), which are inherent to CNNs throughout the whole model.",
        "ViT yields modest accuracies below ResNets on mid-sized datasets like ImageNet, but outperforms ResNets when trained on larger datasets (14M-300M images), showing that large scale training trumps inductive bias.",
        "The best model, ViT-H/14, achieved an accuracy of 88.55% on ImageNet.",
        "Similar to BERT, a learnable [class] token is prepended to the sequence of embedded patches, and its state at the output of the Transformer encoder serves as the image representation used for classification.",
        "ViT requires substantially fewer computational resources; for instance, ViT-L/16 took 0.68k TPUv3-core-days to pre-train on JFT-300M, while BiT-L took 9.9k.",
        "The authors observed no significant performance gains from using advanced 2D-aware position embeddings compared to standard learnable 1D position embeddings.",
        "In the hybrid architecture, the input sequence to the Transformer is formed from patches extracted from the feature maps of a CNN, rather than raw image patches.",
        "With self-supervised pre-training, the ViT-B/16 model achieved 79.9% accuracy on ImageNet, which is a significant improvement over training from scratch but still behind supervised pre-training."
    ]
}


def initialize_rag_pipeline(pdf_path: str):
    """
    Initialize the RAG pipeline with a PDF document.
    
    Args:
        pdf_path: Path to the Vision Transformer paper PDF
        
    Returns:
        Tuple of (rag_pipeline, document_processor)
    """
    print("INITIALIZING RAG PIPELINE")

    # Instantiate components
    document_processor = DocumentProcessor()
    rag_pipeline = RAG_Pipeline(llm)
    reranker = ReRanker_Model(hf_reranker_encoder)
    session_manager = SessionManager()
    
    print(f"\n📄 Loading PDF: {pdf_path}")
    
    # Load and process document
    docs = document_processor.load_and_process_pdf(pdf_path)
    print(f"✅ Processed {len(docs)} document chunks")
    
    # Create retriever
    print("\n🔍 Creating hybrid retriever...")
    hybrid_retriever = document_processor.create_retriever(docs)
    print("✅ Hybrid retriever created")
    
    # Create compression retriever with reranker
    print("🎯 Creating compression retriever with reranker...")
    compression_retriever = reranker.create_compression_retriever(hybrid_retriever)
    rag_pipeline.set_compression_retriever(compression_retriever)
    print("✅ Compression retriever created")
    
    # Update vectorstore
    if document_processor.vectorstore:
        rag_pipeline.update_vectorstore(document_processor.vectorstore)
        print("✅ Vectorstore updated")
    else:
        raise Exception("Vectorstore initialization failed")
    
    # Create RAG chain
    print("\n⛓️  Creating conversational RAG chain...")
    rag_chain = rag_pipeline.create_rag_chain(compression_retriever)
    conversational_chain = rag_pipeline.create_conversational_chain(
        rag_chain, 
        session_manager.get_session_history
    )
    rag_pipeline.conversational_rag = conversational_chain
    print("✅ Conversational RAG chain created")
    
    print("\n" + "=" * 80)
    print("RAG PIPELINE READY")
    print("=" * 80)
    
    return rag_pipeline, document_processor


def run_rag_on_questions(rag_pipeline, questions, session_id="ragas_eval_session"):
    """
    Run the RAG pipeline on all test questions and collect answers + contexts.
    
    Args:
        rag_pipeline: Initialized RAG pipeline
        questions: List of questions to ask
        session_id: Session ID for conversation history
        
    Returns:
        Tuple of (answers, contexts_list)
    """
    print("\n" + "=" * 80)
    print("RUNNING RAG PIPELINE ON TEST QUESTIONS")
    print("=" * 80)
    
    answers = []
    contexts_list = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 80}")
        print(f"Question {i}/{len(questions)}")
        print(f"{'─' * 80}")
        print(f"Q: {question[:100]}...")
        
        try:
            # Get retrieved documents (contexts)
            retrieved_docs = rag_pipeline.compression_retriever.get_relevant_documents(question)
            
            # Extract context text from top 3 documents
            contexts = [doc.page_content for doc in retrieved_docs[:3]]
            
            # Get answer from RAG pipeline
            answer = rag_pipeline.query(question, session_id)
            
            # Store results
            answers.append(answer)
            contexts_list.append(contexts)
            
            print(f"Answer generated ({len(answer)} chars)")
            print(f"Retrieved {len(contexts)} context chunks")
            print(f"Preview: {answer[:150]}...")
            
        except Exception as e:
            print(f"❌ Error processing question: {str(e)}")
            answers.append(f"Error: {str(e)}")
            contexts_list.append([])
    
    print("\n" + "=" * 80)
    print("RAG PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    
    return answers, contexts_list


def save_results(test_data, output_path="backend/app/evaluation/rag_results.json"):
    """
    Save the populated test data to a JSON file.
    
    Args:
        test_data: Dictionary with questions, answers, contexts, and ground truth
        output_path: Path to save the results
    """
    print(f"\n💾 Saving results to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add metadata
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(test_data["question"]),
            "paper": "Vision Transformer (ViT)"
        },
        "data": test_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved successfully")
    return output_path


def print_summary(test_data):
    """Print a summary of the results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    num_questions = len(test_data["question"])
    num_answers = sum(1 for a in test_data["answer"] if a and not a.startswith("Error"))
    num_contexts = sum(1 for c in test_data["contexts"] if c)
    
    print(f"Total Questions: {num_questions}")
    print(f"Successful Answers: {num_answers}/{num_questions}")
    print(f"Questions with Contexts: {num_contexts}/{num_questions}")
    
    avg_answer_length = sum(len(a) for a in test_data["answer"]) / num_questions if num_questions > 0 else 0
    avg_contexts_per_q = sum(len(c) for c in test_data["contexts"]) / num_questions if num_questions > 0 else 0
    
    print(f"Average Answer Length: {avg_answer_length:.0f} characters")
    print(f"Average Contexts per Question: {avg_contexts_per_q:.1f}")
    
    print("\n" + "=" * 80)


def main():
    """
    Main execution function.
    
    IMPORTANT: Update the PDF_PATH variable below to point to your Vision Transformer paper.
    """
    
    # ============================================================================
    # CONFIGURATION: Update this path to your Vision Transformer PDF
    # ============================================================================
    PDF_PATH = r"C:\dev\Projects\ResearchPro\ResearchPro_AdvancedRAG\backend\app\evaluation\Paper.pdf"
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"\n❌ Error: PDF file not found at: {PDF_PATH}")
        print("Please check the path and try again.")
        print("\nMake sure the PDF is located at: backend/app/evaluation/Paper.pdf")
        return
    
    try:
        # Step 1: Initialize RAG pipeline
        rag_pipeline, document_processor = initialize_rag_pipeline(PDF_PATH)
        
        # Step 2: Run RAG on all questions
        answers, contexts_list = run_rag_on_questions(rag_pipeline, test_data["question"])
        
        # Step 3: Populate test data
        test_data["answer"] = answers
        test_data["contexts"] = contexts_list
        
        # Step 4: Save results
        output_path = save_results(test_data)
        
        # Step 5: Print summary
        print_summary(test_data)
        
        print("\n✅ SUCCESS! Your test data is now ready for RAGAS evaluation.")
        print(f"\nNext steps:")
        print(f"1. Review the results in: {output_path}")
        print(f"2. Run RAGAS evaluation using: python -m backend.app.evaluation.evaluate_with_ragas")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
