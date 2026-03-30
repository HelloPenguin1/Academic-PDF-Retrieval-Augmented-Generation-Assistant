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

from sympy import python

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



test_data = {
    "question": [
        "How does the fundamental approach to processing spatial and structural information in images differ between ResNet and the Vision Transformer (ViT)?",
        "How does the performance of Vision Transformers (ViT) compare to ResNet-based models when pre-trained on mid-sized datasets versus large-scale datasets like JFT-300M?",
        "Compare the pre-training computational costs of Vision Transformers (ViT) against state-of-the-art ResNet architectures to achieve similar or better performance.",
        "How does the Vision Transformer (ViT) adapt the architecture and input mechanism of the original Transformer model proposed in 'Attention Is All You Need' for image classification?",
        "While ResNet successfully utilizes CNNs for image tasks, what limitations of convolutional neural networks did the Vision Transformer attempt to overcome?",
        "How does the application of the self-attention mechanism evolve from its use in the original NLP Transformer to its application in the Vision Transformer (ViT)?",
        "How do 'hybrid' architectures combine the key innovations from both 'Deep Residual Learning for Image Recognition' and the Vision Transformer, and how do they perform?",
        "What is the primary trade-off between using a ResNet and a Vision Transformer (ViT) regarding inductive biases and training data size?",
        "Based on empirical evidence from the sources, under what data availability circumstances should a practitioner choose to use a standard ResNet architecture over a pure Vision Transformer (ViT)?",
        "Both the ResNet authors and the original Transformer authors address the challenge of successfully training deep networks. How do their architectural solutions for passing information through deep layers differ?"
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
        "ResNet utilizes convolutional layers where two-dimensional neighborhood structure, locality, and translation equivariance are baked into each layer throughout the whole model. In contrast, ViT has much less image-specific inductive bias; it splits the image into 2D patches and flattens them into a 1D sequence. In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global, meaning spatial relations between patches have to be learned from scratch.",
        
        "When trained on mid-sized datasets like ImageNet without strong regularization, ViT models yield modest accuracies of a few percentage points below ResNets of comparable size. However, when pre-trained on large datasets like JFT-300M, ViT models overtake ResNets, with the ViT-L/16 model outperforming the ResNet-based BiT-L on all tasks.",
        
        "Vision Transformers perform very favourably, attaining state-of-the-art on most recognition benchmarks at a lower pre-training cost than ResNets. Specifically, ViT-L/16 pre-trained on JFT-300M took 0.68k TPUv3-core-days, which is substantially less than the 9.9k TPUv3-core-days required by the ResNet-based BiT-L, while still achieving better accuracy. Overall, ViT uses approximately 2-4x less compute to attain the same performance as ResNets.",
        
        "ViT follows the original Transformer design as closely as possible. To adapt the original 1D sequence input of the Transformer, ViT reshapes a 2D image into a sequence of flattened 2D patches, treating them the same way as tokens (words) in NLP. Like the original Transformer, ViT uses an encoder consisting of alternating layers of multi-headed self-attention and MLP blocks, and adds standard learnable 1D position embeddings to the patch embeddings to retain positional information.",
        
        "ResNets rely on convolutional neural networks which have baked-in inductive biases like translation equivariance and locality. The Vision Transformer attempts to overcome the reliance on these CNN-specific biases by showing that a pure transformer applied directly to sequences of image patches can perform very well. Additionally, ViT addresses computational efficiency at scale; while classic ResNet-like architectures are computationally expensive, ViT requires substantially fewer computational resources to train while effectively scaling on modern hardware.",
        
        "In the original Transformer, self-attention maps a variable-length sequence of symbol representations (like words) to another sequence, allowing the model to draw global dependencies regardless of distance in the input or output sequences. ViT adapts this by applying the global self-attention mechanism to images by treating flattened image patches as the sequence elements. In ViT, self-attention allows the model to integrate information across the entire image even in the lowest layers, contrasting with the local receptive fields of early CNN layers.",
        
        "Hybrid architectures combine the two approaches by applying the Vision Transformer's patch embedding projection to patches extracted from a CNN feature map (e.g., from a ResNet) instead of raw image patches. During scaling studies, hybrids slightly outperform pure ViT models at small computational budgets, but this performance gap vanishes for larger models.",
        
        "ResNets have strong image-specific inductive biases, such as two-dimensional neighborhood structure, locality, and translation equivariance baked into every layer. ViT lacks these biases, relying mostly on global self-attention. Because of this trade-off, ViT does not generalize well and underperforms comparable ResNets when trained on insufficient amounts of data. However, this trade-off flips with large-scale datasets (like JFT-300M), where large scale training trumps inductive bias and ViT overtakes ResNets.",
        
        "A practitioner should use a ResNet architecture when working with smaller pre-training datasets. Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets. Convolutional inductive biases are useful for smaller datasets, making ResNets the better choice there, whereas ViT is better suited for scenarios where large-scale pre-training data (like 14M to 300M images) is available.",
        
        "The ResNet authors address the degradation problem in deep networks by explicitly reformulating layers to learn residual functions using parameter-free shortcut connections that perform identity mapping. The original Transformer eschews recurrence and convolutions entirely, relying instead on a multi-head self-attention mechanism to draw global dependencies, and it successfully routes information through its deep stacks of encoder and decoder layers using residual connections around each sub-layer combined with layer normalization."
    ]
}


def initialize_rag_pipeline(pdf_paths: list):
    """
    Initialize the RAG pipeline with a PDF document.
    
    Args:
        pdf_paths: List of paths to the Vision Transformer paper PDFs
        
    Returns:
        Tuple of (rag_pipeline, document_processor)
    """
    print("INITIALIZING RAG PIPELINE")

    # Instantiate components
    document_processor = DocumentProcessor()
    rag_pipeline = RAG_Pipeline(llm)
    reranker = ReRanker_Model(hf_reranker_encoder)
    session_manager = SessionManager()
    
    # Load and process all requested PDF documents
    if not pdf_paths:
        raise ValueError("pdf_paths must be a non-empty list of PDF file paths")

    docs = []
    for pdf_path in pdf_paths:
        print(f"\n📄 Loading PDF: {pdf_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        page_docs = document_processor.load_and_process_pdf(pdf_path)
        docs.extend(page_docs)
        print(f"✅ Processed {len(page_docs)} chunks from {os.path.basename(pdf_path)}")

    print(f"\n✅ Processed {len(docs)} total document chunks")
    
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
    print("RUNNING RAG PIPELINE ON TEST QUESTIONS")
    
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
            contexts = [doc.page_content for doc in retrieved_docs[:6]]
            
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


def save_results(test_data, output_path="backend/app/evaluation/multi_doc_results3.json"):
    """
    Save the populated test data to a JSON file.
    
    Args:
        test_data: Dictionary with questions, answers, contexts, and ground truth
        output_path: Path to save the results
    """
    print(f"\nSaving results to: {output_path}")
    
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
    
    print(f"Results saved successfully")
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
  
  
    
    PDF_PATHS = [
        r"C:\dev\Projects\ResearchPro\ResearchPro_AdvancedRAG\backend\app\evaluation\attention.pdf",
        r"C:\dev\Projects\ResearchPro\ResearchPro_AdvancedRAG\backend\app\evaluation\vit.pdf",
        r"C:\dev\Projects\ResearchPro\ResearchPro_AdvancedRAG\backend\app\evaluation\resnet.pdf",
    ]
    
    for path in PDF_PATHS:
        if not os.path.exists(path):
            print(f"\n❌ Error: PDF file not found at: {path}")
            print("Please check the path and try again.")
            return
    
    try:
        # Step 1: Initialize RAG pipeline
        rag_pipeline, document_processor = initialize_rag_pipeline(PDF_PATHS)
        
        # Step 2: Run RAG on all questions
        answers, contexts_list = run_rag_on_questions(rag_pipeline, test_data["question"])
        
        # Step 3: Populate test data
        test_data["answer"] = answers
        test_data["contexts"] = contexts_list
        
        # Step 4: Save results
        output_path = save_results(test_data)
        
        # Step 5: Print summary
        print_summary(test_data)
        
        print(f"1. Review the results in: {output_path}")
        
    except Exception as e:
        print(f" Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
