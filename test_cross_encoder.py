#!/usr/bin/env python3
"""Test script for Cross-Encoder reranking functionality."""

import os
import sys
import logging
from typing import List, Dict, Any

# Add services to path
sys.path.append('services/online-rag')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cross_encoder_basic():
    """Test basic cross-encoder functionality."""
    try:
        from reranker.cross_encoder_service import CrossEncoderReranker, rerank_candidates
        
        logger.info("Testing Cross-Encoder reranking...")
        
        # Initialize reranker
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="cpu",  # Use CPU for testing
            max_length=512,
            batch_size=4
        )
        
        # Test query and candidates
        query = "What is machine learning?"
        candidates = [
            {
                "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "metadata": {"source": "ai_textbook.pdf", "page": 1}
            },
            {
                "text": "The weather today is sunny with a chance of rain in the evening.",
                "metadata": {"source": "weather_report.pdf", "page": 1}
            },
            {
                "text": "Deep learning is a subset of machine learning that uses neural networks.",
                "metadata": {"source": "ai_textbook.pdf", "page": 5}
            },
            {
                "text": "Cooking pasta requires boiling water and adding salt.",
                "metadata": {"source": "cooking_guide.pdf", "page": 2}
            }
        ]
        
        logger.info(f"Query: {query}")
        logger.info(f"Testing with {len(candidates)} candidates")
        
        # Test reranking
        result = reranker.rerank_candidates(query, candidates, top_k=3)
        
        logger.info("Reranking results:")
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            logger.info(f"  {i+1}. Score: {score:.4f}")
            logger.info(f"     Text: {candidate['text'][:100]}...")
            logger.info(f"     Source: {candidate['metadata']['source']}")
        
        # Test convenience function
        logger.info("\nTesting convenience function...")
        reranked = rerank_candidates(query, candidates, top_k=2)
        
        logger.info("Convenience function results:")
        for i, candidate in enumerate(reranked):
            logger.info(f"  {i+1}. Rank: {candidate.get('rerank_rank', 'N/A')}")
            logger.info(f"     Score: {candidate.get('rerank_score', 'N/A')}")
            logger.info(f"     Text: {candidate['text'][:100]}...")
        
        logger.info("✅ Cross-Encoder test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cross-Encoder test failed: {e}")
        return False


def test_reranking_integration():
    """Test integration with mock RAG pipeline."""
    try:
        from reranker.cross_encoder_service import rerank_candidates
        
        logger.info("Testing RAG integration...")
        
        # Simulate RAG pipeline
        query = "How to implement neural networks?"
        
        # Mock retrieved documents (as would come from Qdrant)
        mock_docs = [
            {
                "text": "Neural networks are computing systems inspired by biological neural networks.",
                "metadata": {"doc_id": "ai_basics", "chunk_id": 1, "score": 0.85}
            },
            {
                "text": "To implement a neural network, you need to define layers and activation functions.",
                "metadata": {"doc_id": "ai_basics", "chunk_id": 2, "score": 0.78}
            },
            {
                "text": "The recipe for chocolate cake includes flour, sugar, and cocoa powder.",
                "metadata": {"doc_id": "cooking", "chunk_id": 1, "score": 0.65}
            },
            {
                "text": "PyTorch and TensorFlow are popular frameworks for neural network implementation.",
                "metadata": {"doc_id": "frameworks", "chunk_id": 1, "score": 0.72}
            }
        ]
        
        logger.info(f"Original order (by vector similarity):")
        for i, doc in enumerate(mock_docs):
            logger.info(f"  {i+1}. Score: {doc['metadata']['score']:.2f} - {doc['text'][:60]}...")
        
        # Apply reranking
        reranked_docs = rerank_candidates(query, mock_docs, top_k=3)
        
        logger.info(f"\nAfter reranking:")
        for i, doc in enumerate(reranked_docs):
            rerank_score = doc.get('rerank_score', 'N/A')
            logger.info(f"  {i+1}. Rerank Score: {rerank_score:.4f} - {doc['text'][:60]}...")
        
        logger.info("✅ RAG integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG integration test failed: {e}")
        return False


def test_performance():
    """Test performance with larger dataset."""
    try:
        from reranker.cross_encoder_service import CrossEncoderReranker
        import time
        
        logger.info("Testing performance...")
        
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="cpu",
            batch_size=8
        )
        
        query = "artificial intelligence applications"
        
        # Generate test candidates
        candidates = []
        topics = [
            "machine learning algorithms",
            "natural language processing",
            "computer vision",
            "robotics and automation",
            "deep learning networks",
            "artificial neural networks",
            "data science methods",
            "statistical analysis",
            "pattern recognition",
            "intelligent systems"
        ]
        
        for i, topic in enumerate(topics):
            candidates.append({
                "text": f"This document discusses {topic} and its applications in modern technology.",
                "metadata": {"doc_id": f"doc_{i}", "topic": topic}
            })
        
        # Measure performance
        start_time = time.time()
        result = reranker.rerank_candidates(query, candidates, top_k=5)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        logger.info(f"Processed {len(candidates)} candidates in {processing_time:.2f} seconds")
        logger.info(f"Average time per candidate: {processing_time/len(candidates)*1000:.1f} ms")
        
        logger.info("Top 3 results:")
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates[:3], result.scores[:3])):
            logger.info(f"  {i+1}. Score: {score:.4f} - {candidate['metadata']['topic']}")
        
        logger.info("✅ Performance test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting Cross-Encoder tests...")
    
    tests = [
        ("Basic Functionality", test_cross_encoder_basic),
        ("RAG Integration", test_reranking_integration),
        ("Performance", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Cross-Encoder is ready for production.")
    else:
        logger.error("⚠️  Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
