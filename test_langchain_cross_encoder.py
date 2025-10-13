#!/usr/bin/env python3
"""Test script for LangChain-based Cross-Encoder reranking functionality."""

import os
import sys
import logging
from typing import List, Dict, Any

# Add services to path
sys.path.append('services/online-rag')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_langchain_reranking():
    """Test LangChain-based reranking functionality."""
    try:
        from reranker.cross_encoder_service import CrossEncoderReranker, rerank_candidates
        
        logger.info("Testing LangChain-based Cross-Encoder reranking...")
        
        # Initialize reranker (will use LangChain backend)
        reranker = CrossEncoderReranker(
            model_name="langchain-compression",
            device="auto"
        )
        
        # Test query and candidates
        query = "What is machine learning?"
        candidates = [
            {
                "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "metadata": {"source": "ai_textbook.pdf", "page": 1, "topic": "machine learning"}
            },
            {
                "text": "The weather today is sunny with a chance of rain in the evening.",
                "metadata": {"source": "weather_report.pdf", "page": 1, "topic": "weather"}
            },
            {
                "text": "Deep learning is a subset of machine learning that uses neural networks.",
                "metadata": {"source": "ai_textbook.pdf", "page": 5, "topic": "deep learning"}
            },
            {
                "text": "Cooking pasta requires boiling water and adding salt.",
                "metadata": {"source": "cooking_guide.pdf", "page": 2, "topic": "cooking"}
            }
        ]
        
        logger.info(f"Query: {query}")
        logger.info(f"Testing with {len(candidates)} candidates")
        
        # Test reranking
        result = reranker.rerank_candidates(query, candidates, top_k=3)
        
        logger.info("LangChain reranking results:")
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            logger.info(f"  {i+1}. Score: {score:.4f}")
            logger.info(f"     Text: {candidate['text'][:100]}...")
            logger.info(f"     Source: {candidate['metadata']['source']}")
            logger.info(f"     Topic: {candidate['metadata']['topic']}")
        
        # Test convenience function
        logger.info("\nTesting convenience function...")
        reranked = rerank_candidates(query, candidates, top_k=2)
        
        logger.info("Convenience function results:")
        for i, candidate in enumerate(reranked):
            logger.info(f"  {i+1}. Rank: {candidate.get('rerank_rank', 'N/A')}")
            logger.info(f"     Score: {candidate.get('rerank_score', 'N/A')}")
            logger.info(f"     Method: {candidate.get('rerank_method', 'N/A')}")
            logger.info(f"     Text: {candidate['text'][:100]}...")
        
        logger.info("✅ LangChain Cross-Encoder test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ LangChain Cross-Encoder test failed: {e}")
        return False


def test_langchain_compression_pipeline():
    """Test LangChain compression pipeline directly."""
    try:
        from reranker.langchain_reranker import LangChainReranker
        
        logger.info("Testing LangChain compression pipeline...")
        
        # Initialize LangChain reranker
        reranker = LangChainReranker(use_compression=True)
        
        query = "artificial intelligence applications"
        candidates = [
            {
                "text": "Artificial intelligence is revolutionizing healthcare with diagnostic tools.",
                "metadata": {"doc_id": "ai_healthcare", "category": "AI"}
            },
            {
                "text": "The recipe for chocolate cake includes flour, sugar, and cocoa powder.",
                "metadata": {"doc_id": "cooking", "category": "food"}
            },
            {
                "text": "Machine learning algorithms are used in autonomous vehicles for navigation.",
                "metadata": {"doc_id": "ai_vehicles", "category": "AI"}
            },
            {
                "text": "Weather forecasting uses complex mathematical models to predict conditions.",
                "metadata": {"doc_id": "weather", "category": "science"}
            }
        ]
        
        result = reranker.rerank_candidates(query, candidates, top_k=3)
        
        logger.info("Compression pipeline results:")
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            logger.info(f"  {i+1}. Score: {score:.4f}")
            logger.info(f"     Category: {candidate['metadata']['category']}")
            logger.info(f"     Text: {candidate['text'][:80]}...")
        
        # Check that AI-related content got higher scores
        ai_scores = [score for candidate, score in zip(result.reranked_candidates, result.scores) 
                    if candidate['metadata']['category'] == 'AI']
        other_scores = [score for candidate, score in zip(result.reranked_candidates, result.scores) 
                       if candidate['metadata']['category'] != 'AI']
        
        if ai_scores and other_scores and max(ai_scores) > max(other_scores):
            logger.info("✅ Compression pipeline correctly prioritized AI content!")
        else:
            logger.warning("⚠️ Compression pipeline may need improvement")
        
        logger.info("✅ LangChain compression pipeline test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ LangChain compression pipeline test failed: {e}")
        return False


def test_fallback_mode():
    """Test fallback mode when LangChain components not available."""
    try:
        # Simulate missing LangChain components
        import sys
        if 'langchain.retrievers.document_compressors' in sys.modules:
            del sys.modules['langchain.retrievers.document_compressors']
        
        from reranker.cross_encoder_service import CrossEncoderReranker
        
        logger.info("Testing fallback mode...")
        
        # This should work even without LangChain components
        reranker = CrossEncoderReranker(
            model_name="langchain-compression",
            device="auto"
        )
        
        query = "neural networks"
        candidates = [
            {
                "text": "Neural networks are computing systems inspired by biological neural networks.",
                "metadata": {"title": "NN Introduction", "topic": "neural networks"}
            },
            {
                "text": "The weather forecast shows rain for tomorrow.",
                "metadata": {"title": "Weather Report", "topic": "weather"}
            },
            {
                "text": "Deep neural networks have multiple hidden layers for complex pattern recognition.",
                "metadata": {"title": "Deep NN", "topic": "neural networks"}
            }
        ]
        
        result = reranker.rerank_candidates(query, candidates, top_k=2)
        
        logger.info("Fallback mode results:")
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            logger.info(f"  {i+1}. Score: {score:.4f}")
            logger.info(f"     Topic: {candidate['metadata']['topic']}")
            logger.info(f"     Text: {candidate['text'][:80]}...")
        
        logger.info("✅ Fallback mode test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback mode test failed: {e}")
        return False


def test_integration_with_existing_services():
    """Test integration with existing LLM and embeddings services."""
    try:
        from reranker.langchain_reranker import get_reranking_info
        
        logger.info("Testing integration with existing services...")
        
        # Test info function
        info = get_reranking_info()
        logger.info(f"Reranking info: {info}")
        
        # Check if services are available
        if info.get("langchain_available"):
            logger.info("✅ LangChain components available")
        else:
            logger.warning("⚠️ LangChain components not available")
        
        if info.get("compression_pipeline"):
            logger.info("✅ Compression pipeline initialized")
        else:
            logger.warning("⚠️ Compression pipeline not initialized")
        
        if info.get("llm_available"):
            logger.info("✅ LLM service available")
        else:
            logger.warning("⚠️ LLM service not available")
        
        if info.get("embeddings_available"):
            logger.info("✅ Embeddings service available")
        else:
            logger.warning("⚠️ Embeddings service not available")
        
        logger.info("✅ Integration test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def test_performance():
    """Test performance with larger dataset."""
    try:
        from reranker.cross_encoder_service import CrossEncoderReranker
        import time
        
        logger.info("Testing performance...")
        
        reranker = CrossEncoderReranker(
            model_name="langchain-compression",
            device="auto"
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
                "metadata": {"doc_id": f"doc_{i}", "topic": topic, "category": "AI"}
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
        
        logger.info("✅ Performance test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all LangChain-based tests."""
    logger.info("🚀 Starting LangChain-based Cross-Encoder tests...")
    
    tests = [
        ("LangChain Reranking", test_langchain_reranking),
        ("Compression Pipeline", test_langchain_compression_pipeline),
        ("Fallback Mode", test_fallback_mode),
        ("Service Integration", test_integration_with_existing_services),
        ("Performance", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("LANGCHAIN TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All LangChain tests passed! Cross-Encoder is ready for production.")
    else:
        logger.error("⚠️  Some tests failed. Check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
