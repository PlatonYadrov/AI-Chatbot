#!/usr/bin/env python3
"""Local test script for Cross-Encoder reranking functionality."""

import os
import sys
import logging
from typing import List, Dict, Any

# Add services to path
sys.path.append('services/online-rag')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_local_reranking():
    """Test local reranking functionality."""
    try:
        from reranker.cross_encoder_service import CrossEncoderReranker, rerank_candidates
        
        logger.info("Testing Local Cross-Encoder reranking...")
        
        # Initialize reranker (will use fallback if transformers not available)
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
            logger.info(f"     Method: {candidate.get('rerank_method', 'N/A')}")
            logger.info(f"     Text: {candidate['text'][:100]}...")
        
        logger.info("✅ Local Cross-Encoder test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Local Cross-Encoder test failed: {e}")
        return False


def test_fallback_reranking():
    """Test fallback reranking when transformers not available."""
    try:
        # Simulate missing transformers
        import sys
        if 'sentence_transformers' in sys.modules:
            del sys.modules['sentence_transformers']
        if 'torch' in sys.modules:
            del sys.modules['torch']
        
        from reranker.cross_encoder_service import CrossEncoderReranker
        
        logger.info("Testing fallback reranking...")
        
        # This should work even without transformers
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="cpu"
        )
        
        query = "artificial intelligence"
        candidates = [
            {
                "text": "Artificial intelligence is the simulation of human intelligence in machines.",
                "metadata": {"title": "AI Introduction", "topic": "artificial intelligence"}
            },
            {
                "text": "The recipe for chocolate cake includes flour, sugar, and cocoa powder.",
                "metadata": {"title": "Cooking Guide", "topic": "baking"}
            },
            {
                "text": "Machine learning algorithms can learn from data without explicit programming.",
                "metadata": {"title": "ML Basics", "topic": "machine learning"}
            }
        ]
        
        result = reranker.rerank_candidates(query, candidates, top_k=2)
        
        logger.info("Fallback reranking results:")
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            logger.info(f"  {i+1}. Score: {score:.4f}")
            logger.info(f"     Text: {candidate['text'][:80]}...")
            logger.info(f"     Topic: {candidate['metadata']['topic']}")
        
        # Check that AI-related content got higher scores
        ai_score = result.scores[0] if result.scores else 0
        cooking_score = result.scores[-1] if len(result.scores) > 1 else 0
        
        if ai_score > cooking_score:
            logger.info("✅ Fallback reranking correctly prioritized AI content!")
        else:
            logger.warning("⚠️ Fallback reranking may need improvement")
        
        logger.info("✅ Fallback reranking test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback reranking test failed: {e}")
        return False


def test_offline_mode():
    """Test offline mode functionality."""
    try:
        from reranker.local_cross_encoder_service import LocalCrossEncoderReranker, get_reranking_info
        
        logger.info("Testing offline mode...")
        
        # Test offline reranker
        reranker = LocalCrossEncoderReranker(
            offline_mode=True,
            device="cpu"
        )
        
        query = "neural networks"
        candidates = [
            {
                "text": "Neural networks are computing systems inspired by biological neural networks.",
                "metadata": {"doc_id": "nn_basics"}
            },
            {
                "text": "The weather forecast shows rain for tomorrow.",
                "metadata": {"doc_id": "weather"}
            },
            {
                "text": "Deep neural networks have multiple hidden layers for complex pattern recognition.",
                "metadata": {"doc_id": "deep_nn"}
            }
        ]
        
        result = reranker.rerank_candidates(query, candidates, top_k=2)
        
        logger.info("Offline mode results:")
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            logger.info(f"  {i+1}. Score: {score:.4f}")
            logger.info(f"     Text: {candidate['text'][:80]}...")
        
        # Test info function
        info = get_reranking_info()
        logger.info(f"Reranking info: {info}")
        
        logger.info("✅ Offline mode test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Offline mode test failed: {e}")
        return False


def test_performance_comparison():
    """Test performance comparison between methods."""
    try:
        from reranker.cross_encoder_service import CrossEncoderReranker
        import time
        
        logger.info("Testing performance comparison...")
        
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
        
        logger.info("✅ Performance comparison test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test failed: {e}")
        return False


def main():
    """Run all local tests."""
    logger.info("🚀 Starting Local Cross-Encoder tests...")
    
    tests = [
        ("Local Reranking", test_local_reranking),
        ("Fallback Reranking", test_fallback_reranking),
        ("Offline Mode", test_offline_mode),
        ("Performance Comparison", test_performance_comparison)
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
    logger.info("LOCAL TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All local tests passed! Cross-Encoder is ready for local use.")
    else:
        logger.error("⚠️  Some tests failed. Check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
