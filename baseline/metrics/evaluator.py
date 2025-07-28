import json
import os
from datetime import datetime

class LegalAssistantEvaluator:
    def __init__(self, metrics_dir="../metrics"):
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
    def evaluate_response_quality(self, query, response, relevance_score=None):
        """Evaluate the quality of a generated response"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "relevance_score": relevance_score,
            "response_length": len(response),
            "word_count": len(response.split())
        }
        
        return metrics
    
    def save_metrics(self, metrics, filename=None):
        """Save metrics to a JSON file"""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return filepath
    
    def calculate_accuracy(self, predictions, ground_truth):
        """Calculate accuracy for a set of predictions"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
        accuracy = correct / len(predictions)
        
        return accuracy 