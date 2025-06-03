from datetime import datetime, timezone

class ImportanceScorer:
    def initialize(self) -> float:
        """Initialize importance score for a new memory."""
        return 1.0
    
    def decay(self, current_score: float) -> float:
        """Apply decay to importance score."""
        decay_rate = 0.1
        return max(0.0, current_score * (1 - decay_rate))

class RecencyScorer:
    def initialize(self) -> float:
        """Initialize recency score for a new memory."""
        return 1.0
    
    def decay(self, current_score: float) -> float:
        """Apply decay to recency score."""
        decay_rate = 0.05
        return max(0.0, current_score * (1 - decay_rate))