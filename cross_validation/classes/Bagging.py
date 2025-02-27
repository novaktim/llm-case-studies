import numpy as np

class BaggingClassifier:
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        """Train both models on the input data"""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions by majority voting from both models"""
        predictions = np.array([model.predict(X) for model in self.models])
        # Use majority voting to get final class
        final_predictions = []
        for i in range(predictions.shape[1]):
            values, counts = np.unique(predictions[:, i], return_counts=True)
            final_predictions.append(values[np.argmax(counts)])
        return np.array(final_predictions)
    
class BaggingRegressor:
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        """Train both models on the input data"""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions by averaging predictions from both models"""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)