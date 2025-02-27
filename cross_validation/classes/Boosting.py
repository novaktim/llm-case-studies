import numpy as np


class BoostingClassifier:
    def __init__(self, model, n_estimators=50):
        self.model = model
        self.n_estimators = n_estimators
        self.classifiers = []
        self.alpha_values = []

    def fit(self, train_data, train_labels):
        n_samples = train_data.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            clf = self.model.__class__(**self.model.get_params())
            clf.fit(train_data, train_labels, sample_weight=weights)
            predictions = clf.predict(train_data)

            misclassified = predictions != train_labels
            error = np.sum(weights[misclassified]) / np.sum(weights)

            if error > 0.5:
                break
            elif error == 0:
                alpha = 1
            else:
                alpha = 0.5 * np.log((1 - error) / error)

            weights *= np.exp(alpha * misclassified * 2)
            weights /= np.sum(weights)

            self.classifiers.append(clf)
            self.alpha_values.append(alpha)

    def predict(self, test_data):
        final_predictions = np.zeros(test_data.shape[0])

        for clf, alpha in zip(self.classifiers, self.alpha_values):
            final_predictions += alpha * clf.predict(test_data)

        return np.sign(final_predictions)


class BoostingRegressor:
    def __init__(self, model, n_estimators=50):
        self.model = model
        self.n_estimators = n_estimators
        self.regressors = []
        self.alpha_values = []

    def fit(self, train_data, train_labels):
        n_samples = train_data.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            reg = self.model.__class__(**self.model.get_params())
            reg.fit(train_data, train_labels, sample_weight=weights)
            predictions = reg.predict(train_data)

            residuals = train_labels - predictions
            error = np.sum(weights * (residuals ** 2)) / np.sum(weights)

            if error == 0:
                alpha = 1
            else:
                alpha = 0.5 * np.log((1 - error) / error)

            weights *= np.exp(alpha * residuals * 2)
            weights /= np.sum(weights)

            self.regressors.append(reg)
            self.alpha_values.append(alpha)

    def predict(self, test_data):
        final_predictions = np.zeros(test_data.shape[0])

        for reg, alpha in zip(self.regressors, self.alpha_values):
            final_predictions += alpha * reg.predict(test_data)

        return final_predictions