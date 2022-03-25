'''Histogram Gradient Boosting Regressor

HGBR implementation comes from scikit-learn.
This class creates a set of HGBR models and provides fit and prediction functions, since the scikit-learn implementation is unable to provide multi-dimensional regression.
'''

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

class HGBR():

    def __init__(self, loss, learning_rate, max_iter, max_depth, regularization, input_scaler, output_scaler):
        
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

        if regularization < 0:
            regularization = 10**regularization

        self.config = {'loss':loss, 'learning_rate':learning_rate,
                       'max_iter':max_iter, 'max_depth':max_depth, 'l2_regularization': regularization} 

    def _create_model(self):
        model = HistGradientBoostingRegressor(**self.config)
        return model

    def fit(self, X, y):
        self.models = []

        if self.output_scaler is not None:
            y_t = self.output_scaler.fit_transform(y)
        else:
            y_t = y
        
        if self.input_scaler is not None:
            X_t = self.input_scaler.fit_transform(X)
        else:
            X_t = X
        
        for t in range(y_t.shape[1]):
            self.models.append(self._create_model().fit(X_t, y_t[:, t]))

    def predict(self, X):

        prediction = np.zeros((X.shape[0], len(self.models)))
        if self.input_scaler is not None:
            X_t = self.input_scaler.transform(X)
        else:
            X_t = X
        
        for i, m in enumerate(self.models):
            prediction[:, i] = m.predict(X_t)
        
        if self.output_scaler is not None:
            prediction = self.output_scaler.inverse_transform(prediction)

        return prediction
        
        
