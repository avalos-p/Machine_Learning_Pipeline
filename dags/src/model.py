import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from lightgbm import LGBMClassifier
import pickle


class FraudDetectionModel:
    """
    Clase para detección de fraude en tarjetas de crédito con énfasis en recall
    y manejo de costos asimétricos de error.
    """
    
    def __init__(self, threshold=0.5, cost_ratio=10):
        """
        Inicializa el modelo de detección de fraude.
        
        Args:
            threshold (float): Umbral de decisión para clasificar como fraude
            cost_ratio (int): Ratio de costo de falsos negativos vs falsos positivos
        """
        self.model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        )
        self.threshold = threshold
        self.cost_ratio = cost_ratio
        self.is_fitted = False
    
    def fit(self, X, y, validation_size=0.2):
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X: Features preprocesados
            y: Variable objetivo (0: normal, 1: fraude)
            validation_size: Tamaño de la partición de validación
        
        Returns:
            dict: Métricas de rendimiento
        """
        # Separar datos en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, stratify=y, random_state=42
        )
        
        # Entrenamiento del modelo
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluación en conjunto de validación
        y_prob = self.model.predict_proba(X_val)[:, 1]
        
        # Calcular métricas
        metrics = self._calculate_metrics(y_val, y_prob)
        
        # Encontrar el threshold óptimo por costo
        self.threshold = self._find_optimal_threshold(y_val, y_prob)
        
        return metrics
    
    def predict(self, X):
        """
        Realiza predicciones binarias usando el umbral configurado.
        
        Args:
            X: Features preprocesados
        
        Returns:
            numpy.ndarray: Predicciones binarias (0: normal, 1: fraude)
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        y_prob = self.model.predict_proba(X)[:, 1]
        return (y_prob >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        """
        Predice probabilidades de fraude.
        
        Args:
            X: Features preprocesados
        
        Returns:
            numpy.ndarray: Probabilidades de fraude
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        return self.model.predict_proba(X)[:, 1]
    
    def analyze_transaction(self, transaction):
        """
        Analiza una transacción individual (modo streaming).
        
        Args:
            transaction: Array o DataFrame con una sola transacción
        
        Returns:
            dict: Resultado del análisis con predicción y probabilidad
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de analizar")
        
        # Convertir a formato correcto si es necesario
        if isinstance(transaction, pd.DataFrame):
            X = transaction.values
        elif isinstance(transaction, np.ndarray) and transaction.ndim == 1:
            X = transaction.reshape(1, -1)
        else:
            X = transaction
        
        # Obtener probabilidad y predicción binaria
        prob = self.model.predict_proba(X)[0, 1]
        prediction = 1 if prob >= self.threshold else 0
        
        return {
            "prediction": prediction,
            "probability": prob,
            "threshold": self.threshold
        }
    
    def evaluate(self, X, y):
        """
        Evalúa el rendimiento del modelo en un conjunto de datos.
        
        Args:
            X: Features preprocesados
            y: Variable objetivo (0: normal, 1: fraude)
        
        Returns:
            dict: Métricas de rendimiento
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de evaluar")
        
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        
        return self._calculate_metrics(y, y_prob)
    
    def set_threshold(self, threshold):
        """
        Ajusta el umbral de decisión para la clasificación.
        
        Args:
            threshold (float): Nuevo umbral entre 0 y 1
        """
        if threshold < 0 or threshold > 1:
            raise ValueError("El umbral debe estar entre 0 y 1")
        
        self.threshold = threshold
    
    def set_cost_ratio(self, cost_ratio):
        """
        Ajusta el ratio de costo entre falsos negativos y falsos positivos.
        
        Args:
            cost_ratio (float): Ratio de costos FN/FP
        """
        if cost_ratio <= 0:
            raise ValueError("El ratio de costo debe ser positivo")
        
        self.cost_ratio = cost_ratio
    
    def _calculate_metrics(self, y_true, y_prob):
        """
        Calcula métricas de evaluación relevantes.
        
        Args:
            y_true: Valores reales
            y_prob: Probabilidades predichas
        
        Returns:
            dict: Métricas calculadas
        """
        # Calcular predicciones binarias con el umbral actual
        y_pred = (y_prob >= self.threshold).astype(int)
        
        # Calcular matriz de confusión
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        # Calcular métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Costo asimétrico
        cost = fp + (self.cost_ratio * fn)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_prob)
        
        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        
        return {
            "threshold": self.threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "asymmetric_cost": cost,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }
    
    def _find_optimal_threshold(self, y_true, y_prob):
        """
        Encuentra el umbral óptimo basado en el costo asimétrico.
        
        Args:
            y_true: Valores reales
            y_prob: Probabilidades predichas
        
        Returns:
            float: Umbral óptimo
        """
        # Probar diferentes umbrales
        thresholds = np.linspace(0.01, 0.99, 99)
        min_cost = float('inf')
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calcular falsos positivos y falsos negativos
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # Calcular costo asimétrico
            cost = fp + (self.cost_ratio * fn)
            
            if cost < min_cost:
                min_cost = cost
                best_threshold = threshold
        
        return best_threshold
    
    def save_model(self, filepath):
        """
        Guarda el modelo entrenado en disco.
        
        Args:
            filepath (str): Ruta donde guardar el modelo
        """
        if not self.is_fitted:
            raise ValueError("No se puede guardar un modelo que no ha sido entrenado")
        
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'cost_ratio': self.cost_ratio,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath (str): Ruta del modelo guardado
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.threshold = model_data['threshold']
        self.cost_ratio = model_data['cost_ratio']
        self.is_fitted = model_data['is_fitted']


# Ejemplo de uso:
if __name__ == "__main__":
    # Crear instancia del modelo
    fraud_model = FraudDetectionModel(threshold=0.3, cost_ratio=10)
    
    # Supongamos que ya tenemos X e y preprocesados
    # X_train, y_train = ...
    
    # Entrenar el modelo
    # metrics = fraud_model.fit(X_train, y_train)
    # print(f"Métricas de entrenamiento: {metrics}")
    
    # Evaluar en datos de prueba
    # metrics_test = fraud_model.evaluate(X_test, y_test)
    # print(f"Métricas de prueba: {metrics_test}")
    
    # Ajustar umbral si es necesario
    # fraud_model.set_threshold(0.2)
    
    # Procesar una transacción individual
    # result = fraud_model.analyze_transaction(transaction)
    # print(f"Resultado del análisis: {result}")