from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class MLModelTrainer:
    def __init__(self, model_type="LogisticRegression", model_params=None):
        """
        Инициализация класса для обучения моделей.

        Parameters:
        -----------
        model_type : str
            Тип модели для обучения ('LogisticRegression', 'RandomForest').
        model_params : dict, optional
            Гиперпараметры для модели (по умолчанию None).
        """
        self.model_type = model_type
        self.model_params = model_params if model_params is not None else {}

    def _create_model(self):
        """
        Создает новый экземпляр модели в зависимости от выбранного типа,
        используя переданные гиперпараметры.
        """
        if self.model_type == "LogisticRegression":
            return LogisticRegression(**self.model_params, max_iter=1000)
        elif self.model_type == "RandomForest":
            return RandomForestClassifier(**self.model_params)
        else:
            raise ValueError(
                "Неверный тип модели. Доступны: 'LogisticRegression', 'RandomForest'"
            )

    def train_with_bow(self, train_texts, train_labels):
        """
        Обучение модели с использованием BoW.
        """
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_texts)

        model = self._create_model()
        model.fit(X_train, train_labels)
        return model, vectorizer

    def train_with_tfidf(self, train_texts, train_labels):
        """
        Обучение модели с использованием TF-IDF.
        """
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_texts)

        model = self._create_model()
        model.fit(X_train, train_labels)
        return model, vectorizer

    def train_with_embeddings(self, X_train_embeddings, y_train):
        """
        Обучение модели с использованием эмбеддингов.
        """
        model = self._create_model()
        model.fit(X_train_embeddings, y_train)
        return model
