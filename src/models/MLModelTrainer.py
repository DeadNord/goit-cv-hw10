from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class MLModelTrainer:
    def __init__(self, model_type="LogisticRegression"):
        """
        Инициализация класса для обучения моделей.

        Parameters:
        -----------
        model_type : str
            Тип модели для обучения ('LogisticRegression', 'RandomForest').
        """
        if model_type == "LogisticRegression":
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == "RandomForest":
            self.model = RandomForestClassifier()
        else:
            raise ValueError(
                "Неверный тип модели. Доступны: 'LogisticRegression', 'RandomForest'"
            )

    def train_with_bow(self, train_texts, train_labels):
        """
        Обучение модели с использованием BoW.

        Parameters:
        -----------
        train_texts : list
            Список обучающих текстов.
        train_labels : list
            Метки обучающих данных.

        Returns:
        --------
        model : object
            Обученная модель.
        vectorizer : object
            Обученный CountVectorizer для дальнейшего применения.
        """
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_texts)

        self.model.fit(X_train, train_labels)
        return self.model, vectorizer

    def train_with_tfidf(self, train_texts, train_labels):
        """
        Обучение модели с использованием TF-IDF.

        Parameters:
        -----------
        train_texts : list
            Список обучающих текстов.
        train_labels : list
            Метки обучающих данных.

        Returns:
        --------
        model : object
            Обученная модель.
        vectorizer : object
            Обученный TfidfVectorizer для дальнейшего применения.
        """
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_texts)

        self.model.fit(X_train, train_labels)
        return self.model, vectorizer

    def train_with_embeddings(self, X_train_embeddings, y_train):
        """
        Обучение модели с использованием эмбеддингов.

        Parameters:
        -----------
        X_train_embeddings : np.array
            Массив векторов эмбеддингов для обучающих данных.
        y_train : list
            Метки обучающих данных.

        Returns:
        --------
        model : object
            Обученная модель.
        """
        self.model.fit(X_train_embeddings, y_train)
        return self.model
