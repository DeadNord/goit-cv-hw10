import numpy as np
import gensim.downloader as api
from tqdm import tqdm
import pandas as pd


class EmbeddingModel:
    def __init__(self, model_name):
        """
        Инициализация класса для работы с предобученными эмбеддингами.

        Parameters:
        -----------
        model_name : str
            Название предобученной модели эмбеддингов (например, 'word2vec-google-news-300').
        """
        print(f"Загрузка предобученной модели эмбеддингов '{model_name}'...")
        self.model = api.load(model_name)
        self.vector_size = self.model.vector_size
        print(f"Модель '{model_name}' загружена успешно!")

    def text_to_embedding(self, text):
        """
        Преобразует текст в вектор эмбеддингов, усредняя вектора слов.

        Parameters:
        -----------
        text : str
            Исходный текст для преобразования в эмбеддинги.

        Returns:
        --------
        np.array
            Вектор эмбеддингов текста.
        """
        words = text.split()
        word_vectors = [self.model[word] for word in words if word in self.model]

        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.vector_size)

    def apply_embeddings(self, df, text_column):
        """
        Применяет эмбеддинги к столбцу текстов в DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с текстовыми данными.
        text_column : str
            Название столбца с текстовыми данными.

        Returns:
        --------
        np.array
            Массив эмбеддингов для каждого текста.
        """
        tqdm.pandas()
        embeddings = df[text_column].progress_apply(self.text_to_embedding)
        return np.stack(embeddings.values)
