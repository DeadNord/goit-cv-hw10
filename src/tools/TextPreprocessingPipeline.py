import re
import spacy
import string  # Для работы с пунктуацией
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
import pandas as pd
import nltk
import contractions

# Загрузка стоп-слов
nltk.download("stopwords")


class TextPreprocessingPipeline:
    def __init__(self, config):
        """
        Инициализация класса для предобработки текста.

        Parameters:
        -----------
        config : dict
            Словарь с ключами — названиями методов, значениями — True/False, указывающими, применять ли метод.
        """
        self.config = config
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

        # Попытка загрузки модели spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            print("Модель 'en_core_web_sm' не найдена. Устанавливаю...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def preprocess(self, text):
        """
        Выполняет предобработку текста в зависимости от конфигурации.

        Parameters:
        -----------
        text : str
            Исходный текст для обработки.

        Returns:
        --------
        str
            Обработанный текст.
        """
        if self.config.get("remove_html_tags", False):
            text = re.sub(r"<[^>]*>", " ", text)

        if self.config.get("remove_emails", False):
            text = re.sub(r"\S*@\S*\s+", " ", text)

        if self.config.get("remove_urls", False):
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        if self.config.get("lowercase", False):
            text = text.lower()

        if self.config.get("expand_contractions", False):
            text = self.expand_contractions(text)

        if self.config.get("remove_stopwords", False):
            text = self.remove_stopwords(text)

        if self.config.get("remove_punctuation", False):
            text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        if self.config.get("lemmatize", False):
            text = self.lemmatize(text)

        if self.config.get("stem_words", False):
            text = self.stem_words(text)

        if self.config.get("remove_digits", False):
            text = re.sub(r"\b\d+\b", "", text)

        if self.config.get("remove_extra_spaces", False):
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def expand_contractions(self, text):
        expanded_words = [contractions.fix(word) for word in text.split()]
        return " ".join(expanded_words)

    def remove_stopwords(self, text):
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return " ".join(words)

    def lemmatize(self, text):
        doc = self.nlp(text)
        lemmatized = " ".join([token.lemma_ for token in doc if len(token.lemma_) > 1])
        return lemmatized

    def stem_words(self, text):
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed_words)

    def preprocess_column(self, df, column_name):
        """
        Применяет предобработку к указанному столбцу текста в DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с текстовыми данными.
        column_name : str
            Название столбца для обработки.

        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным столбцом обработанного текста.
        """
        tqdm.pandas()  # Подключаем прогресс-бар для pandas
        df[f"{column_name}_processed"] = df[column_name].progress_apply(self.preprocess)
        return df
