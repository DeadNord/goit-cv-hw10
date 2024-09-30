from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import display


class ModelEvaluator:
    def __init__(self):
        """
        Инициализация класса для оценки модели.
        """
        pass

    def display_results(
        self,
        model,
        X_test,
        y_test,
        vectorizer=None,
        average="weighted",
        show_help=False,
    ):
        """
        Метод для отображения предсказаний модели на тестовом наборе данных и вычисления метрик.

        Parameters:
        -----------
        model : object
            Обученная модель для предсказаний.
        X_test : np.array, pd.DataFrame или pd.Series
            Тестовые данные для предсказаний. Если используется BoW или TF-IDF, то это тексты, которые преобразуются
            с помощью vectorizer. Если используются эмбеддинги, то это уже готовые векторные представления.
        y_test : list или np.array
            Истинные метки для тестовых данных.
        vectorizer : object, optional
            Vectorizer (например, CountVectorizer или TfidfVectorizer), который был использован для преобразования
            текста в вектора. Если переданы уже эмбеддинги, то можно не указывать.
        average : str
            Средний параметр для расчёта метрик (по умолчанию 'weighted' для многоклассовых задач).
        show_help : bool
            Флаг для отображения блока с описанием метрик (по умолчанию False).

        Returns:
        --------
        pd.DataFrame
            Таблица с метриками.
        """
        if vectorizer is not None:
            X_test = vectorizer.transform(X_test)

        # Преобразование меток в числа (если они строковые)
        if y_test.dtype == "object":
            y_test = np.where(y_test == "spam", 1, 0)

        # Предсказания модели
        predictions = model.predict(X_test)

        # Преобразование предсказаний в числа, если они строковые
        if isinstance(predictions[0], str):
            predictions = np.where(predictions == "spam", 1, 0)

        pred_probabilities = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Вычисление метрик
        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average=average)
        recall = recall_score(y_test, predictions, average=average)
        f1 = f1_score(y_test, predictions, average=average)
        roc_auc = (
            roc_auc_score(y_test, pred_probabilities)
            if pred_probabilities is not None
            else None
        )

        # Метрики в виде pandas DataFrame
        metrics_df = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Balanced Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC-AUC",
                ],
                "Value": [accuracy, balanced_acc, precision, recall, f1, roc_auc],
            }
        )
        display(metrics_df)

        # Матрица ошибок
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

        # ROC-кривая
        if pred_probabilities is not None:
            fpr, tpr, _ = roc_curve(y_test, pred_probabilities)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.show()

        if show_help:
            self.display_help()

        return metrics_df

    def display_help(self):
        """
        Метод для отображения описания метрик, которые выводятся в методе display_results.
        """
        help_text = """
        Описание метрик:

        1. **Accuracy** — точность. Соотношение правильно предсказанных наблюдений к общему числу наблюдений. 
           Пример: 80% — это значит, что 80% наблюдений были предсказаны правильно.

        2. **Balanced Accuracy** — сбалансированная точность. Среднее значение между точностью по каждому классу, особенно полезна при несбалансированных классах.

        3. **Precision** — точность. Соотношение правильно предсказанных положительных наблюдений ко всем предсказанным положительным наблюдениям. 
           Пример: Если модель предсказала 100 положительных случаев, но только 80 из них были правильными, точность составит 80%.

        4. **Recall** — полнота. Соотношение правильно предсказанных положительных наблюдений ко всем реальным положительным наблюдениям. 
           Пример: Если есть 100 положительных случаев и модель нашла 80 из них, то полнота будет 80%.

        5. **F1 Score** — гармоническое среднее между точностью и полнотой. Это мера, которая помогает найти баланс между точностью и полнотой.

        6. **ROC-AUC** — площадь под ROC-кривой. Мера, показывающая, насколько хорошо модель различает классы. 
           Значение 1.0 означает идеальную модель, значение 0.5 означает, что модель работает не лучше случайного угадывания.

        """
        print(help_text)

    def compare_models(
        self,
        models,
        test_data,
        y_test,
        vectorizers=None,
        average="weighted",
        show_help=False,
    ):
        """
        Метод для сравнения производительности нескольких моделей по метрикам.

        Parameters:
        -----------
        models : dict
            Словарь, где ключ — название модели (str), а значение — обученная модель.
        test_data : dict
            Словарь с тестовыми данными для каждой модели.
        y_test : list или np.array
            Истинные метки для тестовых данных.
        vectorizers : dict, optional
            Словарь с векторизаторами для каждой модели.
        average : str
            Средний параметр для расчёта метрик (по умолчанию 'weighted' для многоклассовых задач).
        show_help : bool
            Флаг для отображения блока с описанием метрик (по умолчанию False).

        Returns:
        --------
        pd.DataFrame
            Таблица с метриками для каждой модели.
        """
        results = {}

        # Преобразование меток в числа
        if y_test.dtype == "object":
            y_test = np.where(y_test == "spam", 1, 0)

        for model_name, model in models.items():
            print(f"\nProcessing model: {model_name}")

            # Получаем соответствующий тестовый набор для модели
            X_test = test_data.get(model_name)

            # Получаем векторизатор для модели, если он используется
            vectorizer = None if vectorizers is None else vectorizers.get(model_name)

            # Логируем размер данных до преобразования
            print(f"Shape of input data for model '{model_name}': {X_test.shape}")

            if vectorizer is not None:
                # Если используется векторизатор, применяем его к тестовым данным
                X_test_transformed = vectorizer.transform(X_test)
                print(
                    f"Shape of vectorized data for model '{model_name}': {X_test_transformed.shape}"
                )
            else:
                # Если используются эмбеддинги, просто используем данные напрямую
                X_test_transformed = X_test
                print(
                    f"Shape of embeddings data for model '{model_name}': {X_test_transformed.shape}"
                )

            # Получаем предсказания модели
            predictions = model.predict(X_test_transformed)

            # Если модель поддерживает predict_proba, получаем вероятности
            pred_probabilities = (
                model.predict_proba(X_test_transformed)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            # Вычисляем метрики
            accuracy = accuracy_score(y_test, predictions)
            balanced_acc = balanced_accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average=average)
            recall = recall_score(y_test, predictions, average=average)
            f1 = f1_score(y_test, predictions, average=average)
            roc_auc = (
                roc_auc_score(y_test, pred_probabilities)
                if pred_probabilities is not None
                else None
            )

            # Сохраняем результаты для каждой модели
            results[model_name] = {
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc,
            }

        # Преобразуем результаты в pandas DataFrame для удобного отображения
        results_df = pd.DataFrame(results).transpose()
        display(results_df)

        if show_help:
            self.display_help()

        return results_df
