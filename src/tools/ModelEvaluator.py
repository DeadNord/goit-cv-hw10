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

        if y_test.dtype == "object":
            y_test = np.where(y_test == "spam", 1, 0)

        predictions = model.predict(X_test)

        if isinstance(predictions[0], str):
            predictions = np.where(predictions == "spam", 1, 0)

        pred_probabilities = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

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

        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

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
        self, models, test_data, y_test, show_help=False, average="weighted", n_cols=3
    ):
        """
        Метод для сравнения производительности нескольких моделей по метрикам.

        Parameters:
        -----------
        models : dict
            Словарь, где ключ — название модели (str), а значение — обученная модель.
        test_data : dict
            Словарь, где ключ — название модели (str), а значение — данные для тестирования модели (обработанные BoW, TF-IDF или эмбеддинги).
        y_test : list или np.array
            Истинные метки для тестовых данных.
        show_help : bool
            Флаг для отображения блока с описанием метрик (по умолчанию False).
        average : str
            Средний параметр для расчёта метрик (по умолчанию 'weighted' для многоклассовых задач).
        n_cols : int
            Количество графиков в строке для отображения метрик (по умолчанию 3).

        Returns:
        --------
        pd.DataFrame
            Таблица с метриками для каждой модели.
        """
        results = {}
        roc_curves = {}

        if y_test.dtype == "object":
            y_test = np.where(y_test == "spam", 1, 0)

        for model_name, model in models.items():
            X_test_transformed = test_data[model_name]

            predictions = model.predict(X_test_transformed)

            if isinstance(predictions[0], str):
                predictions = np.where(predictions == "spam", 1, 0)

            pred_probabilities = (
                model.predict_proba(X_test_transformed)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

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

            results[model_name] = {
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc,
            }

            if pred_probabilities is not None:
                fpr, tpr, _ = roc_curve(y_test, pred_probabilities)
                roc_curves[model_name] = (fpr, tpr, roc_auc)

        results_df = pd.DataFrame(results).transpose()
        display(results_df)

        if show_help:
            self.display_help()

        metrics = [
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
        ]

        num_metrics = len(metrics)
        n_rows = (num_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = [
                results[model_name][metric]
                for model_name in results
                if results[model_name][metric] is not None
            ]
            axes[i].bar(results.keys(), values)
            axes[i].set_title(f"Сравнение моделей по {metric}")
            axes[i].set_ylabel(metric)

        for j in range(i + 1, n_rows * n_cols):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        if roc_curves:
            plt.figure(figsize=(8, 6))
            for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves of All Models")
            plt.legend()
            plt.show()
