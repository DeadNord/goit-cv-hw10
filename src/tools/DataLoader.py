import pandas as pd
from sklearn.model_selection import train_test_split

# from google.colab import drive
import os


class DataLoader:
    def __init__(self):
        """
        Конструктор класса, который принимает путь к датасету. Если путь не указан,
        необходимо будет загрузить его позже через метод.
        """
        self.data = None

    def load_from_local(self, dataset_path, encoding="utf-8"):
        """
        Метод для загрузки датасета из локального файла.
        Аргументы:
        - dataset_path: Путь к файлу с данными.
        - encoding: Кодировка файла (по умолчанию 'utf-8').
        """
        if dataset_path is None:
            raise ValueError("Путь к датасету не указан!")

        try:
            self.data = pd.read_csv(dataset_path, encoding=encoding)
            print("Данные успешно загружены из локального файла.")
        except UnicodeDecodeError:
            print(f"Ошибка декодирования файла. Попробуйте другую кодировку.")
            raise

        return self.data

    # def load_from_google_drive(self, drive_path):
    #     """
    #     Метод для загрузки датасета из Google Drive.
    #     Перед использованием необходимо подключиться к Google Drive через Google Colab.
    #     """
    #     drive.mount("/content/drive")
    #     full_path = os.path.join("/content/drive", drive_path)

    #     if not os.path.exists(full_path):
    #         raise FileNotFoundError("Файл не найден в указанном пути на Google Диске!")

    #     self.data = pd.read_csv(full_path)
    #     print("Данные успешно загружены из Google Drive.")
    #     return self.data

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """
        Метод для разделения данных на тренировочный и тестовый наборы.

        Аргументы:
        - target_column: Название столбца, содержащего целевые метки.
        - test_size: Доля тестового набора (по умолчанию 0.2).
        - random_state: Контроль случайности для повторяемости результата (по умолчанию 42).
        """
        if self.data is None:
            raise ValueError(
                "Данные не загружены. Сначала загрузите данные с помощью load_from_local."
            )

        if target_column not in self.data.columns:
            raise ValueError(f"Указанный столбец '{target_column}' не найден в данных.")

        # Разделяем данные на признаки (X) и целевую переменную (y)
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Разделяем на тренировочный и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(
            f"Данные успешно разделены: {len(X_train)} тренировочных и {len(X_test)} тестовых образцов."
        )
        return X_train, X_test, y_train, y_test
