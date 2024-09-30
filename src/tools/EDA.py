import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


class EDAEmbedding:
    """
    A class to perform Exploratory Data Analysis (EDA) on embedding vectors.
    """

    def __init__(self, embeddings_df):
        """
        Constructor to initialize EDA with embeddings dataframe.

        Parameters
        ----------
        embeddings_df : pd.DataFrame
            The dataframe containing the embeddings for analysis.
        """
        if isinstance(embeddings_df, pd.DataFrame):
            self.embeddings_df = embeddings_df
        else:
            raise ValueError("The input data must be a pandas DataFrame.")

    def dataset_info(self):
        """
        Prints information about the dataset.
        """
        print("Embedding Dataset Information:\n")
        display(self.embeddings_df.info())

    def dataset_shape(self):
        """
        Prints the shape of the dataset.
        """
        print("\nEmbedding Dataset Shape:\n")
        print(self.embeddings_df.shape)

    def descriptive_statistics(self):
        """
        Displays descriptive statistics of the embeddings dataset.
        """
        print("\nDescriptive Statistics of Embeddings:\n")
        display(self.embeddings_df.describe().transpose())

    def find_duplicates(self):
        """
        Finds and displays the duplicate rows in the dataset.
        """
        duplicates = self.embeddings_df[self.embeddings_df.duplicated()]
        num_duplicates = duplicates.shape[0]

        if num_duplicates > 0:
            print(f"Found {num_duplicates} duplicate rows.")
            display(duplicates)
        else:
            print("No duplicate rows found in the dataset.")

    def perform_full_eda(self):
        """
        Performs full EDA on embedding dataset by calling all the methods.
        """
        self.dataset_info()
        self.dataset_shape()
        self.descriptive_statistics()
        self.find_duplicates()

    def drop_column(self, column_name):
        """
        Drops a column from the dataset if it exists.

        Parameters
        ----------
        column_name : str
            Name of the column to be removed.
        """
        if column_name in self.embeddings_df.columns:
            self.embeddings_df.drop(columns=[column_name], inplace=True)
            print(f"Column '{column_name}' has been removed.")
        else:
            print(f"Column '{column_name}' does not exist in the dataset.")

    def rename_column(self, old_name, new_name):
        """
        Renames a column in the dataset.

        Parameters
        ----------
        old_name : str
            The current name of the column.
        new_name : str
            The new name of the column.
        """
        if old_name in self.embeddings_df.columns:
            self.embeddings_df.rename(columns={old_name: new_name}, inplace=True)
            print(f"Column '{old_name}' has been renamed to '{new_name}'.")
        else:
            print(f"Column '{old_name}' does not exist in the dataset.")

    def plot_class_balance(self, target_column):
        """
        Plots a histogram to assess the class balance for the target column.

        Parameters
        ----------
        target_column : str
            The name of the column containing class labels (e.g., 'spam' or 'ham').
        """
        if target_column not in self.embeddings_df.columns:
            print(f"Target column '{target_column}' does not exist in the dataset.")
            return

        class_counts = self.embeddings_df[target_column].value_counts()

        # Построение гистограммы
        plt.figure(figsize=(8, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values, palette="Set2")
        plt.title(f"Class Balance in '{target_column}'")
        plt.xlabel("Class Labels")
        plt.ylabel("Count")
        plt.show()
