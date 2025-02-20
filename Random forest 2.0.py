import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
import operator

from pandas.core.ops import comparison_op


# Train-test split function
def train_test_split(data: pd.DataFrame, test_share: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices = data.index.to_list()
    test_indices = random.sample(population=indices, k=test_share)
    test_df = data.loc[test_indices]
    train_df = data.drop(test_indices)
    return test_df, train_df

# Check purity of a dataset - whether it contains values of one type only
def purity(data: np.ndarray) -> bool:
    label_column = data[:, -1]
    unique = np.unique(label_column)
    return len(unique) == 1

# Classify based on majority vote
def classify(data: np.ndarray) -> np.ndarray:
    label_column = data[:, -1]
    unique, counts_unique = np.unique(label_column, return_counts=True)
    return unique[counts_unique.argmax()]

# Determine points for potential splits - in the middle between dataset values
def get_potential_splits(data: np.ndarray, random_subspace: int) -> dict:
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))
    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)
    for i in column_indices:  # Exclude label column
        potential_splits[i] = []
        values = data[:, i]
        unique_values = np.unique(values)
        for j in range(len(unique_values) - 1):
            potential_splits[i].append((unique_values[j] + unique_values[j + 1]) / 2)
    return potential_splits

# Compute entropy of a dataset
def calculate_entropy(data: np.ndarray) -> float:
    if len(data) == 0:
        return 0  # Avoid log(0) issue
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy

# Compute weighted entropy for a split
def get_overall_entropy(data_below: np.ndarray, data_above: np.ndarray) -> float:
    n = len(data_below) + len(data_above)
    if n == 0:
        return 0  # Avoid division by zero
    pbelow = len(data_below) / n
    pabove = len(data_above) / n
    overall_entropy = (pbelow * calculate_entropy(data_below)) + (pabove * calculate_entropy(data_above))
    return overall_entropy

# Split dataset based on column and threshold value
def split_data(data: np.ndarray, split_column: int, split_value: float) -> tuple[np.ndarray, np.ndarray]:
    data_below = data[data[:, split_column] <= split_value]
    data_above = data[data[:, split_column] > split_value]
    return data_below, data_above

# Find best split using entropy minimization
def best_split(data: np.array, potential_splits: dict) -> tuple[int, float]:
    min_entropy = np.inf
    bs_index, bs_value = None, None

    for i in potential_splits:
        for j in potential_splits[i]:
            data_below, data_above = split_data(data, i, j)
            current_entropy = get_overall_entropy(data_below, data_above)
            if current_entropy <= min_entropy:
                min_entropy = current_entropy
                bs_index = i
                bs_value = j
    return bs_index, bs_value

def decision_tree(df: pd.DataFrame, counter=0, min_samples=2, max_depth=5, random_subspace=None) -> dict | np.ndarray:
    if counter == 0:
        data = df.values
        column_names = df.columns
    else:
        data = df
        column_names = None  # Prevent reference to train_df at deeper recursion levels
    if purity(data):
        return classify(data)
    else:
        counter += 1
        potential_splits = get_potential_splits(data, 3)
        split_column, split_value = best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        column_name = column_names[split_column] if column_names is not None else f"Feature {split_column}"
        question = "{} < {}".format(column_name, split_value)

        sub_tree = {question: []}
        yes = decision_tree(data_below, counter, min_samples, max_depth, random_subspace)
        no = decision_tree(data_above, counter, min_samples, max_depth, random_subspace)

        if yes == no:
            sub_tree = yes
        else:
            sub_tree[question].append(yes)
            sub_tree[question].append(no)
        return sub_tree

def predict_example(ex: list, tree: dict) -> str:
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")
    if comparison_op == "<=":
        if ex[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if str(ex[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    if not isinstance(answer, dict):
        return answer
    else:
        return predict_example(ex, answer)


def tree_predictions(df, tree):
    if len(df) != 0:
        predictions = df.apply(predict_example, args=(tree,), axis=1)
    else:
        # "df.apply()"" with empty dataframe returns an empty dataframe,
        # but "predictions" should be a series instead
        predictions = pd.Series()

    return predictions

# 3.3 Accuracy
def accuracy(df, tree):
    predictions = tree_predictions(df, tree)
    predictions_correct = predictions == df.label
    accuracy = predictions_correct.mean()

    return accuracy

def bootstrap(train_df: pd.DataFrame, n: int) -> pd.DataFrame:
    boot_indices = np.random.randint(low=0, high=len(train_df), size=n)
    return train_df.iloc[boot_indices]

def random_forest(df: pd.DataFrame, n_trees: int, n_bootstrap: int, n_features: int, dt_max_depth: int) -> list:
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrap(df, n_bootstrap)
        forest.append(decision_tree(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features))
    return forest

def forest_predictions(test_df: pd.DataFrame, forest: list) -> pd.Series:
     df_predictions = {}
     for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = tree_predictions(test_df, tree = forest[i])
        df_predictions[column_name] = predictions
     df_predictions = pd.DataFrame(df_predictions)
     return df_predictions.mode(axis = 1)[0]

df = pd.read_csv("D:\\Downloads\\Iris.csv")
df = df.drop("Id", axis=1)
df = df.rename(columns={"Species": "Label"})
random.seed(0)
test_share = 20
n_repetitions = 10
test_df, train_df = train_test_split(df, test_share)

accuracies = []
forest = random_forest(train_df, n_trees=4, n_bootstrap=80, n_features=4, dt_max_depth=4)
print(tree_predictions(test_df, forest[0]))