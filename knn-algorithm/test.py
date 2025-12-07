import time

import pandas as pd

from knn_without_library import knn_algorithm
from knn_with_library import sklearn_model_predict, validate_k


def run_test(k: int = 3, test_input=(32, 160000)) -> None:
    data = pd.read_csv("iphone_purchase_records.csv")
    feature_cols = list(data.columns[:-1])
    label_col = data.columns[-1]
    validate_k(k, len(data))

    age, salary = test_input

    start_custom = time.time()
    weighted, base, distances_k = knn_algorithm(feature_cols, label_col, data, test_input, k)
    duration_custom = time.time() - start_custom

    start_sklearn = time.time()
    sklearn_result = sklearn_model_predict(age, salary, k, csv_path="iphone_purchase_records.csv")
    duration_sklearn = time.time() - start_sklearn

    print("KNN Comparison Report")
    print(f"Test input: age={age}, salary={salary} | k={k}")
    print(f"Custom KNN (weighted, base): ({weighted}, {base}) | Distances: {distances_k} | Time: {duration_custom:.6f}s")
    print(f"sklearn KNN prediction: {sklearn_result} | Time: {duration_sklearn:.6f}s")


if __name__ == "__main__":
    run_test()
