from typing import Iterable, List, Sequence, Tuple

import pandas as pd


def validate_k(k: int, dataset_size: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if k > dataset_size:
        raise ValueError("k cannot be larger than the dataset size.")


def calculate_euclid_distance(feature_cols: Sequence[str], data: pd.DataFrame, prediction: Sequence[float]) -> List[Tuple[int, float]]:
    distances = []
    for idx, row in data.iterrows():
        diff = [(prediction[i] - row[col]) ** 2 for i, col in enumerate(feature_cols)]
        distance = sum(diff) ** 0.5
        distances.append((idx, float(distance)))
    return distances


def find_category(
    distances: List[Tuple[int, float]],
    k: int,
    data: pd.DataFrame,
    label_col: str,
) -> Tuple[int, int, List[Tuple[float, int]]]:
    sorted_distances = sorted(distances, key=lambda x: x[1])[:k]

    count = {}
    weights = {}
    distances_k: List[Tuple[float, int]] = []

    for idx, distance in sorted_distances:
        label = int(data[label_col].iat[idx])
        weight = float("inf") if distance == 0 else 1.0 / distance

        count[label] = count.get(label, 0) + 1
        weights[label] = weights.get(label, 0.0) + weight
        distances_k.append((distance, label))

    weighted_choice = max(weights.items(), key=lambda item: item[1])[0]
    base_choice = max(count.items(), key=lambda item: item[1])[0]
    return weighted_choice, base_choice, distances_k


def knn_algorithm(
    feature_cols: Sequence[str],
    label_col: str,
    data: pd.DataFrame,
    prediction: Sequence[float],
    k: int,
) -> Tuple[int, int, List[Tuple[float, int]]]:
    validate_k(k, len(data))
    distances = calculate_euclid_distance(feature_cols, data, prediction)
    return find_category(distances, k, data, label_col)


def _prompt_features(feature_cols: Iterable[str]) -> List[float]:
    values: List[float] = []
    for col in feature_cols:
        user_input = input(f"Enter value for {col}: ")
        values.append(float(user_input))
    return values


def main() -> None:
    data = pd.read_csv("iphone_purchase_records.csv")
    feature_cols = list(data.columns[:-1])
    label_col = data.columns[-1]

    print(f"Feature columns: {feature_cols}")
    print(f"Dataset size: {len(data)} rows\n")

    try:
        prediction = _prompt_features(feature_cols)
        k = int(input("Enter k (number of neighbors): "))
        weighted, base, distances_k = knn_algorithm(feature_cols, label_col, data, prediction, k)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return

    print(f"\nWeighted KNN predicted category: {weighted}")
    print(f"Basic KNN predicted category: {base}")
    print(f"Nearest {k} distances and labels: {distances_k}")


if __name__ == "__main__":
    main()
