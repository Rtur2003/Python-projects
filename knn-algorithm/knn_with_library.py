import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def validate_k(k: int, dataset_size: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if k > dataset_size:
        raise ValueError("k cannot be larger than the dataset size.")


def sklearn_model_predict(age: float, salary: float, k: int, csv_path: str = "iphone_purchase_records.csv") -> int:
    data = pd.read_csv(csv_path)
    validate_k(k, len(data))

    features = data[["Age", "Salary"]]
    labels = data["Purchase Iphone"]

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(features, labels)

    prediction = model.predict(pd.DataFrame([[age, salary]], columns=["Age", "Salary"]))
    return int(prediction[0])


def main() -> None:
    try:
        age = float(input("Please enter your age: "))
        salary = float(input("Please enter your annual salary: "))
        k = int(input("Enter k (number of neighbors): "))
        result = sklearn_model_predict(age, salary, k)
        print(f"Category: {result}")
    except ValueError as exc:
        print(f"Input error: {exc}")
    except Exception as exc:  # pragma: no cover - defensive logging for interactive use
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
