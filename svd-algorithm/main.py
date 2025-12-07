import numpy as np


def compute_svd(matrix: np.ndarray):
    """
    Compute SVD components with explicit linear algebra steps instead of np.linalg.svd.
    Returns U, singular_values, Vt.
    """
    A = np.asarray(matrix, dtype=float)
    ata = A.T @ A
    eigvals, eigvecs = np.linalg.eigh(ata)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    Vt = eigvecs[:, order].T
    singular_values = np.sqrt(np.clip(eigvals, 0, None))

    U_cols = []
    for i, sigma in enumerate(singular_values):
        if sigma == 0:
            U_cols.append(np.zeros(A.shape[0]))
            continue
        u_col = (A @ Vt[i]) / sigma
        U_cols.append(u_col)

    U = np.stack(U_cols, axis=1)
    return U, singular_values, Vt


def reconstruct(U: np.ndarray, singular_values: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    sigma_matrix = np.diag(singular_values)
    return U @ sigma_matrix @ Vt


def compress(U: np.ndarray, singular_values: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    sigma_matrix = np.diag(singular_values[:k])
    return U[:, :k] @ sigma_matrix @ Vt[:k]


def main() -> None:
    A = np.array([[4, 4], [3, -3]], dtype=float)
    U, singular_values, Vt = compute_svd(A)

    reconstructed = reconstruct(U, singular_values, Vt)
    k = 1
    compressed = compress(U, singular_values, Vt, k)

    np.set_printoptions(precision=4, suppress=True)
    print("\n--- Original Matrix A ---")
    print(A)
    print("\n--- Reconstructed A ---")
    print(reconstructed)
    print(f"\n--- Compressed Matrix (k={k}) ---")
    print(compressed)


if __name__ == "__main__":
    main()
