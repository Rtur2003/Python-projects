# Python Projects

Small ML and math demos, each isolated in its own folder:

- `bert-classifier`: Minimal BERT text classifier; requires `torch` and `transformers` with the `bert-base-uncased` checkpoint cached locally.
- `faiss`: Prompt-based autocomplete demo powered by character n-gram vectors; run `python main.py` in the folder. Stopwords live in `faiss/data/stopwords_tr.txt`.
- `knn-algorithm`: KNN classifier two ways (manual and `sklearn`) on `iphone_purchase_records.csv`; run `python knn_without_library.py` or `python knn_with_library.py`. A quick comparison script is available via `python test.py`.
- `svd-algorithm`: Walks through an explicit SVD computation without calling `numpy.linalg.svd`; run `python main.py`.

Install dependencies per subfolder needs (e.g., `pip install -r knn-algorithm/requirements.txt`).

