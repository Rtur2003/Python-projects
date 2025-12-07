from pathlib import Path
from typing import Iterable, List, Sequence

from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


BASE_KEYWORDS = [
    "mekan",
    "mekanik",
    "mektep",
    "emek",
    "çiçek",
    "kitap",
    "kalem",
    "kalın",
    "araba",
    "armut",
    "asker",
    "aslan",
    "bilgi",
    "bilim",
    "biber",
    "bilgisayar",
    "makine",
    "yazılım",
    "donanım",
    "koşmak",
    "gitmek",
    "gelmek",
    "görmek",
    "konuşmak",
    "yemek",
    "içmek",
    "güzel",
    "hızlı",
    "büyük",
    "küçük",
    "soğuk",
    "sıcak",
    "uzun",
    "kısa",
    "mutlu",
    "üzgün",
    "zor",
    "kolay",
]

DEFAULT_STOPWORDS = {
    "ve",
    "ile",
    "bu",
    "şu",
    "o",
    "mi",
    "mu",
    "mı",
    "mu",
    "bir",
    "hep",
    "çok",
    "bile",
}

STOPWORDS_PATH = Path(__file__).parent / "data" / "stopwords_tr.txt"


def load_stopwords(file_path: Path) -> List[str]:
    if file_path.is_file():
        return [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    # Fallback stopwords keep the CLI usable when the file is missing; adjust only with dataset awareness. @Rtur2003
    return sorted(DEFAULT_STOPWORDS)


class KeywordIndex:
    def __init__(self, keywords: Sequence[str], stopwords: Iterable[str]):
        normalized = [kw.strip() for kw in keywords if kw and kw.strip()]
        if not normalized:
            raise ValueError("Keyword list is empty.")
        self.keywords = sorted(set(normalized))
        self.stopwords = {s.lower() for s in stopwords}
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.embeddings = self.vectorizer.fit_transform(self.keywords)
        self.neighbors = NearestNeighbors(metric="cosine")
        self.neighbors.fit(self.embeddings)

    def search(self, text: str, limit: int = 5) -> List[str]:
        query = text.strip()
        if len(query) < 2:
            return []

        query_vec = self.vectorizer.transform([query])
        requested = min(max(limit, 1) * 2, len(self.keywords))
        distances, indices = self.neighbors.kneighbors(query_vec, n_neighbors=requested)

        suggestions: List[str] = []
        for idx in indices[0]:
            candidate = self.keywords[idx]
            if candidate.lower() in self.stopwords:
                continue
            suggestions.append(candidate)
            if len(suggestions) >= limit:
                break
        return suggestions


class FaissCompleter(Completer):
    def __init__(self, index: KeywordIndex, limit: int = 5):
        self.index = index
        self.limit = limit

    def get_completions(self, document, complete_event):
        query = document.text
        for suggestion in self.index.search(query, limit=self.limit):
            yield Completion(suggestion, start_position=-len(query))


def run_cli() -> None:
    stopwords = load_stopwords(STOPWORDS_PATH)
    index = KeywordIndex(BASE_KEYWORDS, stopwords)
    completer = FaissCompleter(index)

    while True:
        try:
            user_input = prompt("Kelime girin (çıkmak için q): ", completer=completer)
            if user_input.strip().lower() == "q":
                break
            print(f"Seçilen: {user_input}")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_cli()
