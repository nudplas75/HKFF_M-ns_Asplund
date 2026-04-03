import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.movies_df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.title_to_idx = None

    def _load_movies(self):
        path = os.path.join(self.data_dir, "movies.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {path}.")
        df = pd.read_csv(path)
        df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)
        df["genres_clean"] = df["genres_clean"].str.replace("(no genres listed)", "", regex=False)
        return df

    def _load_tags(self):
        for name in ["tags.csv", "tags_light.csv"]:
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                print(f"     Using tag file: {name}")
                tags = pd.read_csv(path, usecols=["movieId", "tag"])
                tags["tag"] = tags["tag"].fillna("").astype(str).str.lower().str.strip()
                tags = tags[tags["tag"] != ""]
                agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
                agg.columns = ["movieId", "tags_text"]
                return agg
        print("[INFO] No tag file found using genres only.")
        return None

    def _natural_title(self, title: str) -> str:
     
        import re
        m = re.match(r'^(.*?),\s*(The|A|An|Les|Los|Las|El|La|Le|Die|Das|Der|Den|De)\s*(\(\d{4}\))?$',
                     title, re.IGNORECASE)
        if m:
            base, article, year = m.group(1), m.group(2), m.group(3) or ""
            return f"{article} {base} {year}".strip()
        return title

    def fit(self):
        print("[1/4] Reading movies.csv ...")
        self.movies_df = self._load_movies()

        print("[2/4] Reading tags ...")
        tags_df = self._load_tags()

        print("[3/4] Building document per movie ...")
        if tags_df is not None:
            self.movies_df = self.movies_df.merge(tags_df, on="movieId", how="left")
            self.movies_df["tags_text"] = self.movies_df["tags_text"].fillna("")
        else:
            self.movies_df["tags_text"] = ""

        self.movies_df["document"] = (
            self.movies_df["genres_clean"] + " " + self.movies_df["tags_text"]
        ).str.strip()

        self.movies_df = self.movies_df[
            self.movies_df["document"].str.strip() != ""
        ].reset_index(drop=True)

        self.movies_df["natural_title"] = self.movies_df["title"].apply(self._natural_title)

        n_tags = (self.movies_df["tags_text"] != "").sum()
        print(f"     {len(self.movies_df)} movies, of which {n_tags} have tags")

        print("[4/4] Vectorizing with TF-IDF ...")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["document"])
        print(f"     Matrix: {self.tfidf_matrix.shape[0]} movies x {self.tfidf_matrix.shape[1]} terms")

        self.title_to_idx = {}
        for i, row in self.movies_df.iterrows():
            self.title_to_idx[row["title"].lower()]         = i
            self.title_to_idx[row["natural_title"].lower()] = i

        print("     Done!\n")
        return self

    def recommend(self, movie_title, n=5):

        if self.tfidf_matrix is None:
            raise RuntimeError("Run .fit() before .recommend().")

        key = movie_title.lower().strip()
        if key in self.title_to_idx:
            idx = self.title_to_idx[key]
        else:
            candidates = [t for t in self.title_to_idx if key in t]
            if not candidates:
                raise ValueError(f"The movie '{movie_title}' was not found. Tip: try 'Matrix, The (1999)'.")
            idx = self.title_to_idx[candidates[0]]
            print(f"[INFO] Used closest match: '{self.movies_df.iloc[idx]['title']}'")

        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        scores[idx] = -1
        top_idx = np.argsort(scores)[::-1][:n]

        result = self.movies_df.iloc[top_idx][["title", "genres"]].copy()
        result["score"] = scores[top_idx].round(4)
        result.reset_index(drop=True, inplace=True)
        return result

    def search(self, query, max_results=10):

        if self.movies_df is None:
            raise RuntimeError("Run .fit() first.")

        STOP = {"the", "a", "an", "le", "la", "les", "el", "los", "las", "de", "die", "das", "der"}
        words = query.strip().lower().split()

        filtered = [w for w in words if w not in STOP]
        if not filtered:
            filtered = words

        mask = pd.Series([True] * len(self.movies_df), index=self.movies_df.index)
        for word in filtered:
            word_mask = (
                self.movies_df["title"].str.contains(word, case=False, na=False, regex=False) |
                self.movies_df["natural_title"].str.contains(word, case=False, na=False, regex=False)
            )
            mask = mask & word_mask

        return self.movies_df[mask]["natural_title"].head(max_results).tolist()


def main():
    if len(sys.argv) < 2:
        print("Usage:   python recommender.py '<movie title>' [data_dir]")
        print("Example: python recommender.py 'Toy Story (1995)' ./ml-latest")
        sys.exit(1)

    movie_title = sys.argv[1]
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    rec = MovieRecommender(data_dir=data_dir)
    rec.fit()

    print(f"Recommendations for: '{movie_title}'")
    print("=" * 52)
    try:
        recs = rec.recommend(movie_title)
        for i, row in recs.iterrows():
            print(f"{i+1}. {row['title']}")
            print(f"   Genres: {row['genres']}")
            print(f"   Score:  {row['score']}\n")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()