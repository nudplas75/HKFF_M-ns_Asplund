import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from recommender import MovieRecommender

# Configure path to data 
DATA_DIR = "."

# Start and train the model on startup 
print("Starting recommendation model...")
rec = MovieRecommender(data_dir=DATA_DIR)
rec.fit()

# Flask app
app = Flask(__name__)
CORS(app)


@app.route("/api/status")
def status():
    """Return info about the model's state."""
    n_movies = len(rec.movies_df) if rec.movies_df is not None else 0
    n_with_tags = int((rec.movies_df["tags_text"] != "").sum()) if rec.movies_df is not None else 0
    return jsonify({
        "ready": rec.tfidf_matrix is not None,
        "total_movies": n_movies,
        "movies_with_tags": n_with_tags,
        "matrix_shape": list(rec.tfidf_matrix.shape) if rec.tfidf_matrix is not None else None
    })


@app.route("/api/search")
def search():
    """Search movie titles. Parameter: q (search term), max (max number of results)."""
    q = request.args.get("q", "").strip()
    max_results = int(request.args.get("max", 8))

    if not q or len(q) < 2:
        return jsonify([])

    titles = rec.search(q, max_results=max_results)
    return jsonify(titles)


@app.route("/api/recommend")
def recommend():
    """Fetch recommendations. Parameter: title, n (count)."""
    title = request.args.get("title", "").strip()
    n = int(request.args.get("n", 5))

    if not title:
        return jsonify({"error": "Provide a movie title via the 'title' parameter"}), 400

    try:
        df = rec.recommend(title, n=n)
        results = []
        for _, row in df.iterrows():
            raw_tags = rec.movies_df.loc[
                rec.movies_df["title"] == row["title"], "tags_text"
            ].values
            tags_str = raw_tags[0] if len(raw_tags) > 0 else ""
            top_tags = [t for t in tags_str.split() if len(t) > 3][:3]

            nat = rec.movies_df.loc[
                rec.movies_df["title"] == row["title"], "natural_title"
            ].values
            natural = nat[0] if len(nat) > 0 else row["title"]

            results.append({
                "title": natural,
                "genres": row["genres"],
                "score": float(row["score"]),
                "score_pct": int(row["score"] * 100),
                "tags": top_tags
            })
        return jsonify({"input": title, "recommendations": results})

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == "__main__":
    print("\nServer ready! Open index.html in the browser.")
    print("   API running at: http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)