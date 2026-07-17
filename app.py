

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


from rag_pipes import ask

app = Flask(__name__)
CORS(app)



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/ask", methods=["POST"])
def handle_ask():
    try:
        data  = request.get_json(force=True)
        query = (data.get("query") or "").strip()

        if not query:
            return jsonify({"error": "Empty query"}), 400

        answer = ask(query)

       
        college_keywords = [
            "iiit", "bhopal", "fee", "hostel", "admission", "semester",
            "placement", "campus", "mess", "branch", "cutoff", "jee",
            "department", "faculty", "college", "institute"
        ]
        used_rag = any(kw in query.lower() for kw in college_keywords)

        return jsonify({
            "answer":   answer,
            "used_rag": used_rag
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "gemini-2.5-flash"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀  Server running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)