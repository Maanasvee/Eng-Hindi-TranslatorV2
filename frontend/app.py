from flask import Flask, request, jsonify, render_template
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from translate import load_model, translate_en_to_hi

app   = Flask(__name__)
CKPT  = os.path.join(os.path.dirname(__file__), '../checkpoints/best_model.pt')
model, en_vocab, hi_vocab, hi_inv = load_model(CKPT)
print("Model loaded!")

@app.route("/")
def index(): return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text","").strip()
    if not text: return jsonify({"translation":""})
    result = translate_en_to_hi(text, model, en_vocab, hi_vocab, hi_inv)
    return jsonify({"translation": result})

if __name__ == "__main__":
    print("Starting at http://localhost:5000")
    app.run(debug=True, port=5000)