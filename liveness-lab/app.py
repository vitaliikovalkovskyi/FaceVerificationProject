import csv, io, os, time
from flask import Flask, jsonify, request, send_from_directory, Response

from liveness_core import analyze_frame
import metrics

app = Flask(__name__, static_folder="static", static_url_path="/static")
DATA = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA, "attempts.csv")
FIELDS = ["timestamp", "version", "attack_type", "accepted", "failed_step",
          "duration_ms", "note"]
os.makedirs(DATA, exist_ok=True)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/detect", methods=["POST"])
def detect():
    import base64
    img = (request.get_json(silent=True) or {}).get("image", "")
    if "," in img:
        img = img.split(",", 1)[1]
    try:
        return jsonify(analyze_frame(base64.b64decode(img)))
    except Exception:
        return jsonify({"faceFound": False, "error": "bad image"}), 400


@app.route("/attempt", methods=["POST"])
def attempt():
    d = request.get_json(force=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": d.get("version"),
            "attack_type": d.get("attack_type"),
            "accepted": int(bool(d.get("accepted"))),
            "failed_step": d.get("failed_step", ""),
            "duration_ms": int(d.get("duration_ms", 0)),
            "note": d.get("note", ""),
        })
    return jsonify({"ok": True})


@app.route("/metrics")
def get_metrics():
    return jsonify(metrics.compute(CSV_PATH))


@app.route("/export")
def export():
    if not os.path.exists(CSV_PATH):
        return Response("", mimetype="text/csv")
    return send_from_directory(DATA, "attempts.csv", as_attachment=True)


@app.route("/undo", methods=["POST"])
def undo():
    """Видаляє останній запис (якщо помилково зафіксовано спробу)."""
    if os.path.exists(CSV_PATH):
        lines = open(CSV_PATH, encoding="utf-8").read().splitlines()
        if len(lines) > 1:
            open(CSV_PATH, "w", encoding="utf-8").write("\n".join(lines[:-1]) + "\n")
    return jsonify({"ok": True})


@app.route("/clear", methods=["POST"])
def clear():
    """Очищає журнал спроб. Попередньо зберігає копію в data/backup_<час>.csv."""
    if os.path.exists(CSV_PATH):
        os.replace(CSV_PATH, os.path.join(DATA, "backup_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"))
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)