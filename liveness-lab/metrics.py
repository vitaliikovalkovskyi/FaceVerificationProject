"""APCER / BPCER / ACER за ISO/IEC 30107-3. Запуск: python metrics.py [шлях до csv]"""
import csv, os, sys
from collections import defaultdict

ATTACKS = ["photo_screen", "video_replay"]
FTA_STEP = "calibrate"   # обличчя не знайдено при калібруванні: не відсіювання атаки, а FTA (failure to acquire)


def compute(path):
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for ver in sorted({r["version"] for r in rows}):
        all_sub = [r for r in rows if r["version"] == ver]
        fta = sum(1 for r in all_sub if r["failed_step"] == FTA_STEP)
        sub = [r for r in all_sub if r["failed_step"] != FTA_STEP]
        gen = [r for r in sub if r["attack_type"] == "genuine"]
        bpcer = (sum(1 for r in gen if r["accepted"] == "0") / len(gen)) if gen else None
        per, vals = {}, []
        for a in ATTACKS:
            att = [r for r in sub if r["attack_type"] == a]
            if att:
                v = sum(1 for r in att if r["accepted"] == "1") / len(att)
                per[a] = {"n": len(att), "apcer": v}
                vals.append(v)
        apcer_max = max(vals) if vals else None   # у 30107-3 для ACER береться найгірший PAI
        acer = (apcer_max + bpcer) / 2 if (apcer_max is not None and bpcer is not None) else None
        times = [int(r["duration_ms"]) for r in gen if r["accepted"] == "1"]
        counts = {}
        for r in sub:
            counts[r["attack_type"]] = counts.get(r["attack_type"], 0) + 1
        out[ver] = {
            "counts": counts,
            "genuine_n": len(gen), "fta": fta, "bpcer": bpcer, "per_attack": per,
            "apcer_worst": apcer_max, "acer": acer,
            "mean_genuine_time_ms": (sum(times) / len(times)) if times else None,
        }
    return out


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "attempts.csv")
    for ver, m in compute(p).items():
        print(f"\n== {ver} ==  genuine n={m['genuine_n']}")
        f = lambda x: "—" if x is None else f"{x*100:.1f}%"
        print(f"FTA (не враховано в метриках): {m['fta']}")
        print(f"BPCER: {f(m['bpcer'])}")
        for a, v in m["per_attack"].items():
            print(f"APCER[{a}] (n={v['n']}): {f(v['apcer'])}")
        print(f"APCER worst: {f(m['apcer_worst'])}   ACER: {f(m['acer'])}")
        if m["mean_genuine_time_ms"]:
            print(f"Mean time of accepted genuine attempt: {m['mean_genuine_time_ms']/1000:.1f} s")