# verifier/utils.py
import os
import json
import random
import numpy as np
from deepface import DeepFace
from django.conf import settings

# Маркери незамаскованих фото в MFR2
MFR2_NO_MASK_INDICES = {
    "AdrianDunbar": [2, 4], "AhmadAli": [2, 5], "ArifAlvi": [3, 4], "BellaHadid": [2],
    "BenAffleck": [3, 7], "BrianKemp": [1, 5], "BrodyJenner": [1], "CarrieLam": [4, 6],
    "CharlieBaker": [1, 2], "DennyTamaki": [7, 8], "DonnellRawlings": [1], "DougDucey": [2, 5],
    "EmmanuelMacron": [1, 4], "EricGarcetti": [2, 5], "FloydMayweather": [1, 3], "GaryHerbert": [7, 8],
    "GavinNewsom": [5, 6], "GretchenWhitmer": [1, 5], "IgorMatovic": [4, 5], "ImranKhan": [1, 5],
    "JaredPolis": [9, 10], "JoeBiden": [2, 5], "JonathanBennet": [2, 3], "JustinTrudeau": [3, 6],
    "KimKardashian": [2], "KyriakosMitsotakis": [3, 5], "LadyGaga": [4, 5], "LarryHogan": [5, 7],
    "MatthewMorrison": [2, 3], "MikeDewine": [3, 5], "MitchMcConnell": [2, 6], "MuhammadFahmi": [1, 4],
    "NancyPelosi": [4], "PharrellWilliams": [3, 4], "PhilMurphy": [4, 6], "RalphNortham": [2, 5],
    "ReneeMorrison": [2, 3], "RonDesantis": [3, 6], "RoyCooper": [1, 6], "SimonCowell": [3, 4],
    "SteveHarvey": [2], "StevenReed": [5, 6], "TakeOff": [2, 3], "TaroKono": [2],
    "TomWolf": [4], "TsaiIngwen": [1, 6], "UddhavThackery": [2, 3], "VassilisKikilias": [1, 3],
    "XiJinping": [2, 5], "YoshihideSuga": [5, 6], "YurikoKoike": [6, 7], "ZacEfron": [1, 3],
    "ZuzanaCaputova": [5, 7]
}

MODEL_NAME = "ArcFace"
DETECTOR   = "retinaface"


# ── Внутрішня функція ─────────────────────────────────────────────────────────
def _get_embedding(img_path: str) -> np.ndarray | None:
    """Повертає L2-нормалізований embedding або None."""
    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False,
            align=True,
        )
        vec = np.array(result[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None
    except Exception:
        return None


def _build_ref_embedding(photo_paths: list[str]) -> np.ndarray | None:
    """
    Multi-Reference: будує усереднений вектор з кількох незамаскованих фото.
    Саме це і є головна різниця від старого підходу (1 фото vs. N фото).
    """
    embeddings = [_get_embedding(p) for p in photo_paths]
    embeddings = [e for e in embeddings if e is not None]
    if not embeddings:
        return None
    mean_vec = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 0 else None


def _cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(1.0 - np.dot(vec_a, vec_b))


# ── Головний генератор ────────────────────────────────────────────────────────
def run_accuracy_generator(limit=50, threshold=0.75, dataset='mfr2', use_embedding=False):
    """
    SSE-генератор результатів бенчмарку.

    Args:
        limit:          кількість пар для тестування
        threshold:      поріг cosine distance
        dataset:        'mfr2' або 'afdb'
        use_embedding:  True  → Multi-Reference Embedding (новий метод)
                        False → DeepFace.verify() один до одного (старий метод)
    """
    threshold = float(threshold)
    base_path = os.path.join(settings.BASE_DIR, 'mfr2')

    # ── Статусне повідомлення про режим ──
    mode_label = "MULTI-REFERENCE EMBEDDING" if use_embedding else "СТАНДАРТНИЙ DEEPFACE.VERIFY"
    yield f"data: {json.dumps({'status': 'info', 'msg': f'Режим: {mode_label}'})}\n\n"

    # ── Підготовка пар ────────────────────────────────────────────────────────
    people = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    all_test_pairs = []

    for person_id in people:
        p_path = os.path.join(base_path, person_id)
        files = [f for f in os.listdir(p_path) if f.lower().endswith(('.png', '.jpg'))]

        no_mask_indices = MFR2_NO_MASK_INDICES.get(person_id, [])

        # Розділяємо на незамасковані (еталони) і замасковані (тест)
        unmasked_files = [
            f for f in files
            if _safe_index(f) in no_mask_indices
        ]
        masked_files = [
            f for f in files
            if _safe_index(f) not in no_mask_indices
        ]

        if not unmasked_files or not masked_files:
            continue

        unmasked_paths = [os.path.join(p_path, f) for f in unmasked_files]

        for m_file in masked_files:
            all_test_pairs.append({
                'id':           person_id,
                'ref_paths':    unmasked_paths,      # всі незамасковані → для embedding
                'ref_single':   os.path.join(p_path, random.choice(unmasked_files)),  # для fallback
                'test':         os.path.join(p_path, m_file),
            })

    random.shuffle(all_test_pairs)
    test_batch = all_test_pairs[:int(limit)]
    total = len(test_batch)

    if total == 0:
        yield f"data: {json.dumps({'status': 'error', 'message': 'Датасет порожній або не знайдено пар'})}\n\n"
        return

    correct = 0
    skipped = 0

    # ── Цикл верифікації ──────────────────────────────────────────────────────
    for index, pair in enumerate(test_batch):
        try:
            if use_embedding:
                # ── НОВИЙ МЕТОД: Multi-Reference Embedding ──
                ref_vec = _build_ref_embedding(pair['ref_paths'])
                if ref_vec is None:
                    skipped += 1
                    yield _noisy_event(pair['id'], index, total)
                    continue

                test_vec = _get_embedding(pair['test'])
                if test_vec is None:
                    skipped += 1
                    yield _noisy_event(pair['id'], index, total)
                    continue

                dist = _cosine_distance(ref_vec, test_vec)
                is_match = dist <= threshold

            else:
                # ── СТАРИЙ МЕТОД: DeepFace.verify() 1-до-1 ──
                res = DeepFace.verify(
                    img1_path=pair['ref_single'],
                    img2_path=pair['test'],
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=False,
                )
                dist = res['distance']
                is_match = dist <= threshold

            if is_match:
                correct += 1

            # Відносні шляхи для відображення
            rel_ref  = _rel_path(pair['ref_single'])
            rel_test = _rel_path(pair['test'])

            yield f"data: {json.dumps({'progress': _pct(index + 1, total), 'person': pair['id'], 'result': str(is_match).upper(), 'dist': round(dist, 4), 'img_ref': rel_ref, 'img_test': rel_test, 'status': 'processing'})}\n\n"

        except Exception:
            skipped += 1
            yield _noisy_event(pair['id'], index, total)

    # ── Фінальний результат ───────────────────────────────────────────────────
    effective_total = total - skipped
    accuracy = (correct / effective_total * 100) if effective_total > 0 else 0

    yield f"data: {json.dumps({'progress': 100, 'accuracy': f'{accuracy:.2f}%', 'skipped': skipped, 'correct': correct, 'total': total, 'mode': mode_label, 'status': 'completed'})}\n\n"


# ── Допоміжні функції ─────────────────────────────────────────────────────────
def _safe_index(filename: str) -> int:
    try:
        return int(filename.split('_')[-1].split('.')[0])
    except (ValueError, IndexError):
        return -1


def _rel_path(full_path: str) -> str:
    return "mfr2/" + full_path.split('mfr2' + os.sep)[-1].replace(os.sep, '/')


def _pct(done: int, total: int) -> int:
    return int(done / total * 100)


def _noisy_event(person_id: str, index: int, total: int) -> str:
    return f"data: {json.dumps({'progress': _pct(index + 1, total), 'person': person_id, 'status': 'noisy'})}\n\n"