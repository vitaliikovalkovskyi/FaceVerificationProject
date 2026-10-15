# verifier/utils.py
import os
import json
import time
import pickle
import random
import threading
import contextlib
import numpy as np
import kagglehub
from PIL import Image
from deepface import DeepFace
from django.conf import settings

from .embedding_utils import retinaface_no_upscale
from . import occlusion

# ── MFR2: маркери незамаскованих фото ────────────────────────────────────────
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
IMG_EXTS   = ('.jpg', '.jpeg', '.png', '.webp')
PAIRS_SEED = 42   # фіксований набір пар: обидва режими й повторні запуски бачать ті самі пари

# ── Детектори ────────────────────────────────────────────────────────────────
# Експеримент (40 пар на датасет, EER): RetinaFace для еталонів не гірший за opencv
# і збігається з реальною реєстрацією.
DETECTOR_REF  = "retinaface"   # еталони в режимі Multi-Reference
DETECTOR_TEST = "retinaface"   # тестові фото та стандартний режим

# RetinaFace за замовчуванням збільшує фото до 1024 px. Великим фото (mfr2 160 px,
# LFW 250 px) це шкодить і в 10–25 разів сповільнює (EER mfr2 0.15 → 0.10,
# LFW 0.125 → 0.05 без збільшення), а дрібним (~100 px, як у вилученому RMFD) потрібно
# (там EER 0.347 зі збільшенням проти 0.473 без). Тому "auto": збільшувати лише фото,
# коротша сторона яких менша за UPSCALE_MIN_SIDE. True/False — завжди / ніколи.
RETINAFACE_UPSCALE = "auto"
UPSCALE_MIN_SIDE   = 150

# ── Кеш embedding ────────────────────────────────────────────────────────────
# Embedding фото не залежить від порогу, режиму чи вибірки, тому кожне фото
# рахується один раз і зберігається на диску. Ключ містить розмір і час зміни
# файлу, детектор і режим збільшення — зміна будь-чого з цього дає новий запис.
_CACHE_PATH = os.path.join(settings.BASE_DIR, '.benchmark_cache', 'embeddings.pkl')
_cache = None
_cache_dirty = 0
_cache_lock = threading.Lock()


def _cache_load() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, 'rb') as f:
                _cache = pickle.load(f)
        except Exception:
            _cache = {}
    return _cache


def _cache_save(force: bool = False) -> None:
    global _cache_dirty
    with _cache_lock:
        if _cache is None or not _cache_dirty or (not force and _cache_dirty < 25):
            return
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        tmp = _CACHE_PATH + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, _CACHE_PATH)
        _cache_dirty = 0


# ── Утиліти ───────────────────────────────────────────────────────────────────
def _is_image(fname: str) -> bool:
    return fname.lower().endswith(IMG_EXTS)


def _resolve_upscale(img_path: str, upscale) -> bool:
    """Перетворює режим "auto" на True/False за розміром фото."""
    if upscale != "auto":
        return bool(upscale)
    try:
        with Image.open(img_path) as im:
            return min(im.size) < UPSCALE_MIN_SIDE
    except Exception:
        return True


def _represent_all(img_path: str, detector: str, upscale) -> "list[tuple[np.ndarray, float]] | None":
    """
    L2-нормалізовані ArcFace embedding усіх облич на фото та впевненість детектора
    (як DeepFace.represent з enforce_detection=False). None — фото не вдалося обробити.
    """
    global _cache_dirty
    upscale = _resolve_upscale(img_path, upscale)
    st = os.stat(img_path)
    key = (os.path.normcase(os.path.abspath(img_path)), st.st_size, st.st_mtime_ns,
           MODEL_NAME, detector, upscale if detector == 'retinaface' else True)
    cache = _cache_load()
    if key in cache:
        return cache[key]

    ctx = retinaface_no_upscale() if detector == 'retinaface' and not upscale else contextlib.nullcontext()
    try:
        with ctx:
            result = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=detector,
                enforce_detection=False,
                align=True,
            )
        faces = []
        for r in result:
            vec = np.array(r["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                faces.append((vec / norm, float(r.get("face_confidence", 0.0))))
        value = faces or None
    except Exception as e:
        print(f"[DEBUG] Помилка DeepFace ({detector}) на {img_path}: {e}")
        value = None

    with _cache_lock:
        cache[key] = value
        _cache_dirty += 1
    _cache_save()
    return value


def _get_embedding(img_path: str, detector: str = DETECTOR_TEST, upscale=RETINAFACE_UPSCALE) -> "tuple[np.ndarray, float] | None":
    """L2-нормалізований ArcFace embedding першого обличчя та впевненість детектора."""
    faces = _represent_all(img_path, detector, upscale)
    return faces[0] if faces else None


def _verify_distance(img1: str, img2: str, upscale) -> "float | None":
    """
    Відстань як у DeepFace.verify(ArcFace, retinaface, cosine, enforce_detection=False):
    мінімальна косинусна відстань між усіма обличчями двох фото.
    """
    a = _represent_all(img1, DETECTOR_TEST, upscale)
    b = _represent_all(img2, DETECTOR_TEST, upscale)
    if not a or not b:
        return None
    return min(float(1.0 - np.dot(x[0], y[0])) for x in a for y in b)


def _build_ref_embedding(photo_paths: list, detector_ref: str = DETECTOR_REF, upscale=RETINAFACE_UPSCALE) -> "np.ndarray | None":
    """Multi-Reference: weighted mean + outlier removal."""
    photo_paths = photo_paths[:5] # Обмеження для стабільності
    raw_data = []

    for p in photo_paths:
        res = _get_embedding(p, detector=detector_ref, upscale=upscale)
        if res is None and detector_ref != DETECTOR_TEST:
            res = _get_embedding(p, detector=DETECTOR_TEST, upscale=upscale)
        if res is not None:
            raw_data.append(res)

    if not raw_data: return None
    if len(raw_data) == 1: return raw_data[0][0]

    # --- Outlier Removal ---
    all_vecs = [d[0] for d in raw_data]
    mean_v = np.mean(all_vecs, axis=0)
    mean_v /= np.linalg.norm(mean_v)

    filtered = [d for d in raw_data if (1.0 - np.dot(mean_v, d[0])) <= 0.4]
    final_src = filtered if filtered else raw_data

    # --- Weighted Average ---
    vecs = [d[0] for d in final_src]
    weights = [d[1] for d in final_src]

    w_sum = np.sum(weights)
    if w_sum > 1e-6:
        weighted_mean = np.average(vecs, axis=0, weights=weights)
    else:
        weighted_mean = np.mean(vecs, axis=0)

    return weighted_mean / np.linalg.norm(weighted_mean)

def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))

def _eer(dists: list, labels: list) -> "dict | None":
    """
    EER — точка, де FAR = FRR; не залежить від обраного порогу.
    Повертає EER, поріг у цій точці та точність на ньому.
    """
    d, pos = np.array(dists), np.array(labels, dtype=bool)
    if not pos.any() or pos.all():
        return None
    best = None
    for th in np.unique(d):
        far = np.mean(d[~pos] <= th); frr = np.mean(d[pos] > th)
        if best is None or abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2, float(th))
    _, eer, th = best
    return {'eer': float(eer), 'eer_threshold': th, 'eer_accuracy': float(np.mean((d <= th) == pos))}

def _pct(done: int, total: int) -> int:
    return int(done / total * 100) if total > 0 else 0

def _noisy_event(person_id: str, index: int, total: int) -> str:
    msg = {'progress': _pct(index+1, total), 'person': person_id, 'status': 'noisy'}
    return f"data: {json.dumps(msg)}\n\n"

def _safe_index(filename: str) -> int:
    try:
        return int(filename.split('_')[-1].split('.')[0])
    except (ValueError, IndexError):
        return -1

def _rel_path(full_path: str, dataset: str) -> str:
    return os.path.basename(full_path)

# ── Завантажувачі пар ─────────────────────────────────────────────────────────

def _load_lfw_pairs(limit: int) -> list:
    """Завантажує LFW та формує 50/50 позитивні та негативні пари."""
    rng = random.Random(PAIRS_SEED)
    print("[INFO] Завантаження LFW через kagglehub...")
    path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    lfw_root = os.path.join(path, "lfw-deepfunneled", "lfw-deepfunneled")

    people = sorted(d for d in os.listdir(lfw_root) if os.path.isdir(os.path.join(lfw_root, d)))
    valid_people = [p for p in people if len([f for f in os.listdir(os.path.join(lfw_root, p)) if _is_image(f)]) >= 2]

    rng.shuffle(valid_people)
    # Пари будуються для всіх людей, тож менша вибірка — префікс більшої (і кеш спрацьовує)

    pairs = []
    for i in range(len(valid_people)):
        p_a = valid_people[i]
        p_b = valid_people[(i + 1) % len(valid_people)]

        dir_a = os.path.join(lfw_root, p_a)
        dir_b = os.path.join(lfw_root, p_b)
        imgs_a = sorted([os.path.join(dir_a, f) for f in os.listdir(dir_a) if _is_image(f)])
        imgs_b = sorted([os.path.join(dir_b, f) for f in os.listdir(dir_b) if _is_image(f)])

        # Позитивна пара (свій)
        pairs.append({
            'id': p_a, 'is_pos': True, 'dataset': 'lfw',
            'ref_paths': imgs_a[:-1], 'ref_single': imgs_a[0], 'test': imgs_a[-1]
        })
        # Негативна пара (чужий)
        pairs.append({
            'id': f"{p_a}_vs_{p_b}", 'is_pos': False, 'dataset': 'lfw',
            'ref_paths': imgs_a[:-1], 'ref_single': imgs_a[0], 'test': imgs_b[0]
        })

    rng.shuffle(pairs)
    return pairs[:limit]

def _load_mfr2_pairs(limit: int) -> list:
    """
    MFR2: еталон — незамасковані фото людини, тест — фото в масці.
    Позитивні пари — маска тієї ж людини, негативні — маска наступної людини
    (по одній на кожну позитивну), щоб FAR мав сенс.
    """
    rng = random.Random(PAIRS_SEED)
    base = os.path.join(settings.BASE_DIR, 'mfr2')
    people = {}
    for person_id in sorted(os.listdir(base)):
        p_path = os.path.join(base, person_id)
        if not os.path.isdir(p_path): continue
        files = sorted(f for f in os.listdir(p_path) if _is_image(f))
        no_mask_idx = MFR2_NO_MASK_INDICES.get(person_id, [])
        unmasked = [os.path.join(p_path, f) for f in files if _safe_index(f) in no_mask_idx]
        masked   = [os.path.join(p_path, f) for f in files if _safe_index(f) not in no_mask_idx]
        if unmasked and masked:
            people[person_id] = (unmasked, masked)

    ids = list(people)
    pairs = []
    for i, person_id in enumerate(ids):
        u_paths, masked = people[person_id]
        next_id = ids[(i + 1) % len(ids)]
        other_masked = people[next_id][1]
        for j, m in enumerate(masked):
            pairs.append({
                'id': person_id, 'is_pos': True, 'dataset': 'mfr2',
                'ref_paths': u_paths, 'ref_single': u_paths[0], 'test': m
            })
            pairs.append({
                'id': f"{person_id}_vs_{next_id}", 'is_pos': False, 'dataset': 'mfr2',
                'ref_paths': u_paths, 'ref_single': u_paths[0], 'test': other_masked[j % len(other_masked)]
            })
    rng.shuffle(pairs)
    return pairs[:limit]

# ── Головний генератор ────────────────────────────────────────────────────────

def run_accuracy_generator(limit=50, threshold=0.75, dataset='mfr2', use_embedding=False,
                           detector_ref=DETECTOR_REF, upscale=RETINAFACE_UPSCALE, dynamic=False):
    """
    dynamic=True — динамічний поріг: якщо на тестовому фото виявлено маску, діє
    occlusion.MASK_THRESHOLD (як і при вході в систему).
    Класифікатор маски — модель, яка не бачила ні LFW (для LFW), ні людину з тестового фото (для mfr2).
    """
    threshold, limit = float(threshold), int(limit)
    t_start = time.perf_counter()
    yield f"data: {json.dumps({'status': 'info', 'msg': f'Dataset: {dataset.upper()} | Mode: {MODEL_NAME}'})}\n\n"

    if dataset == 'lfw': pairs = _load_lfw_pairs(limit)
    else: pairs = _load_mfr2_pairs(limit)

    if not pairs:
        yield f"data: {json.dumps({'status': 'error', 'message': 'Дані не знайдені'})}\n\n"; return

    total = len(pairs)
    yield f"data: {json.dumps({'status': 'info', 'msg': f'Пар для тесту: {total}'})}\n\n"

    # FAR/FRR рахуються лише за оціненими парами (пропущені не входять у знаменник)
    correct, skipped, far_count, frr_count, t_pos, t_neg = 0, 0, 0, 0, 0, 0
    all_dists, all_labels = [], []
    masked_n = 0
    dynamic = dynamic and occlusion.available()
    try:
        ref_cache = {}
        if use_embedding:
            unique_ids = {p['id'].split('_vs_')[0]: p['ref_paths'] for p in pairs}
            yield f"data: {json.dumps({'status': 'info', 'msg': f'Кешування {len(unique_ids)} осіб...'})}\n\n"
            for i, (pid, paths) in enumerate(unique_ids.items()):
                vec = _build_ref_embedding(paths, detector_ref=detector_ref, upscale=upscale)
                if vec is not None: ref_cache[pid] = vec
                if (i + 1) % 5 == 0:
                    yield f"data: {json.dumps({'status': 'cache', 'msg': f'Кеш: {i+1}/{len(unique_ids)}', 'progress': 0})}\n\n"

        for index, pair in enumerate(pairs):
            is_pos = pair.get('is_pos', True)
            try:
                if use_embedding:
                    pid = pair['id'].split('_vs_')[0]
                    ref_vec = ref_cache.get(pid)
                    test_res = _get_embedding(pair['test'], detector=DETECTOR_TEST, upscale=upscale)
                    test_vec = test_res[0] if test_res else None
                    if ref_vec is None or test_vec is None:
                        skipped += 1; yield _noisy_event(pair['id'], index, total); continue
                    dist = _cosine_dist(ref_vec, test_vec)
                else:
                    dist = _verify_distance(pair['ref_single'], pair['test'], upscale)
                    if dist is None:
                        skipped += 1; yield _noisy_event(pair['id'], index, total); continue
                thr_used, masked = threshold, False
                if dynamic:
                    if use_embedding:
                        probe = test_vec
                    else:
                        faces = _represent_all(pair['test'], DETECTOR_TEST, upscale)
                        probe = faces[0][0] if faces else None
                    mask_model = occlusion.benchmark_model(dataset, pair['id'].split('_vs_')[-1])
                    masked = probe is not None and occlusion.is_masked(probe, mask_model)
                    if masked:
                        masked_n += 1
                        thr_used = max(threshold, occlusion.MASK_THRESHOLD)
                is_match = dist <= thr_used
                all_dists.append(dist); all_labels.append(is_pos)

                if is_pos: t_pos += 1
                else: t_neg += 1

                # Основна логіка правильності
                if is_match == is_pos:
                    correct += 1

                # Статистика помилок
                if not is_pos and is_match: far_count += 1
                if is_pos and not is_match: frr_count += 1

                p_msg = {
                    'progress': _pct(index + 1, total),
                    'person': pair['id'],
                    'result': 'MATCH' if is_match else 'NO MATCH',
                    'is_pos': is_pos,
                    'dist': round(dist, 4),
                    'masked': masked,
                    'thr': thr_used,
                    'img_ref': _rel_path(pair['ref_single'], pair['dataset']),
                    'img_test': _rel_path(pair['test'], pair['dataset']),
                    'status': 'processing'
                }
                yield f"data: {json.dumps(p_msg)}\n\n"
            except Exception as e:
                print(f"[ERROR] {e}"); skipped += 1; yield _noisy_event(pair['id'], index, total)
    finally:
        _cache_save(force=True)

    eff = total - skipped
    acc = (correct / eff * 100) if eff > 0 else 0
    far = (far_count / t_neg * 100) if t_neg > 0 else 0
    frr = (frr_count / t_pos * 100) if t_pos > 0 else 0
    eer = _eer(all_dists, all_labels)

    res_final = {
        'progress': 100,
        'accuracy': f'{acc:.2f}%',
        'far': f'{far:.2f}%',
        'frr': f'{frr:.2f}%',
        'skipped': skipped,
        'total': total,
        'correct': correct,
        'effective': eff,
        'pos_n': t_pos,
        'neg_n': t_neg,
        'seconds': round(time.perf_counter() - t_start, 1),
        'dynamic': dynamic,
        'masked_n': masked_n,
        'eer': f"{eer['eer'] * 100:.2f}%" if eer else '—',
        'eer_threshold': round(eer['eer_threshold'], 4) if eer else None,
        'eer_accuracy': f"{eer['eer_accuracy'] * 100:.2f}%" if eer else '—',
        'mode': MODEL_NAME,
        'dataset': dataset.upper(),
        'threshold': threshold,
        'status': 'completed'
    }
    yield f"data: {json.dumps(res_final)}\n\n"
