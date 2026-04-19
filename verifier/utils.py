# verifier/utils.py
import os
import json
import random
import numpy as np
import kagglehub
from deepface import DeepFace
from django.conf import settings

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

# ── Два детектори ────────────────────────────────────────────────────────────
DETECTOR_REF  = "opencv"
DETECTOR_TEST = "retinaface"

# ── Утиліти ───────────────────────────────────────────────────────────────────
def _is_image(fname: str) -> bool:
    return fname.lower().endswith(IMG_EXTS)

def _get_embedding(img_path: str, detector: str = DETECTOR_TEST) -> "tuple[np.ndarray, float] | None":
    """L2-нормалізований ArcFace embedding та впевненість детектора."""
    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=detector,
            enforce_detection=False,
            align=True,
        )
        if not result or len(result) == 0:
            print(f"[DEBUG] Обличчя не знайдено ({detector}): {img_path}")
            return None
        
        vec = np.array(result[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        score = result[0].get("face_confidence", 0.0)
        
        return (vec / norm if norm > 0 else None, score)
    except Exception as e:
        print(f"[DEBUG] Помилка DeepFace на {img_path}: {e}")
        return None

def _build_ref_embedding(photo_paths: list) -> "np.ndarray | None":
    """Multi-Reference: weighted mean + outlier removal."""
    photo_paths = photo_paths[:5] # Обмеження для стабільності
    raw_data = []
    
    for p in photo_paths:
        res = _get_embedding(p, detector=DETECTOR_REF)
        if res is None:
            res = _get_embedding(p, detector=DETECTOR_TEST)
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
    if dataset == 'rmfd':
        for marker in ('AFDB_masked_face_dataset', 'AFDB_face_dataset'):
            if marker in full_path:
                return marker + '/' + full_path.split(marker + os.sep)[-1].replace(os.sep, '/')
    return os.path.basename(full_path)

# ── Завантажувачі пар ─────────────────────────────────────────────────────────

def _load_lfw_pairs(limit: int) -> list:
    """Завантажує LFW та формує 50/50 позитивні та негативні пари."""
    print("[INFO] Завантаження LFW через kagglehub...")
    path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    lfw_root = os.path.join(path, "lfw-deepfunneled", "lfw-deepfunneled")
    
    people = [d for d in os.listdir(lfw_root) if os.path.isdir(os.path.join(lfw_root, d))]
    valid_people = [p for p in people if len([f for f in os.listdir(os.path.join(lfw_root, p)) if _is_image(f)]) >= 2]
    
    random.shuffle(valid_people)
    valid_people = valid_people[:limit] 
    
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
    
    random.shuffle(pairs)
    return pairs[:limit]

def _load_rmfd_pairs(limit: int, rmfd_path: str = 'datasets/rmfd') -> list:
    candidates = [os.path.join(settings.BASE_DIR, rmfd_path), os.path.join(settings.BASE_DIR, 'rmfd')]
    rmfd_root = next((c for c in candidates if os.path.isdir(c)), None)
    if not rmfd_root: return []

    u_dir = os.path.join(rmfd_root, 'AFDB_face_dataset')
    m_dir = os.path.join(rmfd_root, 'AFDB_masked_face_dataset')
    common = sorted(list(set(os.listdir(u_dir)) & set(os.listdir(m_dir))))

    pairs = []
    for i, p_id in enumerate(common):
        u_p, m_p = os.path.join(u_dir, p_id), os.path.join(m_dir, p_id)
        u_f, m_f = sorted([f for f in os.listdir(u_p) if _is_image(f)]), sorted([f for f in os.listdir(m_p) if _is_image(f)])
        
        if u_f and m_f:
            pairs.append({
                'id': p_id, 'is_pos': True, 'dataset': 'rmfd',
                'ref_paths': [os.path.join(u_p, x) for x in u_f],
                'ref_single': os.path.join(u_p, u_f[0]), 'test': os.path.join(m_p, m_f[0])
            })
            # Негативна
            p_next = common[(i + 1) % len(common)]
            m_p_next = os.path.join(m_dir, p_next)
            m_f_next = sorted([f for f in os.listdir(m_p_next) if _is_image(f)])
            if m_f_next:
                pairs.append({
                    'id': f"{p_id}_vs_{p_next}", 'is_pos': False, 'dataset': 'rmfd',
                    'ref_paths': [os.path.join(u_p, x) for x in u_f],
                    'ref_single': os.path.join(u_p, u_f[0]), 'test': os.path.join(m_p_next, m_f_next[0])
                })
    random.shuffle(pairs)
    return pairs[:limit]

def _load_mfr2_pairs(limit: int) -> list:
    base = os.path.join(settings.BASE_DIR, 'mfr2')
    pairs = []
    for person_id in os.listdir(base):
        p_path = os.path.join(base, person_id)
        if not os.path.isdir(p_path): continue
        files = [f for f in os.listdir(p_path) if _is_image(f)]
        no_mask_idx = MFR2_NO_MASK_INDICES.get(person_id, [])
        unmasked = [f for f in files if _safe_index(f) in no_mask_idx]
        masked   = [f for f in files if _safe_index(f) not in no_mask_idx]
        if unmasked and masked:
            u_paths = [os.path.join(p_path, f) for f in unmasked]
            for m in masked:
                pairs.append({
                    'id': person_id, 'is_pos': True, 'dataset': 'mfr2',
                    'ref_paths': u_paths, 'ref_single': u_paths[0], 'test': os.path.join(p_path, m)
                })
    random.shuffle(pairs)
    return pairs[:limit]

# ── Головний генератор ────────────────────────────────────────────────────────

def run_accuracy_generator(limit=50, threshold=0.75, dataset='mfr2', use_embedding=False, rmfd_path='datasets/rmfd'):
    threshold, limit = float(threshold), int(limit)
    yield f"data: {json.dumps({'status': 'info', 'msg': f'Dataset: {dataset.upper()} | Mode: {MODEL_NAME}'})}\n\n"

    if dataset == 'rmfd': pairs = _load_rmfd_pairs(limit, rmfd_path)
    elif dataset == 'lfw': pairs = _load_lfw_pairs(limit)
    else: pairs = _load_mfr2_pairs(limit)

    if not pairs:
        yield f"data: {json.dumps({'status': 'error', 'message': 'Дані не знайдені'})}\n\n"; return

    total = len(pairs)
    yield f"data: {json.dumps({'status': 'info', 'msg': f'Пар для тесту: {total}'})}\n\n"

    ref_cache = {}
    if use_embedding:
        unique_ids = {p['id'].split('_vs_')[0]: p['ref_paths'] for p in pairs}
        yield f"data: {json.dumps({'status': 'info', 'msg': f'Кешування {len(unique_ids)} осіб...'})}\n\n"
        for i, (pid, paths) in enumerate(unique_ids.items()):
            vec = _build_ref_embedding(paths)
            if vec is not None: ref_cache[pid] = vec
            if (i + 1) % 5 == 0:
                yield f"data: {json.dumps({'status': 'cache', 'msg': f'Кеш: {i+1}/{len(unique_ids)}', 'progress': 0})}\n\n"

    correct, skipped, far_count, frr_count, t_pos, t_neg = 0, 0, 0, 0, 0, 0

    for index, pair in enumerate(pairs):
        is_pos = pair.get('is_pos', True)
        if is_pos: t_pos += 1
        else: t_neg += 1
        try:
            if use_embedding:
                pid = pair['id'].split('_vs_')[0]
                ref_vec = ref_cache.get(pid)
                test_res = _get_embedding(pair['test'], detector=DETECTOR_TEST)
                test_vec = test_res[0] if test_res else None
                if ref_vec is None or test_vec is None:
                    skipped += 1; yield _noisy_event(pair['id'], index, total); continue
                dist = _cosine_dist(ref_vec, test_vec)
                is_match = dist <= threshold
            else:
                res = DeepFace.verify(img1_path=pair['ref_single'], img2_path=pair['test'], model_name=MODEL_NAME, detector_backend=DETECTOR_TEST, enforce_detection=False)
                dist, is_match = res['distance'], res['distance'] <= threshold

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
                'img_ref': _rel_path(pair['ref_single'], pair['dataset']),
                'img_test': _rel_path(pair['test'], pair['dataset']), 
                'status': 'processing'
            }
            yield f"data: {json.dumps(p_msg)}\n\n"
        except Exception as e:
            print(f"[ERROR] {e}"); skipped += 1; yield _noisy_event(pair['id'], index, total)

    eff = total - skipped
    acc = (correct / eff * 100) if eff > 0 else 0
    far = (far_count / t_neg * 100) if t_neg > 0 else 0
    frr = (frr_count / t_pos * 100) if t_pos > 0 else 0

    res_final = {
        'progress': 100, 
        'accuracy': f'{acc:.2f}%', 
        'far': f'{far:.2f}%', 
        'frr': f'{frr:.2f}%', 
        'skipped': skipped, 
        'total': total, 
        'correct': correct,
        'effective': eff,
        'mode': MODEL_NAME,
        'dataset': dataset.upper(),
        'threshold': threshold,
        'status': 'completed'
    }
    yield f"data: {json.dumps(res_final)}\n\n"