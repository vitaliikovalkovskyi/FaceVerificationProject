"""
Навчає класифікатор маски на ArcFace embedding фото з mfr2 і LFW
(embedding беруться з кешу бенчмарку, відсутні — обчислюються).

    python manage.py train_mask_classifier [--pairs 200]

Мітки: у mfr2 тестові фото — у масці, еталонні — без; у LFW масок немає.
Зберігає verifier/mask_classifier.npz з моделями:
  all          — mfr2 + LFW, для реальної роботи системи;
  no_lfw       — лише mfr2, для чесного бенчмарку на LFW;
  mfr2_fold0/1 — без людей mfr2 з відповідної половини (+ LFW), для чесного бенчмарку на mfr2.
"""
import numpy as np
from django.core.management.base import BaseCommand

from verifier import occlusion, utils


def _collect(dataset: str, pairs_limit: int):
    """(embedding, маска?, людина) для кожного унікального фото датасету."""
    loader = utils._load_mfr2_pairs if dataset == "mfr2" else utils._load_lfw_pairs
    X, y, who, seen = [], [], [], set()
    for p in loader(pairs_limit):
        owner = p['id'].split('_vs_')[0]
        test_owner = p['id'].split('_vs_')[-1]
        items = [(r, False, owner) for r in p['ref_paths'][:5]] + [(p['test'], dataset == "mfr2", test_owner)]
        for path, masked, person in items:
            if path in seen:
                continue
            seen.add(path)
            e = utils._get_embedding(path, detector=utils.DETECTOR_TEST, upscale=utils.RETINAFACE_UPSCALE)
            if e is not None:
                X.append(e[0]); y.append(masked); who.append(person)
    return np.array(X), np.array(y, dtype=bool), np.array(who)


def _report(w, b, X, y) -> str:
    pred = (X @ w + b) > 0
    mask = f"маска {pred[y].mean():.1%}" if y.any() else "маска —"
    return f"{mask}, чисте {(~pred[~y]).mean():.1%} (n={len(y)})"


class Command(BaseCommand):
    help = "Навчає класифікатор маски на ArcFace embedding (mfr2, LFW)"

    def add_arguments(self, parser):
        parser.add_argument("--pairs", type=int, default=200, help="кількість пар бенчмарку з кожного датасету")

    def handle(self, *args, **opts):
        mfr2 = _collect("mfr2", opts["pairs"])
        lfw = _collect("lfw", opts["pairs"])
        utils._cache_save(force=True)
        self.stdout.write(f"mfr2: {len(mfr2[1])} фото, з них у масці {int(mfr2[1].sum())}; lfw: {len(lfw[1])} фото")

        folds = np.array([occlusion.mfr2_fold(p) for p in mfr2[2]])
        weights = {}

        def fit(name, X, y):
            w, b = occlusion.train_logistic(X, y)
            weights[f"w_{name}"], weights[f"b_{name}"] = w, np.array(b)
            return w, b

        w, b = fit("all", np.vstack([mfr2[0], lfw[0]]), np.concatenate([mfr2[1], lfw[1]]))
        self.stdout.write(f"  all: на навчальних даних — {_report(w, b, np.vstack([mfr2[0], lfw[0]]), np.concatenate([mfr2[1], lfw[1]]))}")

        w, b = fit("no_lfw", mfr2[0], mfr2[1])
        self.stdout.write(f"  no_lfw: на LFW (не бачив) — {_report(w, b, lfw[0], lfw[1])}")

        for k in (0, 1):
            train = folds != k
            w, b = fit(f"mfr2_fold{k}", np.vstack([mfr2[0][train], lfw[0]]), np.concatenate([mfr2[1][train], lfw[1]]))
            self.stdout.write(f"  mfr2_fold{k}: на людях mfr2 з половини {k} (не бачив) — {_report(w, b, mfr2[0][~train], mfr2[1][~train])}")

        np.savez(occlusion.WEIGHTS_PATH, **weights)
        self.stdout.write(self.style.SUCCESS(f"Збережено {occlusion.WEIGHTS_PATH}"))
