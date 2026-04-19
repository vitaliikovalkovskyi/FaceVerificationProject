import os
import base64
import requests
import shutil
import pyotp
import qrcode
import json
import numpy as np
from io import BytesIO
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.conf import settings
from django.core.files.base import ContentFile
from deepface import DeepFace
from .models import UserProfile
from .embedding_utils import (
    build_reference_embedding,
    get_live_embedding,
    compute_personal_threshold,
    verify_with_embedding,
    BASE_THRESHOLD,
)
from .utils import run_accuracy_generator


# ============================================================
# ДОПОМІЖНА ФУНКЦІЯ ЗБЕРЕЖЕННЯ BASE64 ФОТО
# ============================================================
def save_base64_image(data_url, file_name):
    try:
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_captures')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(data)
        return file_path
    except Exception as e:
        print(f"Помилка збереження base64: {e}")
        return None


# ============================================================
# РЕЄСТРАЦІЯ — ЕТАП 1: Перевірка фото + Генерація QR
# Тепер приймає до 3 фото (camera_data_1/2/3 або file uploads)
# і зберігає всі шляхи в сесії для подальшої побудови embedding.
# ============================================================
def register_step1(request):
    if request.method == 'POST':
        # Очищення сесії
        for key in ['reg_username', 'reg_secret', 'reg_temp_photo_paths']:
            request.session.pop(key, None)

        username = request.POST.get('username', '').strip()
        if not username:
            return JsonResponse({"success": False, "error": "Вкажіть ім'я користувача"}, status=400)

        if UserProfile.objects.filter(username=username).exists():
            return JsonResponse({"success": False, "error": "Користувач вже існує!"}, status=400)

        # ── Збираємо фото: підтримуємо до 3 camera_data або file uploads ──
        temp_paths = []

        # Camera frames (base64) — поля camera_data_1, camera_data_2, camera_data_3
        # Для зворотної сумісності: якщо є просто camera_data — теж приймаємо
        camera_fields = ['camera_data_1', 'camera_data_2', 'camera_data_3', 'camera_data']
        for field in camera_fields:
            raw = request.POST.get(field)
            if not raw:
                continue
            # Нормалізуємо формат
            if ';base64,' in raw:
                img_data = raw.split(';base64,')[1]
            else:
                img_data = raw
            idx = len(temp_paths) + 1
            fname = f"temp_reg_{username}_{idx}.jpg"
            path = save_base64_image(f"data:image/jpeg;base64,{img_data}", fname)
            if path:
                temp_paths.append(path)

        # File uploads — поля photo_1, photo_2, photo_3 або просто photo
        file_fields = ['photo_1', 'photo_2', 'photo_3', 'photo']
        for field in file_fields:
            f = request.FILES.get(field)
            if not f:
                continue
            idx = len(temp_paths) + 1
            path = os.path.join(settings.MEDIA_ROOT, 'temp_captures', f"temp_reg_{username}_{idx}.jpg")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as out:
                for chunk in f.chunks():
                    out.write(chunk)
            temp_paths.append(path)

        if not temp_paths:
            return JsonResponse({"success": False, "error": "Фото не надано"}, status=400)

        # ── Валідація: перевіряємо що хоча б на першому фото є обличчя ──
        try:
            DeepFace.extract_faces(
                img_path=temp_paths[0],
                detector_backend='retinaface',
                enforce_detection=True
            )
        except Exception:
            for p in temp_paths:
                if os.path.exists(p):
                    os.remove(p)
            return JsonResponse({"success": False, "error": "Обличчя не знайдено на фото"}, status=400)

        # ── Генерація TOTP QR ──
        secret = pyotp.random_base32()
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username, issuer_name="FaceID System"
        )
        qr = qrcode.make(totp_uri)
        buffer = BytesIO()
        qr.save(buffer, format="PNG")
        qr_base64 = base64.b64encode(buffer.getvalue()).decode()

        # ── Зберігаємо в сесію ──
        request.session['reg_username'] = username
        request.session['reg_secret'] = secret
        request.session['reg_temp_photo_paths'] = temp_paths  # список шляхів

        return JsonResponse({
            "success": True,
            "qr_code": qr_base64,
            "photo_count": len(temp_paths),
            "message": f"Прийнято {len(temp_paths)} фото. Скануйте QR-код.",
        })

    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)


# ============================================================
# РЕЄСТРАЦІЯ — ЕТАП 2: OTP перевірка + Збереження в БД
# Тут будуємо multi-reference embedding і зберігаємо його.
# ============================================================
def register_step2(request):
    if request.method == 'POST':
        otp_code = request.POST.get('otp_code', '').strip()
        username = request.session.get('reg_username')
        secret = request.session.get('reg_secret')
        temp_paths = request.session.get('reg_temp_photo_paths', [])

        if not username or not secret:
            return JsonResponse(
                {"success": False, "error": "Сесія закінчилась. Почніть спочатку."},
                status=400
            )

        # 1. Перевірка OTP
        totp = pyotp.TOTP(secret)
        if not totp.verify(otp_code):
            return JsonResponse({"success": False, "error": "Невірний код! Спробуйте ще раз."}, status=400)

        # 2. Будуємо Multi-Reference Embedding
        try:
            ref_embedding, photo_count = build_reference_embedding(temp_paths)
        except ValueError as e:
            return JsonResponse({"success": False, "error": f"Помилка embedding: {str(e)}"}, status=400)

        # 3. Обчислюємо персональний поріг (якщо є хоча б 2 фото)
        personal_threshold = compute_personal_threshold(temp_paths)

        # 4. Зберігаємо в БД
        try:
            new_user = UserProfile(
                username=username,
                totp_secret=secret,
                face_embedding=ref_embedding.tolist(),
                personal_threshold=personal_threshold,
                embedding_photo_count=photo_count,
            )
            # Зберігаємо перше фото як превью (для user_list)
            first_path = temp_paths[0]
            with open(first_path, 'rb') as f:
                img_content = ContentFile(f.read(), name=f"{username}.jpg")
                new_user.photo.save(f"{username}.jpg", img_content, save=True)

        except Exception as e:
            return JsonResponse({"success": False, "error": f"Помилка БД: {str(e)}"}, status=500)

        # 5. Чистимо тимчасові файли і сесію
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)

        for key in ['reg_username', 'reg_secret', 'reg_temp_photo_paths']:
            request.session.pop(key, None)

        return JsonResponse({
            "success": True,
            "message": f"Реєстрація завершена! Використано {photo_count} фото для embedding.",
            "threshold": round(personal_threshold, 4),
        })

    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)


# ============================================================
# ЛОГІН — БІОМЕТРИЧНА ВЕРИФІКАЦІЯ
# Використовує Multi-Reference Embedding + TTA.
# Якщо embedding відсутній (старий користувач) — fallback
# до старого DeepFace.verify().
# ============================================================
def login_face_verify(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        image_data = request.POST.get('image_data')
        user = get_object_or_404(UserProfile, username=username)

        attempts = request.session.get('face_attempts', 0) + 1
        request.session['face_attempts'] = attempts
        is_blocked = attempts >= 3

        captured_path = save_base64_image(image_data, f"login_attempt_{username}.jpg")

        try:
            # ── НОВИЙ ШЛЯХ: Multi-Reference Embedding ──
            if user.face_embedding:
                live_embedding = get_live_embedding(captured_path)
                threshold = user.personal_threshold or BASE_THRESHOLD
                result = verify_with_embedding(
                    live_embedding=live_embedding,
                    ref_embedding=np.array(user.face_embedding),
                    threshold=threshold,
                )
            else:
                # ── FALLBACK: старий метод для користувачів без embedding ──
                raw = DeepFace.verify(
                    img1_path=user.photo.path,
                    img2_path=captured_path,
                    model_name='ArcFace',
                    detector_backend='retinaface',
                    enforce_detection=True,
                )
                result = {
                    "verified":  raw["verified"],
                    "distance":  round(raw["distance"], 4),
                    "threshold": round(raw["threshold"], 4),
                }

            if result["verified"]:
                request.session['face_attempts'] = 0
                return JsonResponse({
                    "success": True,
                    "message": "Обличчя підтверджено!",
                    "distance": result["distance"],
                    "threshold": result["threshold"],
                })

            error_type = "too_many_attempts" if is_blocked else "fail"
            msg = (
                f"3 невдалі спроби. Перехід на OTP."
                if is_blocked
                else f"Обличчя не збігається. Відстань: {result['distance']:.3f} (поріг: {result['threshold']:.3f}). Спроба {attempts}/3."
            )
            return JsonResponse({
                "success": False,
                "error": error_type,
                "message": msg,
                "distance": result["distance"],
            })

        except Exception as e:
            error_type = "too_many_attempts" if is_blocked else "no_face"
            msg = (
                "3 невдалі спроби (обличчя не знайдено). Перехід на OTP."
                if is_blocked
                else f"Обличчя не знайдено. Спроба {attempts}/3."
            )
            return JsonResponse({"success": False, "error": error_type, "message": msg})

    return JsonResponse({"error": "Method not allowed"}, status=405)


# ============================================================
# ЛОГІН — OTP РЕЗЕРВНИЙ ВХІД
# ============================================================
def login_otp_verify(request):
    if request.method == 'POST':
        otp_code = request.POST.get('otp_code')
        username = request.POST.get('username')
        user = get_object_or_404(UserProfile, username=username)
        totp = pyotp.TOTP(user.totp_secret)
        if totp.verify(otp_code):
            return JsonResponse({"success": True, "message": "Вхід успішний!"})
        return JsonResponse({"success": False, "error": "Невірний код."})
    return JsonResponse({"error": "Method not allowed"}, status=405)


# ============================================================
# БЕНЧМАРК — ПОТОКОВИЙ SSE
# ============================================================
def benchmark_page(request):
    return render(request, 'verifier/benchmark.html')


def benchmark_stream(request):
    limit = int(request.GET.get('limit', 50))
    threshold = request.GET.get('threshold', 0.75)
    dataset = request.GET.get('dataset', 'mfr2')
    use_embedding = request.GET.get('use_embedding', 'false').lower() == 'true'

    return StreamingHttpResponse(
        run_accuracy_generator(limit, threshold, dataset, use_embedding=use_embedding),
        content_type='text/event-stream'
    )


# ============================================================
# СТАРІ VIEW (без змін)
# ============================================================
def verify_image(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    image_data = request.POST.get('image_data')
    if not image_data:
        return JsonResponse({"error": "No data"}, status=400)
    captured_path = save_base64_image(image_data, "captured_image.jpg")
    test_photo_path = os.path.join(settings.STATICFILES_DIRS[0], "testPhoto.jpg")
    if not os.path.exists(test_photo_path):
        return JsonResponse({"error": "no_reference", "message": "Спочатку зробіть еталонне фото!"}, status=400)
    try:
        result = DeepFace.verify(img1_path=test_photo_path, img2_path=captured_path, enforce_detection=True)
        return JsonResponse({"verified": result["verified"], "distance": result["distance"], "threshold": result["threshold"]})
    except Exception as e:
        error_msg = str(e)
        if "Face could not be detected" in error_msg:
            return JsonResponse({"error": "no_face_detected", "message": "На фото не видно обличчя."}, status=400)
        return JsonResponse({"error": "server_error", "message": error_msg}, status=500)


def capture_test_photo(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    image_data = request.POST.get('image_data')
    if not image_data:
        return JsonResponse({"error": "No data"}, status=400)
    temp_path = save_base64_image(image_data, "temp_test.jpg")
    try:
        DeepFace.extract_faces(img_path=temp_path, enforce_detection=True)
        save_base64_image(image_data, "testPhoto.jpg")
        return JsonResponse({"success": True, "message": "Еталонне фото успішно збережено."})
    except ValueError:
        return JsonResponse({"error": "no_face_detected", "message": "На еталонному фото не знайдено обличчя!"}, status=400)
    except Exception as e:
        return JsonResponse({"error": "server_error", "message": str(e)}, status=500)


def faceAnalise(request):
    result = None
    relative_path = None
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    if request.method == 'POST':
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        else:
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Не вдалося видалити {file_path}: {e}')
        try:
            if request.FILES.get('photo'):
                photo = request.FILES['photo']
                relative_path = os.path.join('uploads', photo.name)
                file_path = os.path.join(settings.MEDIA_ROOT, relative_path)
                with open(file_path, 'wb') as f:
                    for chunk in photo.chunks():
                        f.write(chunk)
            elif request.POST.get('photo_url'):
                photo_url = request.POST.get('photo_url')
                response = requests.get(photo_url)
                if response.status_code == 200:
                    file_name = os.path.basename(photo_url.split("?")[0]) or "url_image.jpg"
                    relative_path = os.path.join('uploads', file_name)
                    file_path = os.path.join(settings.MEDIA_ROOT, relative_path)
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception("Не вдалося завантажити зображення за посиланням.")
            else:
                return render(request, 'verifier/faceAnalise.html', {'result': None})

            if relative_path and os.path.exists(file_path):
                analysis_result = DeepFace.analyze(file_path, actions=['age', 'gender', 'emotion'])
                result = analysis_result[0]
                result['img_path'] = relative_path
                if 'gender' in result and isinstance(result['gender'], dict):
                    result['gender'] = max(result['gender'], key=result['gender'].get)
        except ValueError as e:
            result = {"error": "Обличчя не знайдено!" if "Face could not be detected" in str(e) else f"Помилка даних: {str(e)}"}
        except Exception as e:
            result = {"error": f"Помилка: {str(e)}"}
    return render(request, 'verifier/faceAnalise.html', {'result': result, 'MEDIA_URL': settings.MEDIA_URL})


def user_list(request):
    query = request.GET.get('q')
    if query:
        users = UserProfile.objects.filter(username__icontains=query).order_by('-created_at')
    else:
        users = UserProfile.objects.all().order_by('-created_at')
    return render(request, 'verifier/user_list.html', {'users': users, 'query': query})


def delete_user(request, user_id):
    if request.method == 'POST':
        user = get_object_or_404(UserProfile, id=user_id)
        if user.photo:
            user.photo.delete(save=False)
        user.delete()
        return redirect('user_list')
    return redirect('user_list')


def login_page(request, username):
    user = get_object_or_404(UserProfile, username=username)
    request.session['face_attempts'] = 0
    return render(request, 'verifier/login.html', {'user': user})


def home(request):
    return render(request, 'verifier/home.html')


def logined_page(request, username):
    return render(request, 'verifier/logined_page.html', {'username': username})


def startpage(request):
    return render(request, 'verifier/startpage.html')


def magisterjob(request):
    return render(request, 'verifier/magisterjob.html')


def krystyna_view(request):
    return render(request, 'verifier/krystyna.html')