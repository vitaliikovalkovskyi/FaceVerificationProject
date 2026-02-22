import os
import base64
import requests
import shutil
import pyotp          # <--- ДОДАЙ ЦЕ (Для генерації кодів)
import qrcode         # <--- ДОДАЙ ЦЕ (Для малювання QR)
from io import BytesIO # <--- ДОДАЙ ЦЕ (Для збереження QR в пам'ять)
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.conf import settings
from django.core.files.base import ContentFile
from deepface import DeepFace
from .models import UserProfile

# ============================================================
# ФУНКЦІЯ ЗБЕРЕЖЕННЯ BASE64 ФОТО
# ============================================================
def save_base64_image(data_url, file_name):
    try:
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        
        # ЗМІНЕНО: тепер зберігаємо в MEDIA_ROOT/temp_captures/
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_captures')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(data)
        return file_path
    except Exception as e:
        print(f"Помилка збереження: {e}")
        return None

# ============================================================
# ПЕРЕВІРКА ОБЛИЧЧЯ (VERIFY)
# ============================================================
def verify_image(request):
    """
    **ЦЕ ДЛЯ ПЕРЕВІРКИ ОБЛИЧЧЯ**
    Приймає POST з base64 зображення, порівнює з testPhoto
    """
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)

    image_data = request.POST.get('image_data')
    if not image_data:
        return JsonResponse({"error": "No data"}, status=400)

    # Зберігаємо фото з браузера
    captured_path = save_base64_image(image_data, "captured_image.jpg")
    test_photo_path = os.path.join(settings.STATICFILES_DIRS[0], "testPhoto.jpg")

    if not os.path.exists(test_photo_path):
        return JsonResponse({
            "error": "no_reference", 
            "message": "Спочатку зробіть еталонне фото (зліва)!"
        }, status=400)

    try:
        result = DeepFace.verify(
            img1_path=test_photo_path,
            img2_path=captured_path,
            enforce_detection=True
        )

        is_correct = result["verified"]
        accuracy = 1 if is_correct else 0

        return JsonResponse({
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "accuracy": accuracy
        })

    except Exception as e:
        error_msg = str(e)

        if "Face could not be detected" in error_msg:
            return JsonResponse({
                "error": "no_face_detected",
                "message": "На фото не видно обличчя. Будь ласка, станьте перед камерою."
            }, status=400)

        if "img2_path" in error_msg:
            return JsonResponse({
                "error": "no_face_detected",
                "message": "Не вдалося розпізнати обличчя на фото для перевірки. Спробуйте ще раз."
            }, status=400)

        if "img1_path" in error_msg:
            return JsonResponse({
                "error": "no_reference",
                "message": "Проблема з еталонним фото. Будь ласка, перезніміть еталон (зліва)."
            }, status=400)

        print(f"Server Error: {error_msg}")
        return JsonResponse({"error": "server_error", "message": f"Помилка сервера: {error_msg}"}, status=500)

# ============================================================
# ЗАХОПЛЕННЯ TEST PHOTO
# ============================================================
def capture_test_photo(request):
    """
    **ЦЕ ДЛЯ ЗАХОПЛЕННЯ ТЕСТОВОГО ФОТО**
    Перевіряє наявність обличчя перед збереженням як еталон
    """
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
        return JsonResponse({
            "error": "no_face_detected",
            "message": "На еталонному фото не знайдено обличчя! Спробуйте ще раз."
        }, status=400)
    except Exception as e:
        return JsonResponse({"error": "server_error", "message": str(e)}, status=500)

# ============================================================
# FACE ANALYSIS
# ============================================================
def faceAnalise(request):
    """
    ЦЕ ДЛЯ АНАЛІЗУ ОБЛИЧЧЯ
    1. Очищає папку uploads.
    2. Зберігає нове фото.
    3. Аналізує.
    """
    result = None
    relative_path = None  
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')

    if request.method == 'POST':
        # --- КРОК 0: ОЧИЩЕННЯ ПАПКИ ---
        # Створюємо папку, якщо немає
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        else:
            # Видаляємо всі файли в папці uploads перед новим завантаженням
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path) # Видаляємо файл
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Не вдалося видалити {file_path}. Причина: {e}')

        # --- КРОК 1: ЗАВАНТАЖЕННЯ ---
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
                    file_name = os.path.basename(photo_url.split("?")[0])
                    # Якщо ім'я файлу пусте або дивне, даємо дефолтне
                    if not file_name or len(file_name) < 3:
                        file_name = "url_image.jpg"
                        
                    relative_path = os.path.join('uploads', file_name)
                    file_path = os.path.join(settings.MEDIA_ROOT, relative_path)

                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception("Не вдалося завантажити зображення за посиланням.")
            else:
                 return render(request, 'verifier/faceAnalise.html', {'result': None})

            # --- КРОК 2: АНАЛІЗ ЧЕРЕЗ DeepFace ---
            if relative_path and os.path.exists(file_path):
                # Аналізуємо
                analysis_result = DeepFace.analyze(file_path, actions=['age', 'gender', 'emotion'])
                result = analysis_result[0]
                
                # Важливо: передаємо шлях до файлу для HTML
                result['img_path'] = relative_path

                # Виправлення для gender
                if 'gender' in result and isinstance(result['gender'], dict):
                    gender = max(result['gender'], key=result['gender'].get)
                    result['gender'] = gender
                
        except ValueError as e:
            error_msg = str(e)
            if "Face could not be detected" in error_msg:
                result = {"error": "Обличчя не знайдено! Переконайтеся, що на фото чітко видно обличчя."}
                # Якщо помилка, файл все одно залишається в папці, щоб ти бачив, ЩО саме не спрацювало.
                # Він видалиться при наступному запиті.
            else:
                result = {"error": f"Помилка даних: {error_msg}"}
        
        except Exception as e:
            result = {"error": f"Помилка: {str(e)}"}

    return render(request, 'verifier/faceAnalise.html', {'result': result, 'MEDIA_URL': settings.MEDIA_URL})
# ============================================================
# ДОДАТКОВІ VIEW (малі коментарі)
# ============================================================
# ============================================================
# РЕЄСТРАЦІЯ - ЕТАП 1: Перевірка фото та Генерація QR
# ============================================================
# ============================================================
# РЕЄСТРАЦІЯ - ЕТАП 1: Перевірка фото та Генерація QR
# ============================================================
def register_step1(request):
    if request.method == 'POST':
        # ОЧИЩЕННЯ СЕСІЇ перед новою реєстрацією, щоб не було чужих фото
        keys_to_clear = ['reg_username', 'reg_secret', 'reg_temp_photo_path']
        for key in keys_to_clear:
            if key in request.session: del request.session[key]

        username = request.POST.get('username')
        if UserProfile.objects.filter(username=username).exists():
            return JsonResponse({"success": False, "error": "Користувач вже існує!"}, status=400)

        camera_data = request.POST.get('camera_data')
        file_upload = request.FILES.get('photo')
        
        temp_filename = f"temp_reg_{username}.jpg"
        
        if camera_data:
            # Використовуємо нашу оновлену функцію
            img_data = camera_data.split(';base64,')[1] if ';base64,' in camera_data else camera_data
            temp_path = save_base64_image(f"data:image/jpeg;base64,{img_data}", temp_filename)
        elif file_upload:
            temp_path = os.path.join(settings.MEDIA_ROOT, temp_filename)
            with open(temp_path, 'wb') as f:
                for chunk in file_upload.chunks(): f.write(chunk)
        else:
            return JsonResponse({"success": False, "error": "Фото не надано"}, status=400)

        # Валідація FaceID... (залишається як була)
        try:
            DeepFace.extract_faces(img_path=temp_path, detector_backend='retinaface', enforce_detection=True)
            
            # Зберігаємо в сесію ПРАВИЛЬНИЙ шлях
            request.session['reg_username'] = username
            request.session['reg_temp_photo_path'] = temp_path
            
            # Генерація QR... (залишається як була)
            secret = pyotp.random_base32()
            request.session['reg_secret'] = secret
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name="FaceID System")
            qr = qrcode.make(totp_uri)
            buffer = BytesIO()
            qr.save(buffer, format="PNG")
            qr_base64 = base64.b64encode(buffer.getvalue()).decode()

            return JsonResponse({"success": True, "qr_code": qr_base64, "message": "Фото прийнято!"})
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            return JsonResponse({"success": False, "error": "Обличчя не знайдено"}, status=400)


# ============================================================
# РЕЄСТРАЦІЯ - ЕТАП 2: Перевірка коду та Збереження в БД
# ============================================================
def register_step2(request):
    if request.method == 'POST':
        otp_code = request.POST.get('otp_code')
        
        # Отримуємо дані з сесії
        username = request.session.get('reg_username')
        secret = request.session.get('reg_secret')
        temp_path = request.session.get('reg_temp_photo_path')

        if not username or not secret:
             return JsonResponse({"success": False, "error": "Сесія закінчилась. Почніть спочатку."}, status=400)

        # 1. Перевіряємо код
        totp = pyotp.TOTP(secret)
        if not totp.verify(otp_code):
             return JsonResponse({"success": False, "error": "Невірний код! Спробуйте ще раз."}, status=400)

        # 2. Якщо код вірний -> Зберігаємо в БД
        try:
            new_user = UserProfile(username=username, totp_secret=secret)
            
            # Читаємо фото з тимчасового файлу і зберігаємо в модель
            with open(temp_path, 'rb') as f:
                img_content = ContentFile(f.read(), name=f"{username}.jpg")
                new_user.photo.save(f"{username}.jpg", img_content, save=True)
            
            # Очищаємо за собою
            if os.path.exists(temp_path): os.remove(temp_path)
            del request.session['reg_username']
            del request.session['reg_secret']
            del request.session['reg_temp_photo_path']

            return JsonResponse({"success": True, "message": "Реєстрація завершена успішно!"})
            
        except Exception as e:
            return JsonResponse({"success": False, "error": f"Помилка збереження: {str(e)}"}, status=500)

    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)


# ============================================================
# СПИСОК КОРИСТУВАЧІВ (Вхід)
# ============================================================
def user_list(request):
    """
    Виводить список користувачів з можливістю пошуку за іменем.
    """
    query = request.GET.get('q') # Отримуємо текст з пошукового рядка
    
    if query:
        # icontains = case-insensitive contains (шукає входження без урахування регістру)
        users = UserProfile.objects.filter(username__icontains=query).order_by('-created_at')
    else:
        users = UserProfile.objects.all().order_by('-created_at')
        
    return render(request, 'verifier/user_list.html', {'users': users, 'query': query})

def delete_user(request, user_id):
    if request.method == 'POST':
        # Знаходимо користувача або повертаємо 404 помилку
        user = get_object_or_404(UserProfile, id=user_id)
        
        # 1. Видаляємо файл фото з диска (якщо він є)
        if user.photo:
            user.photo.delete(save=False)
        
        # 2. Видаляємо запис з бази даних
        user.delete()
        
        # Повертаємось до списку
        return redirect('user_list')
    
    # Якщо хтось спробує зайти просто за посиланням без кнопки - перекидаємо назад
    return redirect('user_list')

def login_page(request, username):
    """Сторінка входу для конкретного користувача"""
    user = get_object_or_404(UserProfile, username=username)
    request.session['face_attempts'] = 0  # Скидаємо спроби
    return render(request, 'verifier/login.html', {'user': user})

def login_face_verify(request):
    """Етап 1: Біометрична перевірка з обов'язковим підрахунком спроб"""
    if request.method == 'POST':
        username = request.POST.get('username')
        image_data = request.POST.get('image_data')
        user = get_object_or_404(UserProfile, username=username)

        # 1. Отримуємо та збільшуємо лічильник спроб ОДРАЗУ
        attempts = request.session.get('face_attempts', 0) + 1
        request.session['face_attempts'] = attempts

        # Перевірка на ліміт
        is_blocked = attempts >= 3

        captured_path = save_base64_image(image_data, f"login_attempt_{username}.jpg")
        
        try:
            result = DeepFace.verify(
                img1_path=user.photo.path,
                img2_path=captured_path,
                detector_backend='retinaface', 
                enforce_detection=True
            )
            
            if result["verified"]:
                request.session['face_attempts'] = 0 # Скидаємо при успіху
                return JsonResponse({"success": True, "message": "Обличчя підтверджено!"})
            
            # Якщо обличчя видно, але воно не те
            error_type = "too_many_attempts" if is_blocked else "fail"
            msg = "3 невдалі спроби. Перехід на OTP." if is_blocked else f"Обличчя не збігається. Спроба {attempts} з 3."
            return JsonResponse({"success": False, "error": error_type, "message": msg})

        except Exception as e:
            # Якщо обличчя взагалі не знайшли — ЦЕ ТЕЖ РАХУЄТЬСЯ ЯК СПРОБА
            error_type = "too_many_attempts" if is_blocked else "no_face"
            msg = "3 невдалі спроби (обличчя не знайдено). Перехід на OTP." if is_blocked else f"Обличчя не знайдено. Спроба {attempts} з 3."
            return JsonResponse({"success": False, "error": error_type, "message": msg})

    return JsonResponse({"error": "Method not allowed"}, status=405)

def login_otp_verify(request):
    """Етап 2: Резервний вхід через OTP (після 3 помилок FaceID)"""
    if request.method == 'POST':
        otp_code = request.POST.get('otp_code')
        username = request.POST.get('username')
        user = get_object_or_404(UserProfile, username=username)

        totp = pyotp.TOTP(user.totp_secret)
        if totp.verify(otp_code):
            return JsonResponse({"success": True, "message": "Вхід успішний!"})
        else:
            return JsonResponse({"success": False, "error": "Невірний код."})
    return JsonResponse({"error": "Method not allowed"}, status=405)

# ============================================================
# VIEW З ПРОСТИМ РЕНДЕРОМ HTML
# ============================================================
def home(request):
    return render(request, 'verifier/home.html')

def logined_page(request, username):
    """Сторінка, яка відображається після успішної автентифікації"""
    return render(request, 'verifier/logined_page.html', {'username': username})

def startpage(request):
    return render(request, 'verifier/startpage.html')

def magisterjob(request):
    return render(request, 'verifier/magisterjob.html')

def krystyna_view(request):
    return render(request, 'verifier/krystyna.html')