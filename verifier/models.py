from django.db import models


class UserProfile(models.Model):
    username = models.CharField(max_length=100, unique=True, verbose_name="Ім'я користувача")
    photo = models.ImageField(upload_to='users/', verbose_name="Еталонне фото")
    totp_secret = models.CharField(max_length=32, blank=True, null=True, verbose_name="2FA Secret")
    created_at = models.DateTimeField(auto_now_add=True)

    # --- НОВІ ПОЛЯ ДЛЯ MULTI-REFERENCE EMBEDDING ---
    # Зберігаємо усереднений L2-нормалізований вектор ArcFace (512 float)
    face_embedding = models.JSONField(
        null=True, blank=True,
        verbose_name="Усереднений embedding обличчя"
    )
    # Персональний поріг, обчислений під час реєстрації
    personal_threshold = models.FloatField(
        null=True, blank=True, default=0.68,
        verbose_name="Персональний поріг верифікації"
    )
    # Кількість еталонних фото (для інформації)
    embedding_photo_count = models.IntegerField(
        default=1,
        verbose_name="Кількість фото при реєстрації"
    )

    def __str__(self):
        return self.username

# ============================================================
# DEMO USE CASE: СИСТЕМА ПРОХОДЖЕННЯ ТЕСТІВ
# ============================================================
import secrets

from django.utils import timezone


def _new_invite_token():
    return secrets.token_urlsafe(12)


class Invite(models.Model):
    """Посилання-запрошення на реєстрацію (одне на групу, багаторазове)."""
    title = models.CharField(max_length=200, verbose_name="Назва (група)")
    token = models.CharField(max_length=64, unique=True, default=_new_invite_token, verbose_name="Токен")
    is_active = models.BooleanField(default=True, verbose_name="Активне")
    expires_at = models.DateTimeField(null=True, blank=True, verbose_name="Діє до")
    registrations = models.PositiveIntegerField(default=0, verbose_name="Зареєструвалось")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Запрошення"
        verbose_name_plural = "Запрошення"

    def is_valid(self) -> bool:
        return self.is_active and (self.expires_at is None or self.expires_at > timezone.now())

    def __str__(self):
        return self.title


class Test(models.Model):
    title = models.CharField(max_length=200, verbose_name="Назва")
    description = models.TextField(blank=True, verbose_name="Опис")
    is_active = models.BooleanField(default=True, verbose_name="Активний")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Тест"
        verbose_name_plural = "Тести"

    def __str__(self):
        return self.title


class Question(models.Model):
    test = models.ForeignKey(Test, on_delete=models.CASCADE, related_name="questions", verbose_name="Тест")
    text = models.TextField(verbose_name="Питання")
    order = models.PositiveIntegerField(default=0, verbose_name="Порядок")

    class Meta:
        ordering = ["order", "id"]
        verbose_name = "Питання"
        verbose_name_plural = "Питання"

    def __str__(self):
        return self.text[:80]


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name="choices", verbose_name="Питання")
    text = models.CharField(max_length=300, verbose_name="Варіант відповіді")
    is_correct = models.BooleanField(default=False, verbose_name="Правильний")

    class Meta:
        ordering = ["id"]
        verbose_name = "Варіант відповіді"
        verbose_name_plural = "Варіанти відповідей"

    def __str__(self):
        return self.text


class Attempt(models.Model):
    """Спроба проходження тесту (кількість спроб не обмежена)."""
    test = models.ForeignKey(Test, on_delete=models.CASCADE, related_name="attempts", verbose_name="Тест")
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="attempts", verbose_name="Студент")
    score = models.PositiveIntegerField(verbose_name="Правильних відповідей")
    total = models.PositiveIntegerField(verbose_name="Усього питань")
    answers = models.JSONField(default=dict, verbose_name="Відповіді (question_id → choice_id)")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Завершено")

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Спроба"
        verbose_name_plural = "Спроби"

    def __str__(self):
        return f"{self.user.username} — {self.test.title}: {self.score}/{self.total}"
