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