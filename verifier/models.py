from django.db import models

class UserProfile(models.Model):
    username = models.CharField(max_length=100, unique=True, verbose_name="Ім'я користувача")
    photo = models.ImageField(upload_to='users/', verbose_name="Еталонне фото")
    # НОВЕ ПОЛЕ: Секретний ключ для 2FA (32 символи)
    totp_secret = models.CharField(max_length=32, blank=True, null=True, verbose_name="2FA Secret")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username