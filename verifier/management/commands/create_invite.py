"""
Створює посилання-запрошення на реєстрацію (одне на групу).

    python manage.py create_invite "КНм-21" [--days 30] [--host http://127.0.0.1:8000]
"""
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.urls import reverse
from django.utils import timezone

from verifier.models import Invite


class Command(BaseCommand):
    help = "Створює посилання-запрошення на реєстрацію для групи"

    def add_arguments(self, parser):
        parser.add_argument("title", help="назва групи, напр. «КНм-21»")
        parser.add_argument("--days", type=int, default=None, help="скільки днів діє посилання (без обмеження, якщо не вказано)")
        parser.add_argument("--host", default="http://127.0.0.1:8000", help="адреса сервера для повного посилання")

    def handle(self, *args, **opts):
        expires = timezone.now() + timedelta(days=opts["days"]) if opts["days"] else None
        invite = Invite.objects.create(title=opts["title"], expires_at=expires)
        url = opts["host"].rstrip("/") + reverse("invite", args=[invite.token])
        until = f" (діє до {timezone.localtime(expires):%d.%m.%Y %H:%M})" if expires else ""
        self.stdout.write(self.style.SUCCESS(f"Запрошення «{invite.title}»{until}:"))
        self.stdout.write(url)
