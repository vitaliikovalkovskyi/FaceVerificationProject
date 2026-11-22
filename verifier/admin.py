from django import forms
from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from .models import Attempt, Choice, Invite, Question, Test, UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("username", "personal_threshold", "embedding_photo_count", "created_at")
    search_fields = ("username",)
    exclude = ("face_embedding", "totp_secret")


@admin.register(Invite)
class InviteAdmin(admin.ModelAdmin):
    list_display = ("title", "link", "is_active", "expires_at", "registrations", "created_at")
    list_filter = ("is_active",)
    readonly_fields = ("registrations", "link")

    @admin.display(description="Посилання")
    def link(self, obj):
        if not obj.pk:
            return "—"
        url = reverse("invite", args=[obj.token])
        return format_html('<a href="{0}" target="_blank">{0}</a>', url)


class ChoiceInlineFormSet(forms.BaseInlineFormSet):
    """У кожного питання має бути рівно один правильний варіант."""

    def clean(self):
        super().clean()
        correct = sum(
            1 for form in self.forms
            if form.cleaned_data and not form.cleaned_data.get("DELETE") and form.cleaned_data.get("is_correct")
        )
        if correct != 1:
            raise forms.ValidationError("Позначте рівно один правильний варіант відповіді.")


class ChoiceInline(admin.TabularInline):
    model = Choice
    formset = ChoiceInlineFormSet
    extra = 4


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ("text", "test", "order")
    list_filter = ("test",)
    inlines = [ChoiceInline]


class QuestionInline(admin.TabularInline):
    model = Question
    fields = ("order", "text", "edit_choices")
    readonly_fields = ("edit_choices",)
    extra = 0

    @admin.display(description="Варіанти")
    def edit_choices(self, obj):
        if not obj.pk:
            return "збережіть тест, потім додайте варіанти"
        url = reverse("admin:verifier_question_change", args=[obj.pk])
        return format_html('<a href="{}">редагувати варіанти ({})</a>', url, obj.choices.count())


@admin.register(Test)
class TestAdmin(admin.ModelAdmin):
    list_display = ("title", "link", "questions_count", "is_active", "created_at")
    list_filter = ("is_active",)
    inlines = [QuestionInline]

    @admin.display(description="Питань")
    def questions_count(self, obj):
        return obj.questions.count()

    @admin.display(description="Посилання для студентів")
    def link(self, obj):
        url = reverse("test_detail", args=[obj.pk])
        return format_html('<a href="{0}" target="_blank">{0}</a>', url)


@admin.register(Attempt)
class AttemptAdmin(admin.ModelAdmin):
    list_display = ("user", "test", "score", "total", "created_at")
    list_filter = ("test",)
    readonly_fields = ("test", "user", "score", "total", "answers", "created_at")
