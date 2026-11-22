"""
Demo use case «система проходження тестів».

1. Викладач надсилає посилання-запрошення /invite/<token>/ → наявна реєстрація (обличчя + 2FA).
2. Студент відкриває /tests/<id>/; якщо не увійшов — окреме вікно входу до тесту:
   вибір себе зі списку → вхід обличчям (liveness, динамічний поріг) або OTP
   (після 3 невдалих спроб обличчям — лише OTP).
3. Питання з одним правильним варіантом → «Завершити» → результат.
"""
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme

from .models import Attempt, Choice, Invite, Test, UserProfile
from .views import SESSION_USER_KEY, current_user, render_login


# ── Запрошення ───────────────────────────────────────────────────────────────
def invite(request, token):
    inv = Invite.objects.filter(token=token).first()
    if inv is None or not inv.is_valid():
        return render(request, 'verifier/invite_invalid.html', status=404)
    request.session['invite_id'] = inv.id
    return render(request, 'verifier/magisterjob.html', {'invite': inv})


def invite_done(request, token):
    inv = get_object_or_404(Invite, token=token)
    return render(request, 'verifier/invite_done.html', {'invite': inv})


# ── Вхід до тесту ────────────────────────────────────────────────────────────
def test_login(request, test_id):
    """Окреме вікно входу: список зареєстрованих студентів."""
    test = get_object_or_404(Test, pk=test_id, is_active=True)
    students = UserProfile.objects.order_by('username')
    return render(request, 'verifier/test_login.html', {'test': test, 'students': students})


def test_login_user(request, test_id, username):
    """Вхід обраного студента: обличчя або OTP; після успіху — повернення до тесту."""
    test = get_object_or_404(Test, pk=test_id, is_active=True)
    user = get_object_or_404(UserProfile, username=username)
    return render_login(
        request, user,
        next_url=reverse('test_detail', args=[test.pk]),
        back_url=reverse('test_login', args=[test.pk]),
    )


def logout_view(request):
    request.session.pop(SESSION_USER_KEY, None)
    next_url = request.GET.get('next', '')
    if not url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
        next_url = reverse('start')
    return redirect(next_url)


# ── Тест ─────────────────────────────────────────────────────────────────────
def test_detail(request, test_id):
    test = get_object_or_404(Test, pk=test_id, is_active=True)
    student = current_user(request)
    if student is None:
        return redirect('test_login', test_id=test.pk)

    questions = list(test.questions.prefetch_related('choices'))

    if request.method == 'POST':
        answers, score = {}, 0
        for q in questions:
            raw = request.POST.get(f'q_{q.pk}')
            choice = Choice.objects.filter(pk=raw, question=q).first() if raw and raw.isdigit() else None
            if choice:
                answers[str(q.pk)] = choice.pk
                score += int(choice.is_correct)
        attempt = Attempt.objects.create(test=test, user=student, score=score, total=len(questions), answers=answers)
        return redirect('attempt_result', attempt_id=attempt.pk)

    last = Attempt.objects.filter(test=test, user=student).first()
    return render(request, 'verifier/test_detail.html', {
        'test': test, 'questions': questions, 'student': student, 'last_attempt': last,
        'attempts_count': Attempt.objects.filter(test=test, user=student).count(),
    })


def attempt_result(request, attempt_id):
    attempt = get_object_or_404(Attempt.objects.select_related('test', 'user'), pk=attempt_id)
    student = current_user(request)
    if student is None or student.pk != attempt.user_id:
        return HttpResponseForbidden("Результат доступний лише студенту, який проходив тест.")
    percent = round(attempt.score * 100 / attempt.total) if attempt.total else 0
    return render(request, 'verifier/test_result.html', {'attempt': attempt, 'student': student, 'percent': percent})
