# Liveness Lab

Міні-застосунок для експериментальної оцінки модуля перевірки живості (liveness):
порівняння **baseline** (логіка з `login.html` дипломного проєкту) та **improved**
(EAR з індивідуальним калібруванням, випадковий порядок усіх дій, таймаут кроку).
Метрики за ISO/IEC 30107-3: APCER (по кожному типу атаки), BPCER, ACER.

## Запуск
```
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Відкрити http://localhost:5000 (камера працює на localhost без HTTPS).

## Протокол експерименту
Для кожної версії: 20 спроб genuine + по 10 спроб photo_screen і video_replay.
Обирайте тип спроби у списку перед стартом; результат пишеться у `data/attempts.csv`.
Помилково записану спробу можна видалити кнопкою «Видалити останній запис».
Підрахунок у терміналі: `python metrics.py`.

## Перенесення в Django
Уся логіка детекції — у `liveness_core.py` (`analyze_frame(jpeg_bytes) -> dict`),
не залежить від Flask: у Django викликається з view (`/detect`). JS-логіка версій —
у `static/index.html` (функції `runBaseline`, `runImproved`).