# genetic_bt: بک‌تست استراتژی‌های ژنتیکی (فقط بک‌تست)

این ریپو فقط بک‌تست است. هیچ بخش لایو، تلگرام، یا ارسال سفارش ندارد.
**همه‌ی هزینه‌ها صفر است** (کارمزد، slippage، funding).

## ساختار داده
فایل‌های OKX باید به این شکل باشند (ستون‌ها: `ts, open, high, low, close, volume`؛ `ts` زمان **open** کندل به میلی‌ثانیه UTC):

```
data/okx/15m/BTC-USDT-SWAP.csv.gz
data/okx/1h/BTC-USDT-SWAP.csv.gz
data/okx/4h/BTC-USDT-SWAP.csv.gz
... (برای همه‌ی نمادها، نام فایل = نماد)
```
مجموعه‌ی نمادها در سه پوشه باید یکی باشد. برای خانواده‌ی BTC_ALIGNED_PULLBACK باید دقیقاً یک نماد `BTC-USDT-SWAP` در داده باشد.

## نصب
```
pip install -r requirements.txt
```

## ۱) تست سلامت (داده‌ی مصنوعی، بدون داده‌ی واقعی)
```
python -m unittest discover -s tests -t .
```

## ۲) ریپلی قهرمان‌های فریزشده (اولین بک‌تست)
نیاز به `frozen_manifest.json` (خروجی `freeze_candidates` از اجرای ژنتیک).
```
python -m genetic_bt.run_frozen_replay --data-root data/okx --manifest frozen_manifest.json --workers 6
```
- ۸۵٪ اول = development (درون‌نمونه، توصیفی). ۱۵٪ آخر = holdout.
- اگر داده‌ات کمتر از ۷۰۰ روز مشترک دارد: `--min-history-days 650` (و در گزارش ذکر کن).
- تست سریع روی ۶۰ روز آخر: `--smoke-days 60`.
- خروجی: `results/frozen_replay/frozen_replay.json` و `.csv`.

## ۳) بک‌تست پویا (انتخاب سیگنال هر چرخه‌ی ۱۵ دقیقه‌ای، با سقف پوزیشن پرتفوی)
متریک تاریخی هر کاندیدا فقط از foldهای دوره‌ی development محاسبه می‌شود و انتخاب روی ۱۵٪ آخر (holdout) اجرا می‌شود.
```
python -m genetic_bt.run_dynamic_backtest --data-root data/okx --manifest frozen_manifest.json --selector score  --workers 6
python -m genetic_bt.run_dynamic_backtest --data-root data/okx --manifest frozen_manifest.json --selector random --workers 6
```
حتماً هر دو را اجرا و مقایسه کن: اگر `score` از `random` (انتخاب تصادفی بین کاندیدهای مجاز) بهتر نباشد، منطق انتخاب ارزشی اضافه نکرده است.
- `--symbols ETH-USDT-SWAP SOL-USDT-SWAP` برای محدود کردن نمادها (پیش‌فرض: نمادهای مانیفست).
- `--min-trades 30` حداقل تعداد معامله‌ی کاندیدا در foldهای development (برای داده‌ی کوتاه‌تر کمش کن).
- خروجی: `results/dynamic_backtest/<selector>/` شامل `summary.json`, `trades.csv`, `candidate_metrics.csv`, `equity_curve.csv`.

## قواعد شبیه‌سازی (بدون تغییر نسبت به کد اصلی)
- فقط کندل‌های بسته‌شده؛ تصمیم در لحظه‌ی بسته شدن تایم‌فریم رویداد؛ ورود در **open کندل ۱۵ دقیقه‌ی بعد**.
- SL و TP به‌صورت **فاصله** از قیمت ورود واقعی ساخته می‌شوند. اگر هر دو در یک کندل بخورند، SL اول.
- هر کاندیدا همزمان یک پوزیشن؛ ریسک = `risk_percent` از اکوییتی فعلی.
- سیگنال تکراری (همان امضا) فقط یک بار حساب می‌شود.
- افت سرمایه فقط روی معاملات بسته‌شده است.

## تغییرات نسبت به فایل‌های اصلی
- حذف: `make_cycle_evaluator`، `run_symbol_cycle`/checkpoint، `EvolutionConfig`، هر بخش لایو.
- اضافه: `manifest.py` (جایگزین `forward_evaluator.load_frozen_manifest` که نبود)، `candidate_pool.py`، `window.py`.
- ریپلی: `EXPECTED_SYMBOLS=76` و عدد `11` هاردکد حذف شد؛ import از داخل حلقه خارج شد؛ پنجره‌ی ۷۰۰ روز قابل تنظیم شد؛ اجرای موازی.
- سرعت: محاسبه‌ی اندیکاتورها بین کاندیدهای یک نماد مشترک است (نتیجه عیناً یکسان؛ در تست‌ها با کد اصلی مقایسه شد).
- `SWING_OB` به SwingEngine بیرونی نیاز دارد که اینجا نیست؛ ژنوم‌های دارای آن در استخر پویا حذف می‌شوند و در ریپلی هشدار داده می‌شود.

## محدودیت‌ها
- ۸۵٪ اول درون‌نمونه است؛ فقط holdout شاهد نسبتاً مستقل است (به شرط اینکه اجرای ژنتیک هم آن بخش را ندیده باشد).
- ۱۱ کاندیدا همزمان بررسی می‌شوند، پس نتیجه‌ی خوب در holdout بخشی شانسی است.
- `setup_quality` کالیبره نیست؛ ستون‌های `candidate_metrics.csv` و `trades.csv` را بررسی کن.
