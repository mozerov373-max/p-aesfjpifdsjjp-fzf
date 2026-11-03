#!/usr/bin/env python3
"""
AES scanner — Altai Enduro Scalper.
Сканирует EURUSD, GBPUSD, DXY каждую минуту с 13:00 до 19:00 (Барнаул, UTC+7)
и шлёт сообщения в Telegram.

Сетап №1: свип Asia High/Low -> MSS на M5 -> 5м FVG (имбаланс). Сообщение сразу при
          появлении имбаланса, НЕ на ретесте. SMT-дивергенция не используется.
Сетап №2: выход из широкого азиатского бокса без свипа + часовой FVG по ходу.
          Сообщение, когда цена зашла (протестировала) в часовой имбаланс.

Запуск:  python scanner.py          # рабочий режим (ждёт окно 13:00-19:00)
         python scanner.py --test   # тестовое сообщение в Telegram
         python scanner.py --once   # одна проверка сейчас, без ограничения по времени
"""
import os, sys, time, argparse, traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ============================ НАСТРОЙКИ ============================
TZ = timezone(timedelta(hours=7))          # Барнаул
SCAN_START_H, SCAN_END_H = 13, 19          # окно сканирования (местное время)
ASIA_START_H, ASIA_END_H = 4, 11           # азиатский бокс по ТЗ
KILLZONE_ONLY = False                      # True = слать сетапы только 14:30-16:30
KILLZONE = ((14, 30), (16, 30))
BALANCE = float(os.getenv("BALANCE", "5000"))
RISK_PCT = 1.0
INTERVAL_SEC = 60
FRESH_MIN = 20                             # сигнал считается свежим, если ему не больше N минут
SWEEP_SL_BUFFER_PIPS = 2                   # запас за экстремумом свипа
SETUP2_SL_PIPS = 3.5                       # 3-4 пипса за противоположной границей
H1_MAGNET_PIPS = 5                         # противоположный H1 FVG больше 5 пипсов = магнит
SKIP_ON_H1_MAGNET = False                  # True = не слать, False = слать с предупреждением
H1_MIN_FVG_PIPS = 3                        # минимальный часовой FVG для сетапа №2

SYMBOLS = {
    # ticker Yahoo, размер пипса, мин. 5м-имбаланс (пипсы), "широкий" азиатский диапазон (пипсы)
    "EURUSD": dict(ticker="EURUSD=X", pip=0.0001, min_fvg=2, wide=40, usd_lot=True),
    "GBPUSD": dict(ticker="GBPUSD=X", pip=0.0001, min_fvg=2, wide=50, usd_lot=True),
    "DXY":    dict(ticker="DX-Y.NYB", pip=0.01,   min_fvg=2, wide=35, usd_lot=False),
}
# ===================================================================

TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
sent_keys = set()


def log(*a):
    print(datetime.now(TZ).strftime("%H:%M:%S"), *a, flush=True)


def telegram(text):
    if not TOKEN or not CHAT_ID:
        log("[нет TELEGRAM_TOKEN/CHAT_ID] ", text)
        return False
    for _ in range(3):
        try:
            r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                              data={"chat_id": CHAT_ID, "text": text}, timeout=15)
            if r.ok:
                return True
            log("Telegram error", r.status_code, r.text[:200])
        except Exception as e:
            log("Telegram exception", e)
        time.sleep(3)
    return False


# ----------------------------- данные -----------------------------
def fetch(ticker):
    import yfinance as yf
    df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"нет данных {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.columns = ["o", "h", "l", "c"]
    idx = df.index
    df.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).tz_convert(TZ)
    return df[~df.index.duplicated(keep="last")].sort_index()


def closed_only(df, now):
    return df[df.index + timedelta(minutes=5) <= now]


def mirror(df):
    return pd.DataFrame({"o": -df.o, "h": -df.l, "l": -df.h, "c": -df.c}, index=df.index)


def swing_lows(L, left=2, right=2):
    out = []
    for i in range(left, len(L) - right):
        if all(L[i] < L[i - k] for k in range(1, left + 1)) and \
           all(L[i] < L[i + k] for k in range(1, right + 1)):
            out.append(i)
    return out


# --------------------- сетап №1 (в «шорт»-ориентации) ---------------------
def detect_short_reversal(df, asia_end, AH, AL, pip, min_fvg):
    """Свип Asia High -> MSS вниз (закрытие тела ниже защищённого свинг-лоу) -> медвежий FVG."""
    H, L, C = df.h.values, df.l.values, df.c.values
    n = len(df)
    pos = [i for i, t in enumerate(df.index) if t >= asia_end]
    if not pos:
        return None
    start = pos[0]
    s = next((i for i in range(start, n) if H[i] > AH), None)
    if s is None:
        return None
    piv = swing_lows(L)
    ext = s
    mss = ref = None
    for j in range(s, n):
        if H[j] > H[ext]:
            ext = j
        if j == ext:
            continue
        cands = [p for p in piv if p + 2 <= ext and p >= ext - 48]
        if not cands:
            continue
        r = cands[-1]
        if C[j] < L[r] and C[j] < AH:       # тело закрылось ниже свинга и вернулось в бокс
            mss, ref = j, r
            break
    if mss is None:
        return None
    best = None
    for k in range(max(ext + 1, 2), min(mss + 3, n)):
        top, bot = L[k - 2], H[k]
        if top - bot >= min_fvg * pip:
            if best is None or (top - bot) >= (best[1] - best[2]):
                best = (k, top, bot)
    if best is None:
        return None
    k, top, bot = best
    return dict(ext=ext, ext_price=H[ext], mss=mss, mss_level=L[ref], fvg_k=k,
                fvg_top=top, fvg_bot=bot, fvg_time=df.index[k], ext_time=df.index[ext])


# ----------------------------- H1 FVG -----------------------------
def to_h1(df, now):
    h1 = df.resample("1h").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return h1[h1.index + timedelta(hours=1) <= now]


def h1_fvgs(h1):
    res = []
    H, L = h1.h.values, h1.l.values
    for i in range(2, len(h1)):
        if H[i - 2] < L[i]:
            res.append(dict(dir="up", bot=H[i - 2], top=L[i], mid_t=h1.index[i - 1],
                            end_t=h1.index[i] + timedelta(hours=1)))
        elif L[i - 2] > H[i]:
            res.append(dict(dir="down", bot=H[i], top=L[i - 2], mid_t=h1.index[i - 1],
                            end_t=h1.index[i] + timedelta(hours=1)))
    return res


def opposing_h1_magnet(fvgs, full, direction, entry, tp, pip):
    """Незаполненный H1 FVG противоположного направления между входом и целью, > 5 пипсов."""
    for g in fvgs:
        if (g["top"] - g["bot"]) / pip <= H1_MAGNET_PIPS:
            continue
        after = full[full.index >= g["end_t"]]
        if direction == "short" and g["dir"] == "up" and tp < g["top"] < entry:
            if after.empty or after.l.min() > g["bot"]:
                return g
        if direction == "long" and g["dir"] == "down" and entry < g["bot"] < tp:
            if after.empty or after.h.max() < g["top"]:
                return g
    return None


def lot_text(cfg, risk_pips):
    if not cfg["usd_lot"] or risk_pips <= 0:
        return ""
    lots = BALANCE * RISK_PCT / 100 / (risk_pips * 10)
    return f"\nРиск {RISK_PCT:.0f}% = ${BALANCE*RISK_PCT/100:.0f} → ~{lots:.2f} лота"


def fmt(p, pip):
    return f"{p:.5f}" if pip < 0.001 else f"{p:.2f}"


# ----------------------------- сетап №1 -----------------------------
def check_setup1(name, cfg, closed, full, asia_h, asia_l, asia_end, now):
    pip = cfg["pip"]
    mid = (asia_h + asia_l) / 2
    h1 = to_h1(full, now)
    fvgs = h1_fvgs(h1)
    post = closed[closed.index >= asia_end - timedelta(hours=4)]   # запас для свингов
    for direction in ("short", "long"):
        if direction == "short":
            r = detect_short_reversal(post, asia_end, asia_h, asia_l, pip, cfg["min_fvg"])
            if not r:
                continue
            ext, top, bot, tp = r["ext_price"], r["fvg_top"], r["fvg_bot"], asia_l
            entry = (top + bot) / 2
            sl = ext + SWEEP_SL_BUFFER_PIPS * pip
        else:
            m = detect_short_reversal(mirror(post), asia_end, -asia_l, -asia_h, pip, cfg["min_fvg"])
            if not m:
                continue
            r = m
            ext = -m["ext_price"]
            top, bot = -m["fvg_bot"], -m["fvg_top"]          # top > bot в обычных ценах
            tp = asia_h
            entry = (top + bot) / 2
            sl = ext - SWEEP_SL_BUFFER_PIPS * pip
        key = f"S1-{name}-{direction}-{r['ext_time']:%Y%m%d%H%M}"
        if key in sent_keys:
            continue
        age = now - (r["fvg_time"] + timedelta(minutes=5))
        if age > timedelta(minutes=FRESH_MIN):
            continue
        last = full.c.iloc[-1]
        if (direction == "short" and (last >= sl or last <= tp)) or \
           (direction == "long" and (last <= sl or last >= tp)):
            continue
        risk = abs(entry - sl) / pip
        reward = abs(tp - entry) / pip
        magnet = opposing_h1_magnet(fvgs, full, direction, entry, tp, pip)
        if magnet and SKIP_ON_H1_MAGNET:
            sent_keys.add(key)
            continue
        side = "SHORT 🔴" if direction == "short" else "LONG 🟢"
        swept = "Asia High" if direction == "short" else "Asia Low"
        txt = (f"⚡ СЕТАП №1 (разворот Азии) — {name} {side}\n"
               f"Свип {swept}: {fmt(ext, pip)} → MSS на M5 → 5м имбаланс "
               f"{fmt(bot, pip)}–{fmt(top, pip)} ({(top-bot)/pip:.1f} п.)\n"
               f"Лимитка (0.5 FVG): {fmt(entry, pip)}\n"
               f"SL (за свипом): {fmt(sl, pip)} (~{risk:.1f} п.)\n"
               f"TP (граница Азии): {fmt(tp, pip)} (~{reward:.1f} п., R:R {reward/risk:.1f})\n"
               f"Midpoint Азии: {fmt(mid, pip)} — закрытие за ним против сделки → стоп -5 п.\n"
               f"Цена сейчас: {fmt(last, pip)}"
               + lot_text(cfg, risk))
        if magnet:
            txt += (f"\n⚠️ Противоположный H1 FVG {fmt(magnet['bot'], pip)}–{fmt(magnet['top'], pip)} "
                    f"({(magnet['top']-magnet['bot'])/pip:.0f} п.) на пути к цели — по ТЗ это SKIP")
        txt += "\n(SMT-дивергенцию и новости проверяй сам)"
        if telegram(txt):
            sent_keys.add(key)
            log("отправлен", key)


# ----------------------------- сетап №2 -----------------------------
def check_setup2(name, cfg, closed, full, asia_h, asia_l, asia_end, now):
    pip = cfg["pip"]
    if (asia_h - asia_l) / pip < cfg["wide"]:
        return
    post = closed[closed.index >= asia_end]
    p0 = next((i for i in range(len(post))
               if post.h.iloc[i] > asia_h or post.l.iloc[i] < asia_l), None)
    if p0 is None:
        return
    b = post.iloc[p0]
    if b.h > asia_h and b.l < asia_l:
        return
    if b.h > asia_h and b.c > asia_h:
        direction = "long"
    elif b.l < asia_l and b.c < asia_l:
        direction = "short"
    else:
        return          # первый выход был свипом (закрылись внутри) — это территория сетапа №1
    key = f"S2-{name}-{direction}-{now:%Y%m%d}"
    if key in sent_keys:
        return
    h1 = to_h1(full, now)
    want = "up" if direction == "long" else "down"
    cands = [g for g in h1_fvgs(h1)
             if g["dir"] == want and g["mid_t"] >= asia_end - timedelta(hours=1)
             and (g["top"] - g["bot"]) / pip >= H1_MIN_FVG_PIPS]
    for g in reversed(cands):
        after = full[full.index >= g["end_t"]]
        if after.empty:
            continue
        if direction == "long":
            touched = after[after.l <= g["top"]]
        else:
            touched = after[after.h >= g["bot"]]
        if touched.empty:
            continue
        t0 = touched.index[0]
        if now - t0 > timedelta(minutes=FRESH_MIN):
            continue          # тест был давно — момент упущен
        last = full.c.iloc[-1]
        entry = last
        if direction == "long":
            sl = asia_l - SETUP2_SL_PIPS * pip
        else:
            sl = asia_h + SETUP2_SL_PIPS * pip
        risk = abs(entry - sl) / pip
        # HTF-ликвидность: экстремум последних 5 дней
        if direction == "long":
            hi = full.h.max(); tp = hi if hi > entry else None
        else:
            lo = full.l.min(); tp = lo if lo < entry else None
        if tp is None:
            return
        reward = abs(tp - entry) / pip
        if reward < 2 * risk:
            log(f"{name} S2 пропуск: цель {reward:.0f} п. < 2×риск {risk:.0f} п.")
            sent_keys.add(key)
            return
        side = "LONG 🟢" if direction == "long" else "SHORT 🔴"
        txt = (f"⚡ СЕТАП №2 (продолжение) — {name} {side}\n"
               f"Азия {fmt(asia_l, pip)}–{fmt(asia_h, pip)} ({(asia_h-asia_l)/pip:.0f} п.), "
               f"выход без свипа.\n"
               f"Цена протестировала часовой FVG {fmt(g['bot'], pip)}–{fmt(g['top'], pip)}\n"
               f"Вход ~{fmt(entry, pip)}\n"
               f"SL: {fmt(sl, pip)} (~{risk:.0f} п., за противоположной границей Азии)\n"
               f"TP (HTF-ликвидность): {fmt(tp, pip)} (~{reward:.0f} п., R:R {reward/risk:.1f})"
               + lot_text(cfg, risk) +
               "\n(Тренд D1/H4 и новости проверь сам)")
        if telegram(txt):
            sent_keys.add(key)
            log("отправлен", key)
        return


# ----------------------------- основной цикл -----------------------------
def in_killzone(now):
    a, b = KILLZONE
    m = now.hour * 60 + now.minute
    return a[0] * 60 + a[1] <= m <= b[0] * 60 + b[1]


def scan_once(now=None):
    now = now or datetime.now(TZ)
    day = now.date()
    asia_start = datetime(day.year, day.month, day.day, ASIA_START_H, tzinfo=TZ)
    asia_end = datetime(day.year, day.month, day.day, ASIA_END_H, tzinfo=TZ)
    if KILLZONE_ONLY and not in_killzone(now):
        return 0
    errors = 0
    for name, cfg in SYMBOLS.items():
        try:
            full = fetch(cfg["ticker"])
            closed = closed_only(full, now)
            asia = closed[(closed.index >= asia_start) & (closed.index < asia_end)]
            if len(asia) < 40:
                log(name, "мало данных по Азии:", len(asia))
                continue
            ah, al = asia.h.max(), asia.l.min()
            check_setup1(name, cfg, closed, full, ah, al, asia_end, now)
            check_setup2(name, cfg, closed, full, ah, al, asia_end, now)
        except Exception:
            errors += 1
            log(name, "ошибка:", traceback.format_exc().splitlines()[-1])
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if args.test:
        ok = telegram("✅ AES scanner: тест связи прошёл. Сообщения будут приходить сюда.")
        sys.exit(0 if ok else 1)
    if args.once:
        scan_once()
        return

    t0 = datetime.now(TZ)
    day = t0.date()
    w_start = datetime(day.year, day.month, day.day, SCAN_START_H, tzinfo=TZ)
    w_end = datetime(day.year, day.month, day.day, SCAN_END_H, tzinfo=TZ)
    deadline = min(w_end, t0 + timedelta(hours=5, minutes=50))   # лимит GitHub — 6 часов
    if t0.weekday() >= 5 or t0 >= w_end:
        log("вне рабочего времени, выход")
        return
    if t0 < w_start:
        wait = (w_start - t0).total_seconds()
        log(f"жду начала окна {wait:.0f} сек")
        time.sleep(max(0, wait))
    telegram(f"🟢 Сканер запущен ({datetime.now(TZ):%H:%M} Барнаул). EURUSD / GBPUSD / DXY, "
             f"до {deadline:%H:%M}.")
    fails = 0
    while datetime.now(TZ) < deadline:
        started = time.time()
        errs = scan_once()
        fails = fails + 1 if errs == len(SYMBOLS) else 0
        if fails == 10:
            telegram("⚠️ Сканер: 10 минут подряд нет данных (проблема источника Yahoo).")
        time.sleep(max(1, INTERVAL_SEC - (time.time() - started)))
    telegram(f"🔴 Сканер завершил работу ({datetime.now(TZ):%H:%M}).")


if __name__ == "__main__":
    main()
