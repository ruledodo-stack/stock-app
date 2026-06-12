import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import time, os

KST = timezone(timedelta(hours=9))

st.set_page_config(page_title="단타·종가 AI 추천", page_icon="📈", layout="centered")

# Streamlit secrets에서 KRX 인증 정보 자동 로드
for _key in ["KRX_ID", "KRX_PW", "ANTHROPIC_API_KEY"]:
    if _key not in os.environ:
        try:
            os.environ[_key] = st.secrets[_key]
        except Exception:
            pass

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&display=swap');
* { font-family: 'Noto Sans KR', sans-serif !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #080808 !important; }
.block-container { max-width: 700px !important; margin: 0 auto !important; padding: 4rem 1rem 1.5rem !important; }
[data-testid="stHeader"] { display: none !important; }
.stButton > button { background: #111 !important; color: #444 !important; border: 1px solid #1a1a1a !important; border-radius: 8px !important; }
div[data-testid="stSpinner"] > div { border-top-color: #ff6b6b !important; }
</style>
""",
    unsafe_allow_html=True,
)


def now_kst():
    return datetime.now(KST)


def prev_trading_day(d):
    for i in range(1, 8):
        t = d - timedelta(days=i)
        if t.weekday() < 5:
            return t.strftime("%Y%m%d")
    return (d - timedelta(days=1)).strftime("%Y%m%d")


def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        return round(100 - (100 / (1 + rs.iloc[-1])), 1)
    except:
        return 50.0


def rsi_score(rsi, mode):
    if mode == "danta":
        if rsi >= 80 or rsi <= 30:
            return -9999
        if 55 <= rsi <= 68:
            return 10
        if 45 <= rsi < 55:
            return 5
        if 68 < rsi <= 75:
            return 2
        return 0
    else:
        if rsi > 68 or rsi < 38:
            return -9999
        if 45 <= rsi <= 65:
            return 10
        if 38 <= rsi < 45:
            return 6
        return 2


@st.cache_data(ttl=180)
def load_bulk_data(date_key):
    try:
        from pykrx import stock
        now = now_kst()
        today = now.strftime("%Y%m%d")
        yesterday = prev_trading_day(now)
        rows = []
        for market in ["KOSPI", "KOSDAQ"]:
            try:
                dt = stock.get_market_ohlcv_by_ticker(today, market=market)
                dp = stock.get_market_ohlcv_by_ticker(yesterday, market=market)
                if dt.empty or dp.empty:
                    continue
                for ticker in dt.index.intersection(dp.index):
                    t, p = dt.loc[ticker], dp.loc[ticker]
                    if p["거래량"] <= 0 or t["거래량"] <= 0 or p["종가"] <= 0:
                        continue
                    gap = (t["시가"] - p["종가"]) / p["종가"] * 100
                    chg = (t["종가"] - p["종가"]) / p["종가"] * 100
                    vol = t["거래량"] / p["거래량"]
                    val = t.get("거래대금", 0) / 1e8
                    high_ratio = (t["종가"] / t["고가"] * 100) if t["고가"] > 0 else 0
                    rows.append(
                        {
                            "종목코드": ticker,
                            "시장": market,
                            "현재가": int(t["종가"]),
                            "갭": round(gap, 2),
                            "거래량배율": round(vol, 1),
                            "등락률": round(chg, 2),
                            "거래대금억": round(val, 1),
                            "종가고가비율": round(high_ratio, 1),
                        }
                    )
            except:
                continue
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_indicators(ticker, today):
    try:
        from pykrx import stock
        now = now_kst()
        start = (now - timedelta(days=50)).strftime("%Y%m%d")
        hist = stock.get_market_ohlcv_by_date(start, today, ticker)
        if hist is None or len(hist) < 20:
            return 50.0, False, False, ticker
        c = hist["종가"]
        rsi = calc_rsi(c)
        ma_ok = c.rolling(5).mean().iloc[-1] > c.rolling(20).mean().iloc[-1]
        macd_ok = (c.ewm(span=12).mean().iloc[-1] - c.ewm(span=26).mean().iloc[-1]) > 0
        name = stock.get_market_ticker_name(ticker)
        return rsi, ma_ok, macd_ok, name
    except:
        return 50.0, False, False, ticker


@st.cache_data(ttl=600)
def get_comment(name, vol, gap, chg, rsi, mode):
    try:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return f"거래량 {vol}배 + 갭 {gap}% → {'눌림목 진입 검토' if mode == 'danta' else '다음날 갭상승 기대'}"
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=60,
            messages=[
                {
                    "role": "user",
                    "content": f"{name}: 거래량{vol}x 갭{gap}% 등락{chg}% RSI{rsi}. "
                    f"{'마하세븐 단타' if mode == 'danta' else '종가배팅'} 1줄분석 20자이내",
                }
            ],
        )
        return msg.content[0].text.strip()
    except:
        return f"{'단타' if mode == 'danta' else '종가배팅'} 주목"


def get_picks(df, today, mode):
    if df.empty:
        return []
    if mode == "danta":
        cands = df[
            (df["갭"] >= 1.0)
            & (df["갭"] <= 10)
            & (df["거래량배율"] >= 3.0)
            & (df["등락률"] > 0)
            & (df["현재가"] >= 1000)
            & (df["현재가"] <= 200000)
            & (df["거래대금억"] >= 50)
        ].copy()
    else:
        cands = df[
            (df["거래대금억"] >= 200)
            & (df["등락률"] > 0)
            & (df["현재가"] >= 1000)
            & (df["종가고가비율"] >= 85)
        ].copy()
    if cands.empty:
        return []
    pre_col = "거래량배율" if mode == "danta" else "거래대금억"
    top20 = cands.nlargest(20, pre_col)
    results = []
    for _, row in top20.iterrows():
        rsi, ma_ok, macd_ok, name = get_indicators(row["종목코드"], today)
        rs = rsi_score(rsi, mode)
        if rs == -9999:
            continue
        if mode == "jongga" and not ma_ok:
            continue
        if mode == "danta":
            score = (
                row["거래량배율"] * 0.35
                + row["갭"] * 0.25
                + row["등락률"] * 0.20
                + rs * 0.20
            )
            if macd_ok:
                score *= 1.1
            if ma_ok:
                score *= 1.05
        else:
            score = (
                row["거래대금억"] * 0.3
                + row["등락률"] * 0.2
                + rs * 0.3
                + (30 if ma_ok else 0) * 0.2
            )
        results.append(
            {
                **row.to_dict(),
                "RSI": rsi,
                "MA정배열": ma_ok,
                "MACD양수": macd_ok,
                "종목명": name,
                "점수": round(score, 2),
            }
        )
    results.sort(key=lambda x: x["점수"], reverse=True)
    return results[:5]


def card(s, rank, mode):
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#777", "#555"]
    m = medals[rank] if rank < 5 else str(rank + 1)
    c = colors[rank] if rank < 5 else "#444"
    accent = "#ff6b6b" if mode == "danta" else "#4488ff"
    comment = get_comment(
        s["종목명"], s["거래량배율"], s["갭"], s["등락률"], s["RSI"], mode
    )
    badges = []
    if s["MA정배열"]:
        badges.append("MA✅")
    else:
        badges.append("MA⚠️")
    if s["MACD양수"]:
        badges.append("MACD✅")
    extra = (
        f"<span style='color:#2244aa;'>{s['거래대금억']}억</span>"
        if mode == "jongga"
        else ""
    )
    st.markdown(
        f"""
<div style='background:#0c0c0c;border-radius:14px;padding:18px 20px;margin:8px 0;
     border-left:4px solid {c};border-top:1px solid #151515;'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;'>
    <div>
      <span style='font-size:1.1rem;'>{m}</span>
      <span style='font-size:1.2rem;font-weight:900;color:#fff;margin-left:7px;'>{s["종목명"]}</span>
      <span style='color:#666666;font-size:0.72rem;margin-left:6px;'>{s["종목코드"]}·{s["시장"]}</span>
    </div>
    <div style='text-align:right;'>
      <div style='font-size:1.5rem;font-weight:900;color:{accent};'>+{s["등락률"]}%</div>
      <div style='color:#777777;font-size:0.75rem;'>₩{s["현재가"]:,}</div>
    </div>
  </div>
  <div style='display:flex;gap:18px;padding:10px 0;border-top:1px solid #141414;border-bottom:1px solid #141414;'>
    <div style='text-align:center;'>
      <div style='color:{accent};font-weight:700;'>{s["거래량배율"]}x</div>
      <div style='color:#666666;font-size:0.68rem;'>거래량</div>
    </div>
    <div style='text-align:center;'>
      <div style='color:{accent};font-weight:700;'>+{s["갭"]}%</div>
      <div style='color:#666666;font-size:0.68rem;'>갭</div>
    </div>
    <div style='text-align:center;'>
      <div style='color:#ff9900;font-weight:700;'>{s["RSI"]}</div>
      <div style='color:#666666;font-size:0.68rem;'>RSI</div>
    </div>
    <div style='color:#888888;font-size:0.75rem;margin:auto 0;'>{" · ".join(badges)}{extra}</div>
  </div>
  <div style='margin-top:9px;color:#888888;font-size:0.8rem;font-style:italic;'>💡 {comment}</div>
</div>""",
        unsafe_allow_html=True,
    )


def no_pick_box(mode):
    color = "#ff6b6b" if mode == "danta" else "#4488ff"
    label = "단타" if mode == "danta" else "종가배팅"
    cond = (
        "갭1%↑ + 거래량3배↑ + RSI50~75"
        if mode == "danta"
        else "거래대금200억↑ + RSI40~68 + MA정배열"
    )
    st.markdown(
        f"""
<div style='background:#0a0808;border:1px solid #1a1010;border-radius:12px;
     padding:18px 20px;text-align:center;margin-bottom:1rem;'>
  <div style='font-size:1.3rem;margin-bottom:5px;'>🙅</div>
  <div style='color:{color};font-weight:700;'>오늘 {label} 없음 — 쉬는 날</div>
  <div style='color:#1a1010;font-size:0.78rem;margin-top:5px;'>{cond} 동시 충족 종목 없음</div>
</div>""",
        unsafe_allow_html=True,
    )


def slot_header(time_label, color="#ff6b6b"):
    st.markdown(
        f"""
<div style='color:{color};font-weight:700;font-size:0.9rem;margin:14px 0 4px;
     padding:6px 12px;background:#0d0d0d;border-radius:8px;border-left:3px solid {color};'>
  ⏰ {time_label} 확정 추천
</div>""",
        unsafe_allow_html=True,
    )


def next_slot_badge(target, label, color="#ff6b6b"):
    rem = int((target - now_kst()).total_seconds())
    if rem <= 0:
        return
    m, s = divmod(rem, 60)
    st.markdown(
        f"""
<div style='background:#0a0a0a;border:1px solid #151515;border-radius:10px;
     padding:10px 16px;display:flex;justify-content:space-between;
     align-items:center;margin:6px 0;'>
  <div style='color:#888888;font-size:0.78rem;'>{label}</div>
  <div style='color:{color};font-weight:700;'>{m:02d}:{s:02d}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_slot(session_key, mode, time_label):
    picks = st.session_state.get(session_key, [])
    slot_header(time_label, "#ff6b6b" if mode == "danta" else "#4488ff")
    if not picks:
        no_pick_box(mode)
    else:
        for i, s in enumerate(picks):
            card(s, i, mode)


def countdown_box(rem_sec, label, color="#4CAF50"):
    h, r = divmod(rem_sec, 3600)
    m, s = divmod(r, 60)
    st.markdown(
        f"""
<div style='background:#0a0a0a;border:1px solid #151515;border-radius:14px;
     padding:40px 24px;text-align:center;margin:1rem 0;'>
  <div style='color:#666666;font-size:0.78rem;letter-spacing:2px;margin-bottom:12px;'>{label}</div>
  <div style='font-size:3.2rem;font-weight:900;color:{color};letter-spacing:-2px;'>
    {h:02d}:{m:02d}:{s:02d}
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def main():
    now = now_kst()
    today = now.strftime("%Y%m%d")
    t9    = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    t910  = now.replace(hour=9,  minute=10, second=0, microsecond=0)
    t920  = now.replace(hour=9,  minute=20, second=0, microsecond=0)
    t930  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    t1450 = now.replace(hour=14, minute=50, second=0, microsecond=0)
    t1530 = now.replace(hour=15, minute=30, second=0, microsecond=0)

    st.markdown(
        f"""
<div style='margin-bottom:1.2rem;'>
  <div style='font-size:1.7rem;font-weight:900;color:#fff;letter-spacing:-1px;'>📈 단타·종가 AI 추천</div>
  <div style='color:#666666;font-size:0.75rem;margin-top:3px;'>
    마하세븐·고명환·Ross Cameron 공식 · {now.strftime("%Y.%m.%d %H:%M:%S")} KST
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── 장전 ──
    if now < t9:
        countdown_box(int((t9 - now).total_seconds()), "장 시작까지", "#4CAF50")
        st.markdown(
            """
<div style='background:#0a0a0a;border:1px solid #131313;border-radius:10px;
     padding:14px 18px;color:#777777;font-size:0.78rem;line-height:1.9;'>
  🔴 <b style='color:#888888;'>단타</b> 9:10 · 9:20 · 9:30 총 3회 추천 · 갭1%↑ + 거래량3배↑ + RSI50~75<br>
  🌙 <b style='color:#888888;'>종가배팅</b> 14:50 확정 · 거래대금200억↑ + RSI40~68 + MA정배열<br>
  <b style='color:#444;'>📋 조건 미충족 시 추천 없음 (쉬는 날)</b>
</div>""",
            unsafe_allow_html=True,
        )
        time.sleep(1)
        st.rerun()
        return

    # ── 9:00~9:10 모니터링 ──
    if now < t910:
        rem = int((t910 - now).total_seconds())
        m, s = divmod(rem, 60)
        st.markdown(
            f"""
<div style='background:#0a1a0a;border:1px solid #1a3a1a;border-radius:12px;
     padding:18px 22px;margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center;'>
  <div>
    <div style='color:#4CAF50;font-weight:700;'>⚡ 장 시작 · 분석 중</div>
    <div style='color:#1a2a1a;font-size:0.78rem;margin-top:3px;'>9:10 · 9:20 · 9:30 단타 3회 추천 예정</div>
  </div>
  <div style='text-align:right;'>
    <div style='color:#4CAF50;font-size:1.8rem;font-weight:900;'>{m:02d}:{s:02d}</div>
    <div style='color:#1a2a1a;font-size:0.72rem;'>9:10 첫 추천</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )
        time.sleep(1)
        st.rerun()
        return

    # ── 단타 섹션 (9:10 이후) ──
    st.markdown(
        """
<div style='color:#ff6b6b;font-weight:700;font-size:1rem;margin-bottom:4px;'>🔴 단타 추천</div>
<div style='color:#666666;font-size:0.75rem;margin-bottom:8px;'>
  갭1%↑ · 거래량3배↑ · RSI50~75 · MA정배열 · 9:10 / 9:20 / 9:30 각 1회 확정
</div>""",
        unsafe_allow_html=True,
    )

    with st.spinner("데이터 로딩 중..."):
        df = load_bulk_data(today)

    if df.empty:
        st.error("⚠️ 시장 데이터 로드 실패 — 네트워크 오류 또는 장 휴장일")
    else:
        # 9:10 슬롯 (항상 표시)
        if "danta_910" not in st.session_state:
            with st.spinner("9:10 분석 중..."):
                st.session_state["danta_910"] = get_picks(df, today, "danta")
        render_slot("danta_910", "danta", "9:10")

        # 9:20 슬롯
        if now >= t920:
            if "danta_920" not in st.session_state:
                with st.spinner("9:20 분석 중..."):
                    st.session_state["danta_920"] = get_picks(df, today, "danta")
            render_slot("danta_920", "danta", "9:20")
        else:
            next_slot_badge(t920, "다음 단타 추천까지 (9:20)")

        # 9:30 슬롯
        if now >= t930:
            if "danta_930" not in st.session_state:
                with st.spinner("9:30 분석 중..."):
                    st.session_state["danta_930"] = get_picks(df, today, "danta")
            render_slot("danta_930", "danta", "9:30")
        elif now >= t920:
            next_slot_badge(t930, "다음 단타 추천까지 (9:30)")

    st.markdown(
        "<hr style='border:1px solid #111;margin:1.5rem 0;'>",
        unsafe_allow_html=True,
    )

    # ── 종가배팅 섹션 ──
    if now < t1450:
        rem = int((t1450 - now).total_seconds())
        h2, r2 = divmod(rem, 3600)
        m2, s2 = divmod(r2, 60)
        st.markdown(
            f"""
<div style='background:#08080f;border:1px solid #15152a;border-radius:12px;
     padding:16px 20px;display:flex;justify-content:space-between;align-items:center;'>
  <div>
    <div style='color:#4488ff;font-weight:700;'>🌙 종가배팅 대기 중</div>
    <div style='color:#15152a;font-size:0.75rem;margin-top:3px;'>거래대금200억↑ · RSI40~68 · MA정배열</div>
  </div>
  <div style='text-align:right;'>
    <div style='color:#4488ff;font-size:1.7rem;font-weight:900;'>{h2:02d}:{m2:02d}:{s2:02d}</div>
    <div style='color:#15152a;font-size:0.72rem;'>14:50 확정</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )
    elif now < t1530:
        st.markdown(
            """
<div style='color:#4488ff;font-weight:700;font-size:1rem;margin-bottom:4px;'>🌙 종가배팅 추천 · 14:50 확정</div>
<div style='color:#666666;font-size:0.75rem;margin-bottom:12px;'>
  거래대금200억↑ · RSI40~68 · MA정배열 필수 · 다음날 9시 매도
</div>""",
            unsafe_allow_html=True,
        )
        if not df.empty:
            if "jongga_1450" not in st.session_state:
                with st.spinner("종가배팅 분석 중..."):
                    st.session_state["jongga_1450"] = get_picks(df, today, "jongga")
            picks_j = st.session_state.get("jongga_1450", [])
            if not picks_j:
                no_pick_box("jongga")
            else:
                for i, s in enumerate(picks_j):
                    card(s, i, "jongga")
    else:
        st.markdown(
            """
<div style='background:#080808;border:1px solid #111;border-radius:10px;
     padding:14px;text-align:center;color:#666666;font-size:0.82rem;'>
  장 마감 (15:30) · 내일 오전 9:10에 새 추천이 표시됩니다
</div>""",
            unsafe_allow_html=True,
        )

    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        for k in ["danta_910", "danta_920", "danta_930", "jongga_1450"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown(
        f"<div style='color:#111;font-size:0.68rem;margin-top:0.5rem;text-align:center;'>손절 -2% · 단타 +1~3% · 종가배팅 다음날 시초 매도 · {now.strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )
    time.sleep(60)
    st.rerun()


if __name__ == "__main__":
    main()
