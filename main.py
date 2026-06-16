import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import time, os, re, json

KST = timezone(timedelta(hours=9))
CACHE_DIR = "/tmp/stock_picks_v2"

st.set_page_config(page_title="단타·종가 AI 추천", page_icon="📈", layout="centered")

for _key in ["ANTHROPIC_API_KEY"]:
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
.block-container { max-width: 900px !important; margin: 0 auto !important; padding: 4rem 1rem 1.5rem !important; }
[data-testid="stHeader"] { display: none !important; }
.stButton > button { background: #111 !important; color: #444 !important; border: 1px solid #1a1a1a !important; border-radius: 8px !important; }
div[data-testid="stSpinner"] > div { border-top-color: #ff6b6b !important; }
</style>
""",
    unsafe_allow_html=True,
)

_ETF_RE = re.compile(
    r"KODEX|TIGER|KBSTAR|ARIRANG|HANARO|\bACE\b|KOSEF|KINDEX|ETF|레버리지|인버스|선물|채권|국채|MSCI|\bSOL\b|\bPLUS\b|TIMEFOLIO",
    re.IGNORECASE,
)
_NAVER_HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def now_kst():
    return datetime.now(KST)


def save_picks_cache(date_key, slot_key, picks):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(f"{CACHE_DIR}/{date_key}_{slot_key}.json", "w", encoding="utf-8") as f:
            json.dump(picks, f, ensure_ascii=False)
    except Exception:
        pass


def load_picks_cache(date_key, slot_key):
    try:
        path = f"{CACHE_DIR}/{date_key}_{slot_key}.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


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


def _parse_naver_table(soup, market):
    rows = {}
    table = soup.find("table", {"class": "type_2"})
    if not table:
        return rows, 0
    count = 0
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue
        a = tds[1].find("a") if len(tds) > 1 else None
        if not a or "code=" not in a.get("href", ""):
            continue
        ticker = a["href"].split("code=")[1][:6]
        name = a.text.strip()
        if not ticker.isdigit() or _ETF_RE.search(name):
            continue
        try:
            def n(td):
                t = td.text.strip().replace(",", "").replace("+", "").replace("%", "")
                return t.replace("−", "-").replace("▲", "").replace("▼", "-").strip()

            price = int(n(tds[2])) if n(tds[2]) else 0
            chg = float(n(tds[4])) if n(tds[4]) else 0.0
            vol = int(n(tds[5])) if n(tds[5]) else 0
            val_s = n(tds[6])
            val = float(val_s) / 100 if val_s else 0.0

            if price < 1000 or price > 500000:
                continue
            if ticker not in rows:
                rows[ticker] = {
                    "종목코드": ticker,
                    "종목명": name,
                    "시장": market,
                    "현재가": price,
                    "등락률": round(chg, 2),
                    "거래량": vol,
                    "거래대금억": round(val, 1),
                }
            count += 1
        except:
            continue
    return rows, count


@st.cache_data(ttl=180)
def load_naver_candidates(date_key):
    all_rows = {}
    for sosok, market in [("0", "KOSPI"), ("1", "KOSDAQ")]:
        for url_type in ["sise_quant", "sise_rise"]:
            for page in range(1, 4):
                try:
                    url = f"https://finance.naver.com/sise/{url_type}.nhn?sosok={sosok}&page={page}"
                    resp = requests.get(url, headers=_NAVER_HDR, timeout=15)
                    resp.encoding = "euc-kr"
                    soup = BeautifulSoup(resp.text, "html.parser")
                    rows, count = _parse_naver_table(soup, market)
                    for k, v in rows.items():
                        if k not in all_rows:
                            all_rows[k] = v
                    if count == 0:
                        break
                except:
                    break
    return pd.DataFrame(list(all_rows.values())) if all_rows else pd.DataFrame()


# [수정 3·4] rsi14·rsi2 동시 반환 / days 45일로 단축
@st.cache_data(ttl=600)
def get_indicators(ticker, date_key):
    try:
        import FinanceDataReader as fdr

        now = now_kst()
        start = (now - timedelta(days=45)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        df = fdr.DataReader(ticker, start, end)
        if df is None or len(df) < 22:
            return 50.0, 50.0, False, False, 0.0, 0.0, 0.0

        c = df["Close"]
        has_today = df.index[-1].date() >= now.date()

        if has_today and len(df) >= 2:
            gap = (
                (float(df["Open"].iloc[-1]) - float(df["Close"].iloc[-2]))
                / float(df["Close"].iloc[-2])
                * 100
            )
            today_vol = float(df["Volume"].iloc[-1])
            avg_vol = float(df["Volume"].iloc[-21:-1].mean())
            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 0.0
            today_high = float(df["High"].iloc[-1])
            high_ratio = (float(c.iloc[-1]) / today_high * 100) if today_high > 0 else 0.0
        else:
            gap, vol_ratio, high_ratio = 0.0, 0.0, 0.0

        rsi14 = calc_rsi(c, 14)
        rsi2 = calc_rsi(c, 2)
        ma_ok = bool(c.rolling(5).mean().iloc[-1] > c.rolling(20).mean().iloc[-1])
        macd_ok = bool(
            (c.ewm(span=12).mean().iloc[-1] - c.ewm(span=26).mean().iloc[-1]) > 0
        )
        return rsi14, rsi2, ma_ok, macd_ok, round(gap, 2), round(vol_ratio, 1), round(high_ratio, 1)
    except:
        return 50.0, 50.0, False, False, 0.0, 0.0, 0.0


# [수정 1] 모델명 claude-sonnet-4-5
@st.cache_data(ttl=600)
def get_comment(name, vol, gap, chg, rsi, mode):
    try:
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return f"거래량 {vol}배 + 갭 {gap}% → {'눌림목 진입 검토' if mode == 'danta' else '다음날 갭상승 기대'}"
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=60,
            messages=[
                {
                    "role": "user",
                    "content": f"{name}: 거래량{vol}x 갭{gap}% 등락{chg}% RSI{rsi}. "
                    f"{'단타브레이크아웃' if mode == 'danta' else '오버나이트종가배팅'} 1줄분석 20자이내",
                }
            ],
        )
        return msg.content[0].text.strip()
    except:
        return f"{'단타' if mode == 'danta' else '종가배팅'} 주목"


def get_picks(df, date_key, mode):
    if df.empty:
        return []

    if mode == "danta":
        cands = df[
            (df["등락률"] > 0)
            & (df["현재가"] >= 1000)
            & (df["현재가"] <= 200000)
            & (df["거래대금억"] >= 50)
        ].copy()
        sort_col = "거래량"
    else:
        # [수정 2] 등락률 8% 미만: 급등주 갭다운 리스크 제외
        cands = df[
            (df["등락률"] > 0)
            & (df["등락률"] < 8)
            & (df["현재가"] >= 1000)
            & (df["거래대금억"] >= 200)
        ].copy()
        sort_col = "거래대금억"

    if cands.empty:
        return []

    # [수정 4] top30 → top15
    top15 = cands.nlargest(15, sort_col)
    results = []

    for _, row in top15.iterrows():
        # [수정 3] rsi2 추가 언패킹
        rsi, rsi2, ma_ok, macd_ok, gap, vol_ratio, high_ratio = get_indicators(
            row["종목코드"], date_key
        )

        if mode == "danta":
            if gap < 1.0 or gap > 10:
                continue
            if vol_ratio < 3.0:
                continue
        else:
            # [수정 2] 종가 >= 고가 * 0.95 + 거래량 1.5배 조건 추가
            if high_ratio < 95:
                continue
            if vol_ratio < 1.5:
                continue
            if not ma_ok:
                continue

        rs = rsi_score(rsi, mode)
        if rs == -9999:
            continue

        if mode == "danta":
            score = (
                vol_ratio * 0.35
                + gap * 0.25
                + row["등락률"] * 0.20
                + rs * 0.20
            )
            if macd_ok:
                score *= 1.1
            if ma_ok:
                score *= 1.05
            # [수정 3] RSI(2) < 10: 단기 눌림목 확인 시 보너스
            if rsi2 < 10:
                score += 15
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
                "갭": gap,
                "거래량배율": vol_ratio,
                "종가고가비율": high_ratio,
                "RSI": rsi,
                "MA정배열": ma_ok,
                "MACD양수": macd_ok,
                "점수": round(score, 2),
            }
        )

    results.sort(key=lambda x: x["점수"], reverse=True)
    return results[:3]


def mini_card(s, rank, mode):
    medals = ["🥇", "🥈", "🥉"]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    accent = "#ff6b6b" if mode == "danta" else "#4488ff"
    m = medals[rank] if rank < 3 else str(rank + 1)
    c = colors[rank] if rank < 3 else "#777"

    if mode == "jongga":
        metric1_val = f"{s['종가고가비율']}%"
        metric1_label = "종가/고가"
    else:
        gap_v = s["갭"]
        metric1_val = f"+{gap_v}%" if gap_v >= 0 else f"{gap_v}%"
        metric1_label = "갭"

    badges = []
    if s["MA정배열"]:
        badges.append("MA✅")
    if s["MACD양수"]:
        badges.append("MACD✅")

    comment = get_comment(s["종목명"], s["거래량배율"], s["갭"], s["등락률"], s["RSI"], mode)
    chg_sign = "+" if s["등락률"] >= 0 else ""

    st.markdown(
        f"""
<div style='background:#0c0c0c;border-radius:12px;padding:14px 12px;
     border-left:3px solid {c};border-top:1px solid #151515;'>
  <div style='font-size:0.85rem;margin-bottom:2px;'>{m}</div>
  <div style='font-size:0.95rem;font-weight:900;color:#fff;line-height:1.2;'>{s["종목명"]}</div>
  <div style='color:#555;font-size:0.62rem;margin-bottom:6px;'>{s["종목코드"]}·{s["시장"]}</div>
  <div style='font-size:1.35rem;font-weight:900;color:{accent};'>{chg_sign}{s["등락률"]}%</div>
  <div style='color:#666;font-size:0.68rem;margin-bottom:8px;'>₩{s["현재가"]:,}</div>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;
       padding:6px 0;border-top:1px solid #141414;margin-bottom:6px;'>
    <div style='text-align:center;'>
      <div style='color:{accent};font-weight:700;font-size:0.82rem;'>{s["거래량배율"]}x</div>
      <div style='color:#555;font-size:0.58rem;'>거래량</div>
    </div>
    <div style='text-align:center;'>
      <div style='color:{accent};font-weight:700;font-size:0.82rem;'>{metric1_val}</div>
      <div style='color:#555;font-size:0.58rem;'>{metric1_label}</div>
    </div>
    <div style='text-align:center;'>
      <div style='color:#ff9900;font-weight:700;font-size:0.82rem;'>{s["RSI"]}</div>
      <div style='color:#555;font-size:0.58rem;'>RSI</div>
    </div>
  </div>
  <div style='color:#666;font-size:0.6rem;margin-bottom:4px;'>{" · ".join(badges) if badges else "—"}</div>
  <div style='color:#555;font-size:0.68rem;font-style:italic;'>💡 {comment}</div>
</div>""",
        unsafe_allow_html=True,
    )


def no_pick_box(mode):
    color = "#ff6b6b" if mode == "danta" else "#4488ff"
    label = "단타" if mode == "danta" else "종가배팅"
    cond = (
        "갭1%↑ + 거래량3배↑ + RSI50~75 + RSI(2)보너스"
        if mode == "danta"
        else "거래대금200억↑ + 등락률<8% + RSI40~68 + MA정배열 + 종가/고가≥95% + 거래량1.5배↑"
    )
    st.markdown(
        f"""
<div style='background:#0a0808;border:1px solid #1a1010;border-radius:12px;
     padding:18px 20px;text-align:center;margin-bottom:1rem;'>
  <div style='font-size:1.3rem;margin-bottom:5px;'>🙅</div>
  <div style='color:{color};font-weight:700;'>오늘 {label} 없음 — 쉬는 날</div>
  <div style='color:#444;font-size:0.78rem;margin-top:5px;'>{cond} 동시 충족 종목 없음</div>
</div>""",
        unsafe_allow_html=True,
    )


def slot_header(time_label, color="#ff6b6b"):
    st.markdown(
        f"""
<div style='color:{color};font-weight:700;font-size:0.9rem;margin:14px 0 8px;
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
  <div style='color:#888;font-size:0.78rem;'>{label}</div>
  <div style='color:{color};font-weight:700;'>{m:02d}:{s:02d}</div>
</div>""",
        unsafe_allow_html=True,
    )


def compute_slot(df, date_key, slot_key, mode, spinner_label):
    """파일 캐시 → session_state 순으로 확인, 없으면 계산 후 파일에 저장."""
    if slot_key in st.session_state:
        return
    cached = load_picks_cache(date_key, slot_key)
    if cached is not None:
        st.session_state[slot_key] = cached
        return
    with st.spinner(spinner_label):
        picks = get_picks(df, date_key, mode)
    save_picks_cache(date_key, slot_key, picks)
    st.session_state[slot_key] = picks


def render_slot(session_key, mode, time_label):
    picks = st.session_state.get(session_key, [])
    slot_header(time_label, "#ff6b6b" if mode == "danta" else "#4488ff")
    if not picks:
        no_pick_box(mode)
    else:
        cols = st.columns(len(picks))
        for i, s in enumerate(picks):
            with cols[i]:
                mini_card(s, i, mode)


def countdown_box(rem_sec, label, color="#4CAF50"):
    h, r = divmod(rem_sec, 3600)
    m, s = divmod(r, 60)
    st.markdown(
        f"""
<div style='background:#0a0a0a;border:1px solid #151515;border-radius:14px;
     padding:40px 24px;text-align:center;margin:1rem 0;'>
  <div style='color:#666;font-size:0.78rem;letter-spacing:2px;margin-bottom:12px;'>{label}</div>
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

    # [수정 5] 헤더 전략명 변경
    st.markdown(
        f"""
<div style='margin-bottom:1.2rem;'>
  <div style='font-size:1.7rem;font-weight:900;color:#fff;letter-spacing:-1px;'>📈 단타·종가 AI 추천</div>
  <div style='color:#666;font-size:0.75rem;margin-top:3px;'>
    신고가돌파·거래량급증·오버나이트 전략 기반 · {now.strftime("%Y.%m.%d %H:%M:%S")} KST
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
     padding:14px 18px;color:#777;font-size:0.78rem;line-height:1.9;'>
  🔴 <b style='color:#888;'>단타</b> 9:10 · 9:20 · 9:30 총 3회 · 갭1%↑ + 거래량3배↑ + RSI50~75 + RSI(2)보너스<br>
  🌙 <b style='color:#888;'>종가배팅</b> 14:50 확정 · 거래대금200억↑ + 등락률&lt;8% + RSI40~68 + MA정배열 + 종가/고가≥95% + 거래량1.5배↑<br>
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
        m2, s2 = divmod(rem, 60)
        st.markdown(
            f"""
<div style='background:#0a1a0a;border:1px solid #1a3a1a;border-radius:12px;
     padding:18px 22px;margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center;'>
  <div>
    <div style='color:#4CAF50;font-weight:700;'>⚡ 장 시작 · 분석 중</div>
    <div style='color:#446644;font-size:0.78rem;margin-top:3px;'>9:10 · 9:20 · 9:30 단타 3회 추천 예정</div>
  </div>
  <div style='text-align:right;'>
    <div style='color:#4CAF50;font-size:1.8rem;font-weight:900;'>{m2:02d}:{s2:02d}</div>
    <div style='color:#446644;font-size:0.72rem;'>9:10 첫 추천</div>
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
<div style='color:#666;font-size:0.75rem;margin-bottom:8px;'>
  갭1%↑ · 거래량3배↑ · RSI50~75 · RSI(2)눌림목 · 각 시간 확정 후 고정
</div>""",
        unsafe_allow_html=True,
    )

    with st.spinner("네이버 데이터 로딩 중..."):
        df = load_naver_candidates(today)

    if df.empty:
        st.error("⚠️ 시장 데이터 로드 실패 — 네트워크 오류 또는 장 휴장일")
    else:
        compute_slot(df, today, "danta_910", "danta", "9:10 분석 중...")
        render_slot("danta_910", "danta", "9:10")

        if now >= t920:
            compute_slot(df, today, "danta_920", "danta", "9:20 분석 중...")
            render_slot("danta_920", "danta", "9:20")
        else:
            next_slot_badge(t920, "다음 단타 추천까지 (9:20)")

        if now >= t930:
            compute_slot(df, today, "danta_930", "danta", "9:30 분석 중...")
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
        m3, s3 = divmod(r2, 60)
        st.markdown(
            f"""
<div style='background:#08080f;border:1px solid #15152a;border-radius:12px;
     padding:16px 20px;display:flex;justify-content:space-between;align-items:center;'>
  <div>
    <div style='color:#4488ff;font-weight:700;'>🌙 종가배팅 대기 중</div>
    <div style='color:#334466;font-size:0.75rem;margin-top:3px;'>거래대금200억↑ · 등락률&lt;8% · RSI40~68 · MA정배열 · 종가/고가≥95% · 거래량1.5배↑</div>
  </div>
  <div style='text-align:right;'>
    <div style='color:#4488ff;font-size:1.7rem;font-weight:900;'>{h2:02d}:{m3:02d}:{s3:02d}</div>
    <div style='color:#334466;font-size:0.72rem;'>14:50 확정</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )
    elif now < t1530:
        st.markdown(
            """
<div style='color:#4488ff;font-weight:700;font-size:1rem;margin-bottom:4px;'>🌙 종가배팅 추천 · 14:50 확정</div>
<div style='color:#666;font-size:0.75rem;margin-bottom:12px;'>
  거래대금200억↑ · 등락률&lt;8% · RSI40~68 · MA정배열 · 종가/고가≥95% · 거래량1.5배↑ · 다음날 9시 매도
</div>""",
            unsafe_allow_html=True,
        )
        if not df.empty:
            compute_slot(df, today, "jongga_1450", "jongga", "종가배팅 분석 중...")
            render_slot("jongga_1450", "jongga", "14:50")
    else:
        st.markdown(
            """
<div style='background:#080808;border:1px solid #111;border-radius:10px;
     padding:14px;text-align:center;color:#666;font-size:0.82rem;'>
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
        f"<div style='color:#444;font-size:0.68rem;margin-top:0.5rem;text-align:center;'>손절 -2% · 단타 +1~3% · 종가배팅 다음날 시초 매도 · {now.strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )
    time.sleep(60)
    st.rerun()


if __name__ == "__main__":
    main()
