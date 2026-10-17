#!/usr/bin/env python3
"""
크립토 시황 모니터 v4 - 분석 인사이트 중심
급등/급락 나열 X → 시장 구조, 자금흐름, 섹터 로테이션, 액션 포인트
"""

import json
import time
import datetime
import urllib.request
import urllib.error
import sys
import os
import signal

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================
# 설정
# ============================================================

EXCLUDE = {
    "USDCUSDT", "TUSDUSDT", "BUSDUSDT", "FDUSDUSDT", "EURUSDT",
    "DAIUSDT", "USDPUSDT", "GBPUSDT", "AEURUSDT", "USTCUSDT",
    "USDTUSDT",
}
MIN_VOLUME_USD = 1_000_000
REPORT_INTERVAL = 1 * 60 * 60   # 정기 리포트: 1시간
CHECK_INTERVAL = 5 * 60          # 체크 간격: 5분

# 포지션 알림 기준
POS_ALERT_24H_PCT = 2.0
POS_SHORT_ALERT_PCT = 1.5

MY_POSITIONS = [
    {"coin": "FIGHT", "entry": None, "qty": None},  # LONG Cross 10x
]

POSITION_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "position_history.json")

def load_position_history():
    try:
        with open(POSITION_HISTORY_FILE) as f:
            return json.load(f)
    except:
        return {}

def save_position_history(history):
    try:
        with open(POSITION_HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except:
        pass

SLACK_WEBHOOK_URLS = []

# 환경변수에서 읽기
_env_url = os.environ.get("SLACK_WEBHOOK_URL", "")
if _env_url:
    SLACK_WEBHOOK_URLS.append(_env_url)

# .env 파일에서 읽기 (SLACK_WEBHOOK_URL, SLACK_WEBHOOK_URL_2, ...)
_env_path = os.path.expanduser("~/.openclaw/.env.slack-crypto")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SLACK_WEBHOOK_URL") and "=" in line:
                val = line.split("=", 1)[1]
                if val and val not in SLACK_WEBHOOK_URLS:
                    SLACK_WEBHOOK_URLS.append(val)

# 하위 호환용
SLACK_WEBHOOK_URL = SLACK_WEBHOOK_URLS[0] if SLACK_WEBHOOK_URLS else ""

# ============================================================
# 섹터 분류
# ============================================================

SECTORS = {
    "L1": ["BTC","ETH","SOL","AVAX","SUI","NEAR","APT","TON","ADA","DOT",
           "ATOM","ALGO","XLM","HBAR","FTM","SEI","TIA","INJ","MINA","ICP",
           "ETC","TRX","XTZ","EGLD","KAVA","ONE","CELO","ROSE","ZIL","IOTX","KAS","STX"],
    "L2/Infra": ["ARB","OP","MATIC","POL","MNT","METIS","IMX","STRK","ZK","MANTA","BLAST","MODE"],
    "AI": ["TAO","RENDER","FET","RNDR","AGIX","OCEAN","AKT","WLD","ARKM","PRIME",
           "AI","TURBO","NMR","CTXC","PHB","TNSR"],
    "DeFi": ["HYPE","UNI","AAVE","LINK","MKR","SKY","CRV","COMP","SNX","SUSHI",
             "DYDX","1INCH","BAL","YFI","CAKE","JUP","RAY","ORCA","MORPHO","ENA",
             "PENDLE","RSR","LQTY","RDNT","QNT","ZRX","LDO","RPL","SSV","EIGEN"],
    "Gaming/NFT": ["AXS","SAND","MANA","GALA","ENJ","ILV","PIXEL","RONIN","IMX",
                    "BLUR","LOOKS","MAGIC","YGG","BEAM","PORTAL","SUPER","NAKA","ALICE"],
    "Meme": ["DOGE","SHIB","PEPE","BONK","FLOKI","WIF","MEME","NEIRO","BOME",
             "DOGS","PEOPLE","LUNC","BABYDOGE","COQ","MYRO","BRETT","TRUMP","TURBO"],
    "Privacy": ["XMR","ZEC","DASH","ZEN","SCRT","ROSE"],
    "RWA": ["ONDO","POLYX","CFG","PROPS","RIO","CPOOL"],
}

def get_sector(coin):
    for s, coins in SECTORS.items():
        if coin in coins:
            return s
    return "기타"

# ============================================================
# API
# ============================================================

BINANCE = "https://api.binance.com"
GATEIO = "https://api.gateio.ws/api/v4"

# 바이낸스에 없는 코인 → Gate.io 폴백
GATEIO_COINS = {c["coin"] for c in MY_POSITIONS if c["coin"] not in []}  # 동적 체크

def api_get(endpoint, params=None):
    url = f"{BINANCE}{endpoint}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "CM/4"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception as e:
        print(f"[ERR] API: {e}")
        return None

def gateio_get(endpoint, params=None):
    url = f"{GATEIO}{endpoint}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "CM/4"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception as e:
        print(f"[ERR] Gate.io API: {e}")
        return None

def get_gateio_ticker(coin):
    """Gate.io에서 특정 코인 티커 가져오기"""
    pair = f"{coin}_USDT"
    data = gateio_get("/spot/tickers", {"currency_pair": pair})
    if not data or len(data) == 0:
        return None
    d = data[0]
    vol = float(d.get("quote_volume", 0))
    return {
        "symbol": f"{coin}USDT", "coin": coin,
        "price": float(d["last"]),
        "chg": float(d.get("change_percentage", 0)),
        "high": float(d.get("high_24h", d["last"])),
        "low": float(d.get("low_24h", d["last"])),
        "vol": vol, "trades": 0,
        "_source": "gateio",
    }

def get_gateio_klines(coin, interval="1h", limit=24):
    """Gate.io에서 캔들 데이터 가져오기 (Binance klines 형식으로 변환)"""
    pair = f"{coin}_USDT"
    # Gate.io interval: 1h, 4h, 1d 등
    interval_map = {"1h": "1h", "4h": "4h", "1d": "1d"}
    gi = interval_map.get(interval, interval)
    data = gateio_get("/spot/candlesticks", {"currency_pair": pair, "interval": gi, "limit": str(limit)})
    if not data:
        return None
    # Gate.io: [timestamp, volume, close, high, low, open, ...]
    # Binance: [open_time, open, high, low, close, volume, ...]
    converted = []
    for c in data:
        converted.append([
            int(c[0]) * 1000,  # timestamp → ms
            c[5],  # open
            c[3],  # high
            c[4],  # low
            c[2],  # close
            c[1],  # volume
        ])
    return converted

def get_tickers():
    data = api_get("/api/v3/ticker/24hr")
    if not data:
        return []
    out = []
    for d in data:
        s = d["symbol"]
        if not s.endswith("USDT") or s in EXCLUDE:
            continue
        v = float(d["quoteVolume"])
        if v < MIN_VOLUME_USD:
            continue
        out.append({
            "symbol": s, "coin": s.replace("USDT",""),
            "price": float(d["lastPrice"]),
            "chg": float(d["priceChangePercent"]),
            "high": float(d["highPrice"]),
            "low": float(d["lowPrice"]),
            "vol": v, "trades": int(d["count"]),
        })

    # Gate.io 폴백: 바이낸스에 없는 포지션 코인 추가
    binance_coins = {r["coin"] for r in out}
    for pos in MY_POSITIONS:
        coin = pos["coin"]
        if coin not in binance_coins:
            gt = get_gateio_ticker(coin)
            if gt:
                out.append(gt)
                print(f"  [Gate.io] {coin} 추가: {gt['price']}")

    return out

def get_klines(symbol, interval="1h", limit=24):
    # Gate.io 코인인지 체크
    coin = symbol.replace("USDT", "")
    pos_coins = {p["coin"] for p in MY_POSITIONS}
    if coin in pos_coins:
        # 먼저 Binance 시도, 실패하면 Gate.io
        result = api_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        if result:
            return result
        return get_gateio_klines(coin, interval, limit)
    return api_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})

def get_fng():
    try:
        req = urllib.request.Request("https://api.alternative.me/fng/?limit=1", headers={"User-Agent": "CM/4"})
        d = json.loads(urllib.request.urlopen(req, timeout=10).read())["data"][0]
        return int(d["value"]), d["value_classification"]
    except:
        return None, None

# ============================================================
# 기술 분석 유틸
# ============================================================

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al == 0:
        return 100
    return 100 - (100 / (1 + ag / al))

def fmt_price(p):
    if p >= 1000: return f"${p:,.0f}"
    elif p >= 1: return f"${p:,.2f}"
    elif p >= 0.01: return f"${p:,.4f}"
    else: return f"${p:,.6f}"

def pct_str(v):
    return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"

# ============================================================
# 분석 엔진
# ============================================================

def analyze_market_structure(results):
    """시장 구조 분석"""
    total = len(results)
    up = sum(1 for r in results if r["chg"] > 0)
    down = total - up
    up_ratio = up / total * 100 if total else 0
    avg = sum(r["chg"] for r in results) / total if total else 0
    median_chg = sorted(r["chg"] for r in results)[total // 2] if total else 0
    total_vol = sum(r["vol"] for r in results)

    # 대형/중형/소형 분류
    large = [r for r in results if r["vol"] >= 50_000_000]     # $50M+
    mid = [r for r in results if 10_000_000 <= r["vol"] < 50_000_000]
    small = [r for r in results if r["vol"] < 10_000_000]

    large_avg = sum(r["chg"] for r in large) / len(large) if large else 0
    mid_avg = sum(r["chg"] for r in mid) / len(mid) if mid else 0
    small_avg = sum(r["chg"] for r in small) / len(small) if small else 0

    # BTC 도미넌스 프록시: BTC 대비 알트 성과
    btc = next((r for r in results if r["coin"] == "BTC"), None)
    alt_avg = sum(r["chg"] for r in results if r["coin"] != "BTC") / max(len(results)-1, 1)

    return {
        "total": total, "up": up, "down": down, "up_ratio": up_ratio,
        "avg": avg, "median": median_chg, "total_vol": total_vol,
        "large_avg": large_avg, "mid_avg": mid_avg, "small_avg": small_avg,
        "large_n": len(large), "mid_n": len(mid), "small_n": len(small),
        "btc_chg": btc["chg"] if btc else 0,
        "alt_avg": alt_avg,
        "btc_vs_alt": (btc["chg"] if btc else 0) - alt_avg,
    }

def analyze_sector_rotation(results):
    """섹터 로테이션 분석"""
    sec_data = {}
    for r in results:
        s = get_sector(r["coin"])
        if s not in sec_data:
            sec_data[s] = {"coins": [], "vol": 0}
        sec_data[s]["coins"].append(r)
        sec_data[s]["vol"] += r["vol"]

    sectors = []
    for s, d in sec_data.items():
        chgs = [c["chg"] for c in d["coins"]]
        avg = sum(chgs) / len(chgs)
        variance = sum((c - avg) ** 2 for c in chgs) / len(chgs) if chgs else 0
        # 변동률 순 정렬
        sorted_coins = sorted(d["coins"], key=lambda x: x["chg"], reverse=True)
        sectors.append({
            "name": s, "avg": avg, "n": len(chgs),
            "vol": d["vol"], "spread": variance ** 0.5,
            "coins": sorted_coins,
        })
    sectors.sort(key=lambda x: x["avg"], reverse=True)
    return sectors

def analyze_money_flow(results):
    """자금 흐름 분석 - 가격 상승 + 거래량 동반 여부"""
    # 상승 + 거래량 높은 = 자금 유입
    # 하락 + 거래량 높은 = 자금 유출
    inflow_coins = []
    outflow_coins = []
    vol_median = sorted(r["vol"] for r in results)[len(results)//2] if results else 0

    for r in results:
        if r["vol"] < vol_median:
            continue  # 거래량 중앙값 이상만 분석
        if r["chg"] > 2 and r["vol"] >= vol_median * 1.5:
            inflow_coins.append(r)
        elif r["chg"] < -2 and r["vol"] >= vol_median * 1.5:
            outflow_coins.append(r)

    inflow_coins.sort(key=lambda x: x["vol"], reverse=True)
    outflow_coins.sort(key=lambda x: x["vol"], reverse=True)

    # 섹터별 자금 흐름
    sec_flow = {}
    for r in results:
        s = get_sector(r["coin"])
        if s not in sec_flow:
            sec_flow[s] = {"in_vol": 0, "out_vol": 0}
        if r["chg"] > 0:
            sec_flow[s]["in_vol"] += r["vol"]
        else:
            sec_flow[s]["out_vol"] += r["vol"]

    return {
        "inflow": inflow_coins[:5],
        "outflow": outflow_coins[:5],
        "sector_flow": sec_flow,
    }

def generate_insights(structure, sectors, money, fng):
    """규칙 기반 인사이트 생성"""
    insights = []

    # 시장 전체
    if structure["up_ratio"] >= 70:
        if structure["avg"] > 3:
            insights.append("시장 전반 강한 매수세. 대부분 코인이 상승 중이며 평균 변동폭도 크다.")
        else:
            insights.append("다수 코인이 소폭 상승 중. 강한 랠리는 아니지만 분위기는 긍정적.")
    elif structure["up_ratio"] <= 30:
        insights.append("시장 전반 매도 압력. 선택적 매수 필요. 반등 시도 확인 후 진입 권장.")
    else:
        insights.append("상승/하락 혼조세. 섹터별 차별화 장세.")

    # FNG vs 가격 괴리
    if fng is not None:
        if fng <= 20 and structure["avg"] > 1:
            insights.append(f"심리({fng})는 극도의 공포인데 가격은 반등 중 → 역발상 매수 기회 가능성. 역사적으로 이런 괴리 후 90일 내 중앙값 +38% 수익.")
        elif fng <= 20 and structure["avg"] < -2:
            insights.append(f"심리({fng})도 공포, 가격도 하락 → 아직 바닥 확인 안됨. 분할매수 또는 관망.")
        elif fng >= 75 and structure["avg"] < 0:
            insights.append(f"심리({fng})는 탐욕인데 가격 하락 → 고점 부근 조정 가능성.")

    # 대형 vs 소형
    gap = structure["small_avg"] - structure["large_avg"]
    if gap > 3:
        insights.append(f"소형주({pct_str(structure['small_avg'])}) >> 대형주({pct_str(structure['large_avg'])}). 위험선호 장세 — 알트 시즌 신호일 수 있으나 변동성 주의.")
    elif gap < -3:
        insights.append(f"대형주({pct_str(structure['large_avg'])}) >> 소형주({pct_str(structure['small_avg'])}). 자금이 안전자산으로 이동 중. 알트 진입 신중하게.")

    # BTC vs 알트
    if structure["btc_vs_alt"] > 2:
        insights.append("BTC가 알트 대비 강세 → BTC 도미넌스 상승 구간. 알트 약세 지속 가능.")
    elif structure["btc_vs_alt"] < -2:
        insights.append("알트가 BTC 대비 강세 → 자금이 알트로 분산. 알트 랠리 가능성.")

    # 섹터 로테이션
    named_sectors = [s for s in sectors if s["name"] != "기타"]
    if len(named_sectors) >= 2:
        top_sec = named_sectors[0]
        bot_sec = named_sectors[-1]
        if top_sec["avg"] - bot_sec["avg"] > 5:
            top_coins = ", ".join(f"{c['coin']}({pct_str(c['chg'])})" for c in top_sec["coins"][:3])
            insights.append(f"섹터 차별화 뚜렷: {top_sec['name']}({pct_str(top_sec['avg'])}) vs {bot_sec['name']}({pct_str(bot_sec['avg'])}). 로테이션 진행 중.")

    # 자금 유입 섹터
    flow = money["sector_flow"]
    net_flows = []
    for s, f in flow.items():
        if s == "기타":
            continue
        net = f["in_vol"] - f["out_vol"]
        net_flows.append((s, net, f["in_vol"] + f["out_vol"]))
    net_flows.sort(key=lambda x: x[1], reverse=True)
    if net_flows:
        top_inflow = net_flows[0]
        top_outflow = net_flows[-1]
        if top_inflow[1] > 0:
            insights.append(f"자금 유입 1위: {top_inflow[0]} (순유입 ${top_inflow[1]/1e6:,.0f}M)")
        if top_outflow[1] < 0:
            insights.append(f"자금 유출 1위: {top_outflow[0]} (순유출 ${abs(top_outflow[1])/1e6:,.0f}M)")

    return insights

def generate_action_points(structure, sectors, pos_infos, fng):
    """액션 포인트 생성"""
    actions = []

    # 포지션 관련
    for info in pos_infos:
        if not info:
            continue
        coin = info["coin"]
        rsi = info.get("rsi")
        rsi4 = info.get("rsi_4h")
        vs = info.get("vs_sector")
        trend = info.get("trend")

        if rsi is not None and rsi >= 75:
            actions.append(f"{coin}: RSI 과매수({rsi:.0f}) — 단기 익절 고려")
        elif rsi is not None and rsi <= 30:
            actions.append(f"{coin}: RSI 과매도({rsi:.0f}) — 추가 매수 검토 구간")
        if vs is not None and vs < -5:
            actions.append(f"{coin}: 같은 섹터 대비 크게 부진({pct_str(vs)}p) — 원인 확인 필요")
        if trend == "하락" and rsi4 is not None and rsi4 < 40:
            actions.append(f"{coin}: 단기 하락추세 + 4h RSI 약세 — 지지선 이탈 시 손절 기준 점검")
        elif trend == "상승" and rsi is not None and rsi < 65:
            actions.append(f"{coin}: 상승추세 + RSI 여유 — 홀딩 유지")

    # 시장 전체
    if fng is not None and fng <= 15:
        actions.append("시장: 극단적 공포 구간 — 장기 투자자에겐 역사적 매수 기회이나 단기 추가 하락 가능")
    if structure["up_ratio"] >= 80:
        actions.append("시장: 과열 경고 — 신규 진입보다 보유물량 관리에 집중")

    # 강한 섹터
    named_secs = [s for s in sectors if s["name"] != "기타"]
    if named_secs and named_secs[0]["avg"] > 5:
        s = named_secs[0]
        top3 = ", ".join(f"{c['coin']}({pct_str(c['chg'])})" for c in s["coins"][:3])
        actions.append(f"주목 섹터: {s['name']}({pct_str(s['avg'])}) — {top3}")

    return actions

# ============================================================
# 포지션 분석
# ============================================================

def analyze_position(pos, results, pos_history):
    coin = pos["coin"]
    sym = f"{coin}USDT"
    r = next((x for x in results if x["coin"] == coin), None)
    if not r:
        return None

    info = {
        "coin": coin, "price": r["price"], "chg": r["chg"],
        "high": r["high"], "low": r["low"], "vol": r["vol"],
        "range_pos": ((r["price"] - r["low"]) / (r["high"] - r["low"]) * 100) if r["high"] != r["low"] else 50,
    }

    if pos.get("entry"):
        info["entry"] = pos["entry"]
        info["pnl_pct"] = ((r["price"] - pos["entry"]) / pos["entry"]) * 100
        if pos.get("qty"):
            info["qty"] = pos["qty"]
            info["pnl_usd"] = (r["price"] - pos["entry"]) * pos["qty"]
            info["value"] = r["price"] * pos["qty"]

    # 1h RSI + 지지/저항
    kl = get_klines(sym, "1h", 24)
    if kl:
        closes = [float(k[4]) for k in kl]
        highs = [float(k[2]) for k in kl]
        lows = [float(k[3]) for k in kl]
        info["rsi"] = calc_rsi(closes)
        info["support"] = min(lows)
        info["resistance"] = max(highs)
        if len(closes) >= 6:
            r3 = sum(closes[-3:]) / 3
            e3 = sum(closes[-6:-3]) / 3
            info["trend"] = "상승" if r3 > e3 * 1.005 else ("하락" if r3 < e3 * 0.995 else "횡보")

    # 4h RSI
    kl4 = get_klines(sym, "4h", 24)
    if kl4:
        info["rsi_4h"] = calc_rsi([float(k[4]) for k in kl4])

    # 1d RSI
    kld = get_klines(sym, "1d", 20)
    if kld:
        info["rsi_1d"] = calc_rsi([float(k[4]) for k in kld])

    # 섹터 대비
    sec = get_sector(coin)
    same = [x for x in results if get_sector(x["coin"]) == sec and x["coin"] != coin]
    if same:
        sec_avg = sum(x["chg"] for x in same) / len(same)
        info["vs_sector"] = r["chg"] - sec_avg
        info["sector"] = sec
        info["sector_avg"] = sec_avg

    # BTC 상관
    btc = next((x for x in results if x["coin"] == "BTC"), None)
    if btc:
        info["vs_btc"] = r["chg"] - btc["chg"]

    # 히스토리
    if coin not in pos_history:
        pos_history[coin] = {"prices": []}
    pos_history[coin]["prices"].append({
        "t": datetime.datetime.now().strftime("%H:%M"), "p": r["price"]
    })
    pos_history[coin]["prices"] = pos_history[coin]["prices"][-288:]

    return info

# ============================================================
# 리포트 포맷
# ============================================================

def format_report(results, fng_val, fng_label, pos_infos, structure, sectors, money):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")
    L = []

    L.append(f"{'='*46}")
    L.append(f"  CRYPTO MARKET ANALYSIS")
    L.append(f"  {now}")
    L.append(f"{'='*46}")
    L.append("")

    # ── 내 포지션 ──
    L.append("[MY POSITION]")
    for info in pos_infos:
        if not info:
            continue
        c = info["coin"]
        L.append(f"  {c}  {fmt_price(info['price'])}  {pct_str(info['chg'])}")

        # 범위 바
        rp = int(info["range_pos"] // 5)
        bar = f"  24h [{fmt_price(info['low'])}] {'='*rp}|{'.'*(20-rp)} [{fmt_price(info['high'])}]"
        L.append(bar)

        # 기술 지표
        parts = []
        for label, key in [("1h", "rsi"), ("4h", "rsi_4h"), ("1d", "rsi_1d")]:
            v = info.get(key)
            if v is not None:
                tag = ""
                if v >= 70: tag = "!OB"
                elif v <= 30: tag = "!OS"
                parts.append(f"RSI-{label}:{v:.0f}{tag}")
        if info.get("trend"):
            parts.append(f"추세:{info['trend']}")
        if parts:
            L.append(f"  {' | '.join(parts)}")

        # 레벨
        lvl = []
        if info.get("support"): lvl.append(f"지지 {fmt_price(info['support'])}")
        if info.get("resistance"): lvl.append(f"저항 {fmt_price(info['resistance'])}")
        if lvl:
            L.append(f"  {' | '.join(lvl)}")

        # 상대 성과
        rel = []
        if info.get("vs_btc") is not None:
            rel.append(f"vs BTC {pct_str(info['vs_btc'])}p")
        if info.get("vs_sector") is not None:
            rel.append(f"vs {info.get('sector','섹터')} {pct_str(info['vs_sector'])}p")
        if rel:
            L.append(f"  {' | '.join(rel)}")

        # 손익
        if "pnl_pct" in info:
            L.append(f"  매수가 {fmt_price(info['entry'])} | PnL {pct_str(info['pnl_pct'])}" +
                      (f" (${info['pnl_usd']:+,.0f})" if "pnl_usd" in info else ""))

    L.append("")

    # ── 시장 상태 ──
    L.append("[MARKET STATE]")
    fng_str = f"{fng_val}/100" if fng_val else "N/A"
    if fng_val:
        if fng_val <= 15: mood = "극단적 공포"
        elif fng_val <= 25: mood = "공포"
        elif fng_val <= 45: mood = "불안"
        elif fng_val <= 55: mood = "중립"
        elif fng_val <= 75: mood = "탐욕"
        else: mood = "극단적 탐욕"
    else:
        mood = ""
    L.append(f"  심리: {fng_str} {mood}")
    L.append(f"  방향: 상승 {structure['up']}/{structure['total']} ({structure['up_ratio']:.0f}%)")
    L.append(f"  변동: 평균 {pct_str(structure['avg'])} | 중앙값 {pct_str(structure['median'])}")
    L.append(f"  거래량: ${structure['total_vol']/1e9:,.1f}B")
    L.append("")

    # ── 시장 구조 ──
    L.append("[MARKET STRUCTURE]")
    L.append(f"  BTC {pct_str(structure['btc_chg'])} vs 알트 평균 {pct_str(structure['alt_avg'])}")
    if structure["btc_vs_alt"] > 1:
        L.append(f"  → BTC 우위 ({pct_str(structure['btc_vs_alt'])}p) : BTC 도미넌스 상승")
    elif structure["btc_vs_alt"] < -1:
        L.append(f"  → 알트 우위 ({pct_str(-structure['btc_vs_alt'])}p) : 자금 분산")
    else:
        L.append(f"  → 동조 움직임")
    L.append(f"  대형({structure['large_n']}개) {pct_str(structure['large_avg'])} | 중형({structure['mid_n']}개) {pct_str(structure['mid_avg'])} | 소형({structure['small_n']}개) {pct_str(structure['small_avg'])}")
    L.append("")

    # ── 섹터 로테이션 ──
    L.append("[SECTOR ROTATION]")
    named_secs = [s for s in sectors if s["name"] != "기타"]
    for i, s in enumerate(named_secs):
        bar = ">" * min(int(abs(s["avg"])), 6) if s["avg"] > 0 else "<" * min(int(abs(s["avg"])), 6)
        if not bar:
            bar = "-"
        L.append(f"  {s['name']:<12} {pct_str(s['avg']):>7} {bar}  ({s['n']}개)")
        # 상위 3개 섹터는 코인 전부, 나머지는 TOP 3만
        show_coins = s["coins"] if i < 3 else s["coins"][:3]
        for c in show_coins:
            L.append(f"    {c['coin']:<8} {fmt_price(c['price']):>10}  {pct_str(c['chg']):>7}  vol ${c['vol']/1e6:,.0f}M")
        if i >= 3 and len(s["coins"]) > 3:
            rest = len(s["coins"]) - 3
            L.append(f"    ...외 {rest}개")
        L.append("")
    L.append("")

    # ── 자금 흐름 ──
    L.append("[MONEY FLOW]")
    if money["inflow"]:
        coins_str = ", ".join(f"{r['coin']}({pct_str(r['chg'])})" for r in money["inflow"][:4])
        L.append(f"  유입: {coins_str}")
    if money["outflow"]:
        coins_str = ", ".join(f"{r['coin']}({pct_str(r['chg'])})" for r in money["outflow"][:4])
        L.append(f"  유출: {coins_str}")

    # 섹터 순유입
    net_flows = []
    for s, f in money["sector_flow"].items():
        if s == "기타":
            continue
        net = f["in_vol"] - f["out_vol"]
        net_flows.append((s, net))
    net_flows.sort(key=lambda x: x[1], reverse=True)
    if net_flows:
        top3 = net_flows[:3]
        L.append(f"  섹터 순유입: " + " > ".join(f"{s}(${n/1e6:+,.0f}M)" for s, n in top3))
    L.append("")

    # ── 인사이트 ──
    insights = generate_insights(structure, sectors, money, fng_val)
    L.append("[INSIGHT]")
    for i, ins in enumerate(insights, 1):
        L.append(f"  {i}. {ins}")
    L.append("")

    # ── 액션 포인트 ──
    actions = generate_action_points(structure, sectors, pos_infos, fng_val)
    if actions:
        L.append("[ACTION]")
        for a in actions:
            L.append(f"  > {a}")
        L.append("")

    L.append(f"{'='*46}")
    return "\n".join(L)

# ============================================================
# 알림 (5분 체크)
# ============================================================

def check_alerts(results, prev_prices):
    pos_coins = {p["coin"] for p in MY_POSITIONS}
    pos_lines = []
    other_lines = []

    for r in results:
        coin = r["coin"]
        sym = r["symbol"]

        if coin in pos_coins:
            # 보유 코인: 매 5분마다 항상 가격 알림
            prev = prev_prices.get(sym)
            if prev and prev > 0:
                sp = ((r["price"] - prev) / prev) * 100
                sp_str = f" | 5min {pct_str(sp)}" if abs(sp) >= 0.1 else ""
            else:
                sp_str = ""
            rng = ((r["price"] - r["low"]) / (r["high"] - r["low"]) * 100) if r["high"] != r["low"] else 50
            bar_pos = int(rng // 5)
            bar = f"{'='*bar_pos}|{'.'*(20-bar_pos)}"
            pos_lines.append(f"  {coin} {fmt_price(r['price'])} ({pct_str(r['chg'])}){sp_str}")
            pos_lines.append(f"  [{fmt_price(r['low'])}]{bar}[{fmt_price(r['high'])}]")

    if pos_lines:
        now = datetime.datetime.now().strftime("%H:%M")
        return [f"[내 포지션 {now}]"] + pos_lines
    return []

# ============================================================
# Slack
# ============================================================

def send_slack(msg):
    return  # Retired: replaced by momentum-alert; historical analysis remains available.
    if not SLACK_WEBHOOK_URLS:
        print(f"[NO SLACK]\n{msg}\n")
        return
    # Slack Webhook은 ~4000bytes 정도가 안전. 넘으면 분할 전송.
    MAX_LEN = 3500
    chunks = []
    if len(msg) <= MAX_LEN:
        chunks = [msg]
    else:
        lines = msg.split("\n")
        chunk = []
        chunk_len = 0
        for line in lines:
            if chunk_len + len(line) + 1 > MAX_LEN and chunk:
                chunks.append("\n".join(chunk))
                chunk = [line]
                chunk_len = len(line)
            else:
                chunk.append(line)
                chunk_len += len(line) + 1
        if chunk:
            chunks.append("\n".join(chunk))

    for i, c in enumerate(chunks):
        payload = json.dumps({"text": f"```\n{c}\n```"}).encode("utf-8")
        for wh_url in SLACK_WEBHOOK_URLS:
            req = urllib.request.Request(wh_url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                urllib.request.urlopen(req, timeout=10)
            except Exception as e:
                print(f"[ERR] Slack ({wh_url[:50]}...): {e}")
        if i == len(chunks) - 1:
            print(f"[OK] Slack ({datetime.datetime.now().strftime('%H:%M')}) [{len(chunks)}파트] → {len(SLACK_WEBHOOK_URLS)}곳")
        if i < len(chunks) - 1:
            time.sleep(1)  # rate limit 방지

# ============================================================
# Main
# ============================================================

def do_report(results, pos_history):
    fng_val, fng_label = get_fng()
    pos_infos = [analyze_position(p, results, pos_history) for p in MY_POSITIONS]
    save_position_history(pos_history)
    structure = analyze_market_structure(results)
    sectors = analyze_sector_rotation(results)
    money = analyze_money_flow(results)
    report = format_report(results, fng_val, fng_label, pos_infos, structure, sectors, money)
    send_slack(report)
    return time.time()

def main():
    print("=" * 50)
    print("  크립토 시황 모니터 v4 (분석 중심)")
    print(f"  포지션: {', '.join(p['coin'] for p in MY_POSITIONS)}")
    print(f"  리포트: {REPORT_INTERVAL//3600}h | 체크: {CHECK_INTERVAL//60}min")
    print("=" * 50)

    prev_prices = {}
    pos_history = load_position_history()
    running = True

    def stop(sig, frame):
        nonlocal running
        print("\n[STOP]")
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    results = get_tickers()
    if results:
        last_report = do_report(results, pos_history)
        for r in results:
            prev_prices[r["symbol"]] = r["price"]
        print(f"  스캔: {len(results)}개")
    else:
        last_report = 0

    while running:
        time.sleep(CHECK_INTERVAL)
        if not running:
            break
        try:
            results = get_tickers()
            if not results:
                continue

            alerts = check_alerts(results, prev_prices)
            if alerts:
                now = datetime.datetime.now().strftime("%H:%M")
                send_slack(f"--- ALERT {now} ---\n" + "\n".join(alerts))

            for r in results:
                prev_prices[r["symbol"]] = r["price"]

            if time.time() - last_report >= REPORT_INTERVAL:
                last_report = do_report(results, pos_history)
        except Exception as e:
            print(f"[ERR] {e}")

    save_position_history(pos_history)
    print("[EXIT]")

if __name__ == "__main__":
    raise SystemExit("Legacy notifier retired. Use node momentum-alert/scan.mjs; see momentum-alert/USAGE.txt")
    if "--once" in sys.argv:
        results = get_tickers()
        if results:
            pos_history = load_position_history()
            fng_val, fng_label = get_fng()
            pos_infos = [analyze_position(p, results, pos_history) for p in MY_POSITIONS]
            save_position_history(pos_history)
            structure = analyze_market_structure(results)
            sectors = analyze_sector_rotation(results)
            money = analyze_money_flow(results)
            print(format_report(results, fng_val, fng_label, pos_infos, structure, sectors, money))
    else:
        main()
