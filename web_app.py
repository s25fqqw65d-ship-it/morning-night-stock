import streamlit as st
from FinMind.data import DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
from bs4 import BeautifulSoup

# 載入機器學習套件
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    st.markdown("**系統提示：** 未安裝 scikit-learn，請更新 requirements.txt。")

# ==========================================
# 網頁配置與大盤濾網
# ==========================================
st.set_page_config(page_title="早安晚上好", layout="wide")
st.title("早安晚上好 - 終極量化終端機")

@st.cache_data(ttl=3600)
def get_market_status():
    try:
        twii = yf.Ticker("^TWII").history(period="3mo")
        twii['MA20'] = twii['Close'].rolling(20).mean()
        last_close = twii['Close'].iloc[-1]
        last_ma20 = twii['MA20'].iloc[-1]
        if last_close > last_ma20:
            return "多頭格局 (站上月線)", 1.2
        else:
            return "空頭格局 (跌破月線)", 0.7
    except: return "震盪不明", 1.0

# 側邊欄與資料讀取
st.sidebar.header("系統設定")
finmind_token = st.sidebar.text_input("輸入 FinMind Token", type="password", help="留空即為訪客模式")

@st.cache_data(ttl=86400) 
def load_taiwan_stocks():
    dl_temp = DataLoader()
    info = dl_temp.taiwan_stock_info()
    return info[info['stock_id'].str.len() == 4].dropna(subset=['industry_category'])

tw_stocks_df = load_taiwan_stocks()
industry_list = sorted(tw_stocks_df['industry_category'].unique().tolist())
market_status_text, market_multiplier = get_market_status()

# ==========================================
# 核心大腦 1.5：新聞輿情分析爬蟲
# ==========================================
def get_news_sentiment(stock_code):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_code}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        res = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(res.text, 'html.parser')

        headlines = []
        for a in soup.find_all('a'):
            text = a.get_text().strip()
            if len(text) > 10 and text not in headlines: 
                headlines.append(text)
        
        headlines = headlines[:8] 

        bullish_kw = ['創高', '成長', '受惠', '看好', '大增', '突破', '買超', '利多', '漲停', '報喜', '優於預期', '接單', '強攻']
        bearish_kw = ['衰退', '大跌', '降評', '砍單', '下修', '看壞', '賣超', '利空', '跌破', '探底', '保守', '出脫', '重挫']

        bull_score = sum(1 for t in headlines for k in bullish_kw if k in t)
        bear_score = sum(1 for t in headlines for k in bearish_kw if k in t)

        if bull_score > bear_score: return "利多發酵", 10, headlines[:3]
        elif bear_score > bull_score: return "賣壓利空", -10, headlines[:3]
        else: return "情緒中性", 0, headlines[:3]
    except:
        return "暫無新聞", 0, []

# ==========================================
# 核心大腦 2.0：分析模組
# ==========================================
def analyze_stock(stock_code, stock_name, dl, is_single_mode=False, price_filter="不限"):
    try:
        pe = yf.Ticker(f"{stock_code}.TW").info.get('trailingPE', 999)
        start_dt = '2022-01-01' if is_single_mode else '2023-06-01'
        df_price = dl.taiwan_stock_daily(stock_id=stock_code, start_date=start_dt) 
        df_chip = dl.taiwan_stock_institutional_investors(stock_id=stock_code, start_date='2024-02-01')
        
        if df_price.empty or df_chip.empty or len(df_price) < 60: return None

        last_close = df_price.iloc[-1]['close']
        if not is_single_mode:
            if price_filter == "100元以下" and last_close >= 100: return None
            elif price_filter == "100~500元" and (last_close < 100 or last_close >= 500): return None
            elif price_filter == "500元以上" and last_close < 500: return None

        df_price['MA5'] = df_price['close'].rolling(5).mean()
        df_price['MA20'] = df_price['close'].rolling(20).mean()
        df_price['MA60'] = df_price['close'].rolling(60).mean()
        df_price['Volume'] = df_price['Trading_Volume'] / 1000 
        df_price['Vol_MA5'] = df_price['Volume'].rolling(5).mean()
        df_price['Turnover'] = df_price['close'] * df_price['Volume']
        df_price['VWAP_20'] = df_price['Turnover'].rolling(20).sum() / df_price['Volume'].rolling(20).sum()
        
        delta = df_price['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df_price['RSI'] = 100 - (100 / (1 + (gain / loss)))

        today, prev = df_price.iloc[-1], df_price.iloc[-2]
        foreign_net = int((df_chip[df_chip['name']=='Foreign_Investor'].iloc[-3:]['buy'].sum() - df_chip[df_chip['name']=='Foreign_Investor'].iloc[-3:]['sell'].sum()) / 1000)
        trust_net = int((df_chip[df_chip['name']=='Investment_Trust'].iloc[-3:]['buy'].sum() - df_chip[df_chip['name']=='Investment_Trust'].iloc[-3:]['sell'].sum()) / 1000)

        # 新聞輿情：只有在「單股模式」才啟動，避免海選卡死！
        news_status, news_score, latest_headlines = "未掃描", 0, []
        rev_status, rev_score = "無資料", 0
        
        if is_single_mode:
            news_status, news_score, latest_headlines = get_news_sentiment(stock_code)
            try:
                df_rev = dl.taiwan_stock_month_revenue(stock_id=stock_code, start_date='2023-01-01')
                yoy = ((df_rev.iloc[-1]['revenue'] - df_rev.iloc[-13]['revenue']) / df_rev.iloc[-13]['revenue']) * 100
                rev_score = 20 if yoy > 20 else (10 if yoy > 0 else -15)
                rev_status = f"{'成長' if yoy>0 else '衰退'} ({yoy:.1f}%)"
            except: pass

        cdp = (today['max'] + today['min'] + 2 * today['close']) / 4
        
        # 修復 TR 計算方式，確保動態停利線精準
        df_price['H-L'] = df_price['max'] - df_price['min']
        df_price['H-PC'] = abs(df_price['max'] - df_price['close'].shift(1))
        df_price['L-PC'] = abs(df_price['min'] - df_price['close'].shift(1))
        df_price['TR'] = df_price[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr_14 = df_price['TR'].rolling(14).mean().iloc[-1]
        swing_target = today['close'] + (atr_14 * 2.5)
        trailing_stop = df_price['max'].rolling(20).max().iloc[-1] - (atr_14 * 2.5)

        # AI 引擎
        ai_prob = "50.0%"
        if is_single_mode and len(df_price) > 200:
            try:
                ml_df = df_price[['close', 'Volume', 'MA20', 'RSI']].copy().dropna()
                ml_df['Target'] = (ml_df['close'].shift(-1) > ml_df['close']).astype(int)
                train = ml_df.dropna()
                clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                clf.fit(train[['close', 'Volume', 'MA20', 'RSI']], train['Target'])
                ai_prob = f"{clf.predict_proba(ml_df[['close', 'Volume', 'MA20', 'RSI']].iloc[-1:])[0][1]*100:.1f}%"
            except: ai_prob = "運算失敗"

        # 評分與判定
        score = int((50 + (15 if today['MA5']>today['MA20'] else -15) + (25 if today['Volume']>prev['Vol_MA5']*1.5 else 0) + rev_score + news_score + (15 if today['close']>today['VWAP_20'] else 0)) * market_multiplier)
        action = "爆發前夕" if score >= 100 else "強勢佈局" if score >= 80 else "偏多觀察" if score >= 60 else "嚴格避開"

        return {
            "代號": stock_code, "名稱": stock_name, "收盤價": round(today['close'], 1),
            "VWAP大戶成本": round(today['VWAP_20'], 1), "明日壓力": round(2*cdp-today['min'], 1),
            "明日支撐": round(2*cdp-today['max'], 1), "波段目標": round(swing_target, 1),
            "動態停利": round(trailing_stop, 1), "防守價": round(today['MA20'], 1),
            "綜合分數": score, "AI勝率": ai_prob, "營收動能": rev_status,
            "外資": foreign_net, "投信": trust_net, "判定": action,
            "新聞情緒": news_status, "最新標題": latest_headlines,
            "新聞": f"https://tw.stock.yahoo.com/quote/{stock_code}/news", "歷史資料": df_price 
        }
    except: return None

# ==========================================
# 介面顯示邏輯
# ==========================================
dl = DataLoader()
if finmind_token: dl.login_by_token(api_token=finmind_token)

tab1, tab2 = st.tabs(["單股解析", "產業海選"])

with tab1:
    st.markdown(f"**大盤環境指示：** {market_status_text}")
    c_in, c_cap, c_btn = st.columns([2, 1, 1])
    target_code = c_in.text_input("股票代號", "2330")
    enable_cap = c_cap.checkbox("計算資金部位")
    btn_single = c_btn.button("啟動深度解析", use_container_width=True)
    my_capital = st.number_input("總資金 (萬)", value=50) if enable_cap else 0

    if btn_single:
        name_match = tw_stocks_df[tw_stocks_df['stock_id'] == target_code.strip()]['stock_name']
        if name_match.empty: st.markdown("**錯誤：** 查無此代號")
        else:
            with st.spinner("AI 運算與網路輿情掃描中..."):
                r = analyze_stock(target_code.strip(), name_match.values[0], dl, is_single_mode=True)
            if r:
                st.markdown("---")
                try: ai_val = float(r['AI勝率'].replace('%',''))
                except: ai_val = 50.0

                if r['綜合分數'] >= 80 and ai_val > 55: diag = f"多頭強勢：AI 看漲機率高 ({r['AI勝率']})。支撐位 ${r['明日支撐']} 可注意，跌破 ${r['動態停利']} 離場。"
                elif ai_val < 40: diag = f"風險警告：AI 判定短線過熱，勝率僅 {r['AI勝率']}。建議不追高，等回測 ${r['明日支撐']}。"
                elif r['收盤價'] < r['VWAP大戶成本']: diag = f"弱勢整理：股價低於大戶成本 ${r['VWAP大戶成本']}，上方賣壓重，暫避開。"
                else: diag = f"震盪格局：目前動能中性。防守位為 ${r['防守價']}。"
                st.markdown(f"> **系統綜合診斷：** {diag}")

                c_t, c_a = st.columns([3, 1])
                c_t.markdown(f"### {r['代號']} {r['名稱']} 戰情報告")
                c_a.metric("AI 預測勝率", r['AI勝率'], help="隨機森林模型根據價量特徵預測下一日收紅機率")

                st.markdown(f"**明日 CDP 區間：** 壓力 ${r['明日壓力']} / 支撐 ${r['明日支撐']}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("收盤價", f"${r['收盤價']}")
                m2.metric("動態停利線", f"${r['動態停利']}", help="跌破代表趨勢改變")
                m3.metric("市場輿情", r['新聞情緒'], help="系統自動抓取最新新聞進行 NLP 情緒判定")
                m4.metric("營收 YoY", r['營收動能'])

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("系統評分", f"{r['綜合分數']} ({r['判定']})")
                m6.metric("月線防守", f"${r['防守價']}")
                m7.metric("大戶成本", f"${r['VWAP大戶成本']}", help="成交量加權平均價格")
                m8.metric("法人動向", "偏多" if r['外資']>0 else "偏空", f"外:{r['外資']} / 投:{r['投信']}")

                if r['最新標題']:
                    st.markdown("#### 📰 最新輿情焦點")
                    for title in r['最新標題']:
                        st.markdown(f"- {title}")

                st.markdown("#### 價格走勢與關鍵防守線")
                chart_data = r['歷史資料'].tail(60).set_index('date')
                st.line_chart(pd.DataFrame({"股價":chart_data['close'], "月線":chart_data['MA20'], "大戶成本":chart_data['VWAP_20']}), color=["#FFFFFF", "#FF4B4B", "#00D4FF"])

                if enable_cap:
                    risk = (my_capital * 10000) * 0.02
                    diff = r['收盤價'] - r['動態停利']
                    if diff > 0: st.markdown(f"**操作建議：** 建議買進 **{int(risk/(diff*1000))}** 張，風險控制於 2% 內。")
                
                st.link_button("進入 Yahoo 新聞中心看全文", r['新聞'], use_container_width=True)

with tab2:
    st.markdown("### 特定產業快速掃描")
    s_ind = st.selectbox("產業板塊", industry_list)
    s_p = st.selectbox("價格區間", ["不限", "100元以下", "100~500元", "500元以上"])
    if st.button("啟動掃描", use_container_width=True):
        res = []
        targets = tw_stocks_df[tw_stocks_df['industry_category'] == s_ind]
        bar = st.progress(0)
        for i, (idx, row) in enumerate(targets.iterrows()):
            bar.progress((i+1)/len(targets))
            out = analyze_stock(row['stock_id'], row['stock_name'], dl, price_filter=s_p)
            if out: res.append({"代號":out['代號'], "名稱":out['名稱'], "收盤價":out['收盤價'], "評分":out['綜合分數'], "判定":out['判定']})
            time.sleep(0.05)
        if res: st.dataframe(pd.DataFrame(res).sort_values("評分", ascending=False), use_container_width=True, hide_index=True)
