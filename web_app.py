import streamlit as st
from FinMind.data import DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
from bs4 import BeautifulSoup

# 載入進階套件
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    st.markdown("**系統提示：** 未安裝 scikit-learn，請更新 requirements.txt。")

try:
    import plotly.graph_objects as go
except ImportError:
    st.markdown("**系統提示：** 未安裝 plotly，無法顯示專業 K 線圖。請更新 requirements.txt。")

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
st.sidebar.markdown("若頻繁查詢導致沒畫面，請至 [FinMind 官網](https://finmindtrade.com/) 免費註冊獲取 Token。")
finmind_token = st.sidebar.text_input("輸入 FinMind Token", type="password", help="留空即為訪客模式 (每小時300次額度)")

@st.cache_data(ttl=86400) 
def load_taiwan_stocks():
    dl_temp = DataLoader()
    info = dl_temp.taiwan_stock_info()
    return info[info['stock_id'].str.len() == 4].dropna(subset=['industry_category'])

tw_stocks_df = load_taiwan_stocks()
industry_list = sorted(tw_stocks_df['industry_category'].unique().tolist())
market_status_text, market_multiplier = get_market_status()

# ==========================================
# 核心大腦 1.0：歷史回測與資金曲線引擎
# ==========================================
def run_backtest(df_price):
    position = 0
    buy_price = 0
    dates = []
    returns = []
    
    for i in range(20, len(df_price)):
        today = df_price.iloc[i]
        prev = df_price.iloc[i-1]
        
        is_bullish = today['MA5'] > today['MA20'] and today['MA20'] > today['MA60']
        vol_fire = today['Volume'] > (prev['Vol_MA5'] * 1.5)
        is_red = today['close'] > today['open']
        
        if position == 0 and is_bullish and vol_fire and is_red:
            position = 1
            buy_price = today['close']
        elif position == 1 and today['close'] < today['MA20']:
            position = 0
            sell_price = today['close']
            profit_pct = (sell_price - buy_price) / buy_price
            dates.append(today['date'])
            returns.append(profit_pct)
            
    if not returns: return {"交易次數": 0, "勝率": "0%", "總報酬": "0%", "曲線": pd.DataFrame()}
    
    win_trades = [t for t in returns if t > 0]
    win_rate = (len(win_trades) / len(returns)) * 100
    total_return = (np.prod([1 + t for t in returns]) - 1) * 100
    
    eq_curve = [1.0]
    for r in returns: eq_curve.append(eq_curve[-1] * (1 + r))
    df_curve = pd.DataFrame({"累積資金率": eq_curve[1:]}, index=dates)
    
    return {"交易次數": len(returns), "勝率": f"{win_rate:.1f}%", "總報酬": f"{total_return:.1f}%", "曲線": df_curve}

# ==========================================
# 核心大腦 1.5：新聞輿情分析爬蟲
# ==========================================
def get_news_sentiment(stock_code):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_code}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=2)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        headlines = []
        for a in soup.find_all('a'):
            text = a.get_text().strip()
            if len(text) > 10 and text not in headlines: 
                headlines.append(text)
        
        headlines = headlines[:8] 
        bullish_kw = ['創高', '成長', '受惠', '看好', '大增', '突破', '買超', '利多', '漲停', '報喜', '接單', '強攻', '反彈']
        bearish_kw = ['衰退', '大跌', '降評', '砍單', '下修', '看壞', '賣超', '利空', '跌破', '探底', '保守', '重挫', '出脫']
        bull_score = sum(1 for t in headlines for k in bullish_kw if k in t)
        bear_score = sum(1 for t in headlines for k in bearish_kw if k in t)

        if bull_score > bear_score: return "利多發酵", 10, headlines[:3]
        elif bear_score > bull_score: return "賣壓利空", -10, headlines[:3]
        else: return "情緒中性", 0, headlines[:3]
    except: return "暫無新聞", 0, []

# ==========================================
# 核心大腦 2.0：分析模組
# ==========================================
def analyze_stock(stock_code, stock_name, dl, is_single_mode=False, price_filter="不限"):
    try:
        pe = 999
        try: pe = yf.Ticker(f"{stock_code}.TW").info.get('trailingPE', 999)
        except: pass

        start_dt = '2022-01-01' if is_single_mode else '2023-06-01'
        df_price = dl.taiwan_stock_daily(stock_id=stock_code, start_date=start_dt) 
        df_chip = dl.taiwan_stock_institutional_investors(stock_id=stock_code, start_date='2024-01-01')
        
        if is_single_mode:
            if df_price.empty:
                st.error(f"🕵️‍♂️ 抓蟲報告：FinMind 伺服器拒絕提供【{stock_code} 歷史股價】。可能原因：免費額度已滿或無資料。")
                return None
            if df_chip.empty:
                st.error(f"🕵️‍♂️ 抓蟲報告：FinMind 伺服器拒絕提供【{stock_code} 三大法人籌碼】。")
                return None
            if len(df_price) < 60:
                st.error(f"🕵️‍♂️ 抓蟲報告：【{stock_code}】歷史資料不足 60 天。")
                return None
        else:
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
        
        df_price['STD20'] = df_price['close'].rolling(20).std()
        df_price['BB_Upper'] = df_price['MA20'] + (2 * df_price['STD20'])

        delta = df_price['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df_price['RSI'] = 100 - (100 / (1 + (gain / loss)))

        today, prev = df_price.iloc[-1], df_price.iloc[-2]

        df_chip['net'] = (df_chip['buy'] - df_chip['sell']) / 1000
        fi_data = df_chip[df_chip['name']=='Foreign_Investor'].groupby('date')['net'].sum()
        it_data = df_chip[df_chip['name']=='Investment_Trust'].groupby('date')['net'].sum()
        
        foreign_net = int(fi_data.iloc[-3:].sum() if len(fi_data)>=3 else 0)
        trust_net = int(it_data.iloc[-3:].sum() if len(it_data)>=3 else 0)
        
        fi_cons, it_cons = 0, 0
        for val in fi_data.values[::-1]:
            if val > 0: fi_cons += 1
            else: break
        for val in it_data.values[::-1]:
            if val > 0: it_cons += 1
            else: break

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
        df_price['H-L'] = df_price['max'] - df_price['min']
        df_price['H-PC'] = abs(df_price['max'] - df_price['close'].shift(1))
        df_price['L-PC'] = abs(df_price['min'] - df_price['close'].shift(1))
        df_price['TR'] = df_price[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr_14 = df_price['TR'].rolling(14).mean().iloc[-1]
        swing_target = today['close'] + (atr_14 * 2.5)
        trailing_stop = df_price['max'].rolling(20).max().iloc[-1] - (atr_14 * 2.5)

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

        # 細部評分系統 (供戰力解剖使用)
        s_trend = 15 if today['MA5']>today['MA20'] else 0
        s_vwap = 15 if today['close']>today['VWAP_20'] else 0
        s_vol = 25 if today['Volume']>prev['Vol_MA5']*1.5 else 0
        s_chip = (10 if fi_cons >= 3 else 0) + (10 if it_cons >= 3 else 0)
        s_fund = (20 if rev_score > 0 else 0) + (10 if news_score > 0 else 0)

        raw_score = 50 + (s_trend if s_trend>0 else -15) + s_vol + rev_score + news_score + s_chip + s_vwap
        score = int(raw_score * market_multiplier)
        action = "爆發前夕" if score >= 100 else "強勢佈局" if score >= 80 else "偏多觀察" if score >= 60 else "嚴格避開"

        return {
            "代號": stock_code, "名稱": stock_name, "收盤價": round(today['close'], 1),
            "VWAP大戶成本": round(today['VWAP_20'], 1), "明日壓力": round(2*cdp-today['min'], 1),
            "明日支撐": round(2*cdp-today['max'], 1), "波段目標": round(swing_target, 1),
            "動態停利": round(trailing_stop, 1), "防守價": round(today['MA20'], 1),
            "布林上軌": round(today['BB_Upper'], 1), "綜合分數": score, "AI勝率": ai_prob,
            "營收動能": rev_status, "外資": foreign_net, "投信": trust_net, 
            "外資連買": fi_cons, "投信連買": it_cons, "判定": action,
            "新聞情緒": news_status, "最新標題": latest_headlines,
            "子分數": {"技術": (s_trend+s_vwap)/30, "動能": s_vol/25, "籌碼": s_chip/20, "基本": s_fund/30},
            "新聞": f"https://tw.stock.yahoo.com/quote/{stock_code}/news", "歷史資料": df_price 
        }
    except Exception as e: 
        if is_single_mode: st.error(f"🕵️‍♂️ 抓蟲報告：系統內部發生錯誤 ({e})")
        return None

# ==========================================
# 介面顯示邏輯
# ==========================================
dl = DataLoader()
if finmind_token: dl.login_by_token(api_token=finmind_token)

tab1, tab2 = st.tabs(["單股解析", "產業海選"])

with tab1:
    st.markdown(f"**大盤環境指示：** {market_status_text}")
    # 移除資金計算，簡化搜尋欄位
    c_in, c_btn = st.columns([4, 1])
    target_input = c_in.text_input("輸入股票代號或名稱 (如: 2330 或 台積電)", "2330")
    st.write("") # 調整排版間距
    btn_single = c_btn.button("啟動深度解析", use_container_width=True)

    if btn_single:
        user_query = target_input.strip()
        match_df = tw_stocks_df[(tw_stocks_df['stock_id'] == user_query) | (tw_stocks_df['stock_name'] == user_query)]
        
        if match_df.empty: 
            st.error(f"**錯誤：** 找不到代號或名稱為「{user_query}」的股票。")
        else:
            real_code = match_df['stock_id'].values[0]
            real_name = match_df['stock_name'].values[0]
            
            with st.spinner(f"正在鎖定 {real_code} {real_name}，機構級運算模組啟動中..."):
                r = analyze_stock(real_code, real_name, dl, is_single_mode=True)
            
            if r:
                st.markdown("---")
                try: ai_val = float(r['AI勝率'].replace('%',''))
                except: ai_val = 50.0

                st.markdown("## 🎯 系統最終判定與明日劇本")
                
                if r['收盤價'] >= r['布林上軌']: 
                    st.error(f"🚨 **【極度過熱 - 嚴禁追高】**\n\n股價已觸及布林上軌 (\${r['布林上軌']})，短線隨時面臨主力獲利了結賣壓。空手者請觀望，持股者可考慮逢高減碼。")
                elif r['綜合分數'] >= 80 and ai_val <= 45: 
                    st.warning(f"⚠️ **【誘多警告 - 逢低再接】**\n\n雖然籌碼與趨勢極佳（系統評分 {r['綜合分數']}分），但 AI 預測明日勝率僅 {r['AI勝率']}。主力極可能開高走低洗盤！\n\n👉 **行動劇本：** 明日若見急拉至壓力區 \${r['明日壓力']} 切勿追高，請耐心等回測支撐 \${r['明日支撐']} 再進場。")
                elif r['綜合分數'] >= 80 and ai_val > 55: 
                    st.success(f"🚀 **【強勢多頭 - 綠燈通行】**\n\n趨勢、籌碼與 AI 預測達成高度共識！\n\n👉 **行動劇本：** 可於目前價位或支撐 \${r['明日支撐']} 附近分批佈局。嚴守跌破動態停利線 \${r['動態停利']} 停損出場。")
                elif r['收盤價'] < r['VWAP大戶成本']: 
                    st.info(f"❄️ **【弱勢套牢 - 嚴格避開】**\n\n目前股價低於大戶均價 \${r['VWAP大戶成本']}，上方套牢賣壓沉重，資金效率低，請勿進場接刀。")
                else: 
                    st.info(f"⚖️ **【震盪整理 - 靜待表態】**\n\n目前多空不明，缺乏爆發動能。\n\n👉 **行動劇本：** 防守底線為 \${r['防守價']}，建議空手觀望，靜待量能放大。")
                
                st.markdown("---")

                c_t, c_a = st.columns([3, 1])
                c_t.markdown(f"### {r['代號']} {r['名稱']} 戰情報告")
                c_a.metric("AI 預測勝率", r['AI勝率'], help="隨機森林模型根據價量特徵預測下一日收紅機率")

                # 新增：戰力雷達解剖
                st.markdown("##### 🔍 系統評分維度解剖")
                col_bar1, col_bar2, col_bar3, col_bar4 = st.columns(4)
                with col_bar1:
                    st.caption("均線與大戶成本 (技術)")
                    st.progress(min(r['子分數']['技術'], 1.0))
                with col_bar2:
                    st.caption("成交量爆發 (動能)")
                    st.progress(min(r['子分數']['動能'], 1.0))
                with col_bar3:
                    st.caption("法人連買 (籌碼)")
                    st.progress(min(r['子分數']['籌碼'], 1.0))
                with col_bar4:
                    st.caption("營收與新聞 (基本/消息)")
                    st.progress(min(r['子分數']['基本'], 1.0))

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**明日 CDP 區間：** 壓力 \${r['明日壓力']} / 支撐 \${r['明日支撐']}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("收盤價", f"\${r['收盤價']}")
                m2.metric("動態停利線", f"\${r['動態停利']}", help="跌破代表趨勢改變")
                m3.metric("市場輿情", r['新聞情緒'])
                m4.metric("營收 YoY", r['營收動能'])

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("系統評分", f"{r['綜合分數']} ({r['判定']})")
                m6.metric("布林上軌", f"\${r['布林上軌']}", help="股價觸及此線代表短線過熱")
                m7.metric("大戶成本", f"\${r['VWAP大戶成本']}")
                
                chip_txt = "土洋齊買" if r['外資']>0 and r['投信']>0 else "外資偏多" if r['外資']>0 else "投信偏多" if r['投信']>0 else "偏空"
                sub_chip_txt = f"外連買{r['外資連買']}天 / 投連買{r['投信連買']}天" if r['外資連買']>0 or r['投信連買']>0 else f"外:{r['外資']} / 投:{r['投信']}"
                m8.metric("法人連續籌碼", chip_txt, sub_chip_txt)

                if r['最新標題']:
                    st.markdown("#### 📰 最新輿情焦點")
                    for title in r['最新標題']:
                        st.markdown(f"- {title}")

                # 升級：專業 K 線圖 (Plotly)
                st.markdown("#### 📈 專業 K 線與關鍵防守線")
                chart_data = r['歷史資料'].tail(90).set_index('date') # 拉長到 90 天讓 K 線更完整
                try:
                    fig = go.Figure(data=[go.Candlestick(x=chart_data.index,
                                    open=chart_data['open'],
                                    high=chart_data['max'],
                                    low=chart_data['min'],
                                    close=chart_data['close'],
                                    name='日 K 線',
                                    increasing_line_color='#FF4B4B', decreasing_line_color='#00D4FF')])
                    # 疊加均線與成本線
                    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MA20'], mode='lines', name='月線防守', line=dict(color='yellow', width=1.5)))
                    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['VWAP_20'], mode='lines', name='大戶成本', line=dict(color='#A259FF', width=1.5, dash='dot')))
                    
                    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("⚠️ K線圖渲染失敗，請確認已安裝 plotly 套件。")
                
                with st.expander("展開查看：策略回測與資金曲線 (近一年)"):
                    backtest_result = run_backtest(r['歷史資料'])
                    b1, b2, b3 = st.columns(3)
                    b1.metric("交易次數", f"{backtest_result['交易次數']} 次")
                    b2.metric("策略勝率", backtest_result['勝率'])
                    b3.metric("累積總報酬", backtest_result['總報酬'])
                    if not backtest_result['曲線'].empty:
                        st.markdown("##### 策略資金累積曲線")
                        st.line_chart(backtest_result['曲線'])

                st.markdown("---")
                st.link_button("進入 Yahoo 新聞中心看全文", r['新聞'], use_container_width=True)
            else:
                st.error("🚨 **系統終止產出報告！** (請參考上方的🕵️‍♂️抓蟲報告釐清原因)")

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
        else: st.error("🚨 掃描失敗。無法獲取該產業的歷史資料，請確認是否為 API 額度限制。")st.sidebar.markdown("若頻繁查詢導致沒畫面，請至 [FinMind 官網](https://finmindtrade.com/) 免費註冊獲取 Token。")
finmind_token = st.sidebar.text_input("輸入 FinMind Token", type="password", help="留空即為訪客模式 (每小時300次額度)")

@st.cache_data(ttl=86400) 
def load_taiwan_stocks():
    dl_temp = DataLoader()
    info = dl_temp.taiwan_stock_info()
    return info[info['stock_id'].str.len() == 4].dropna(subset=['industry_category'])

tw_stocks_df = load_taiwan_stocks()
industry_list = sorted(tw_stocks_df['industry_category'].unique().tolist())
market_status_text, market_multiplier = get_market_status()

# ==========================================
# 核心大腦 1.0：歷史回測與資金曲線引擎
# ==========================================
def run_backtest(df_price):
    position = 0
    buy_price = 0
    dates = []
    returns = []
    
    for i in range(20, len(df_price)):
        today = df_price.iloc[i]
        prev = df_price.iloc[i-1]
        
        is_bullish = today['MA5'] > today['MA20'] and today['MA20'] > today['MA60']
        vol_fire = today['Volume'] > (prev['Vol_MA5'] * 1.5)
        is_red = today['close'] > today['open']
        
        if position == 0 and is_bullish and vol_fire and is_red:
            position = 1
            buy_price = today['close']
        elif position == 1 and today['close'] < today['MA20']:
            position = 0
            sell_price = today['close']
            profit_pct = (sell_price - buy_price) / buy_price
            dates.append(today['date'])
            returns.append(profit_pct)
            
    if not returns: return {"交易次數": 0, "勝率": "0%", "總報酬": "0%", "曲線": pd.DataFrame()}
    
    win_trades = [t for t in returns if t > 0]
    win_rate = (len(win_trades) / len(returns)) * 100
    total_return = (np.prod([1 + t for t in returns]) - 1) * 100
    
    eq_curve = [1.0]
    for r in returns: eq_curve.append(eq_curve[-1] * (1 + r))
    df_curve = pd.DataFrame({"累積資金率": eq_curve[1:]}, index=dates)
    
    return {"交易次數": len(returns), "勝率": f"{win_rate:.1f}%", "總報酬": f"{total_return:.1f}%", "曲線": df_curve}

# ==========================================
# 核心大腦 1.5：新聞輿情分析爬蟲
# ==========================================
def get_news_sentiment(stock_code):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_code}/news"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=2)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        headlines = []
        for a in soup.find_all('a'):
            text = a.get_text().strip()
            if len(text) > 10 and text not in headlines: 
                headlines.append(text)
        
        headlines = headlines[:8] 
        bullish_kw = ['創高', '成長', '受惠', '看好', '大增', '突破', '買超', '利多', '漲停', '報喜', '接單', '強攻', '反彈']
        bearish_kw = ['衰退', '大跌', '降評', '砍單', '下修', '看壞', '賣超', '利空', '跌破', '探底', '保守', '重挫', '出脫']
        bull_score = sum(1 for t in headlines for k in bullish_kw if k in t)
        bear_score = sum(1 for t in headlines for k in bearish_kw if k in t)

        if bull_score > bear_score: return "利多發酵", 10, headlines[:3]
        elif bear_score > bull_score: return "賣壓利空", -10, headlines[:3]
        else: return "情緒中性", 0, headlines[:3]
    except: return "暫無新聞", 0, []

# ==========================================
# 核心大腦 2.0：分析模組
# ==========================================
def analyze_stock(stock_code, stock_name, dl, is_single_mode=False, price_filter="不限"):
    try:
        pe = 999
        try: pe = yf.Ticker(f"{stock_code}.TW").info.get('trailingPE', 999)
        except: pass

        start_dt = '2022-01-01' if is_single_mode else '2023-06-01'
        df_price = dl.taiwan_stock_daily(stock_id=stock_code, start_date=start_dt) 
        df_chip = dl.taiwan_stock_institutional_investors(stock_id=stock_code, start_date='2024-01-01')
        
        if is_single_mode:
            if df_price.empty:
                st.error(f"🕵️‍♂️ 抓蟲報告：FinMind 伺服器拒絕提供【{stock_code} 歷史股價】。可能原因：免費額度已滿或無資料。")
                return None
            if df_chip.empty:
                st.error(f"🕵️‍♂️ 抓蟲報告：FinMind 伺服器拒絕提供【{stock_code} 三大法人籌碼】。")
                return None
            if len(df_price) < 60:
                st.error(f"🕵️‍♂️ 抓蟲報告：【{stock_code}】歷史資料不足 60 天。")
                return None
        else:
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
        
        df_price['STD20'] = df_price['close'].rolling(20).std()
        df_price['BB_Upper'] = df_price['MA20'] + (2 * df_price['STD20'])

        delta = df_price['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df_price['RSI'] = 100 - (100 / (1 + (gain / loss)))

        today, prev = df_price.iloc[-1], df_price.iloc[-2]

        df_chip['net'] = (df_chip['buy'] - df_chip['sell']) / 1000
        fi_data = df_chip[df_chip['name']=='Foreign_Investor'].groupby('date')['net'].sum()
        it_data = df_chip[df_chip['name']=='Investment_Trust'].groupby('date')['net'].sum()
        
        foreign_net = int(fi_data.iloc[-3:].sum() if len(fi_data)>=3 else 0)
        trust_net = int(it_data.iloc[-3:].sum() if len(it_data)>=3 else 0)
        
        fi_cons, it_cons = 0, 0
        for val in fi_data.values[::-1]:
            if val > 0: fi_cons += 1
            else: break
        for val in it_data.values[::-1]:
            if val > 0: it_cons += 1
            else: break

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
        df_price['H-L'] = df_price['max'] - df_price['min']
        df_price['H-PC'] = abs(df_price['max'] - df_price['close'].shift(1))
        df_price['L-PC'] = abs(df_price['min'] - df_price['close'].shift(1))
        df_price['TR'] = df_price[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr_14 = df_price['TR'].rolling(14).mean().iloc[-1]
        swing_target = today['close'] + (atr_14 * 2.5)
        trailing_stop = df_price['max'].rolling(20).max().iloc[-1] - (atr_14 * 2.5)

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

        bonus = (10 if fi_cons >= 3 else 0) + (10 if it_cons >= 3 else 0)
        score = int((50 + (15 if today['MA5']>today['MA20'] else -15) + (25 if today['Volume']>prev['Vol_MA5']*1.5 else 0) + rev_score + news_score + bonus + (15 if today['close']>today['VWAP_20'] else 0)) * market_multiplier)
        action = "爆發前夕" if score >= 100 else "強勢佈局" if score >= 80 else "偏多觀察" if score >= 60 else "嚴格避開"

        return {
            "代號": stock_code, "名稱": stock_name, "收盤價": round(today['close'], 1),
            "VWAP大戶成本": round(today['VWAP_20'], 1), "明日壓力": round(2*cdp-today['min'], 1),
            "明日支撐": round(2*cdp-today['max'], 1), "波段目標": round(swing_target, 1),
            "動態停利": round(trailing_stop, 1), "防守價": round(today['MA20'], 1),
            "布林上軌": round(today['BB_Upper'], 1), "綜合分數": score, "AI勝率": ai_prob,
            "營收動能": rev_status, "外資": foreign_net, "投信": trust_net, 
            "外資連買": fi_cons, "投信連買": it_cons, "判定": action,
            "新聞情緒": news_status, "最新標題": latest_headlines,
            "新聞": f"https://tw.stock.yahoo.com/quote/{stock_code}/news", "歷史資料": df_price 
        }
    except Exception as e: 
        if is_single_mode: st.error(f"🕵️‍♂️ 抓蟲報告：系統內部發生錯誤 ({e})")
        return None

# ==========================================
# 介面顯示邏輯
# ==========================================
dl = DataLoader()
if finmind_token: dl.login_by_token(api_token=finmind_token)

tab1, tab2 = st.tabs(["單股解析", "產業海選"])

with tab1:
    st.markdown(f"**大盤環境指示：** {market_status_text}")
    c_in, c_cap, c_btn = st.columns([2, 1, 1])
    target_input = c_in.text_input("輸入股票代號或名稱 (如: 2330 或 台積電)", "2330")
    enable_cap = c_cap.checkbox("計算資金部位")
    btn_single = c_btn.button("啟動深度解析", use_container_width=True)
    my_capital = st.number_input("總資金 (萬)", value=50) if enable_cap else 0

    if btn_single:
        user_query = target_input.strip()
        match_df = tw_stocks_df[(tw_stocks_df['stock_id'] == user_query) | (tw_stocks_df['stock_name'] == user_query)]
        
        if match_df.empty: 
            st.error(f"**錯誤：** 找不到代號或名稱為「{user_query}」的股票。")
        else:
            real_code = match_df['stock_id'].values[0]
            real_name = match_df['stock_name'].values[0]
            
            with st.spinner(f"正在鎖定 {real_code} {real_name}，機構級運算模組啟動中..."):
                r = analyze_stock(real_code, real_name, dl, is_single_mode=True)
            
            if r:
                st.markdown("---")
                try: ai_val = float(r['AI勝率'].replace('%',''))
                except: ai_val = 50.0

                st.markdown("## 🎯 系統最終判定與明日劇本")
                
                if r['收盤價'] >= r['布林上軌']: 
                    st.error(f"🚨 **【極度過熱 - 嚴禁追高】**\n\n股價已觸及布林上軌 (\${r['布林上軌']})，短線隨時面臨主力獲利了結賣壓。空手者請觀望，持股者可考慮逢高減碼。")
                elif r['綜合分數'] >= 80 and ai_val <= 45: 
                    st.warning(f"⚠️ **【誘多警告 - 逢低再接】**\n\n雖然籌碼與趨勢極佳（系統評分 {r['綜合分數']}分），但 AI 預測明日勝率僅 {r['AI勝率']}。主力極可能開高走低洗盤！\n\n👉 **行動劇本：** 明日若見急拉至壓力區 \${r['明日壓力']} 切勿追高，請耐心等回測支撐 \${r['明日支撐']} 再進場。")
                elif r['綜合分數'] >= 80 and ai_val > 55: 
                    st.success(f"🚀 **【強勢多頭 - 綠燈通行】**\n\n趨勢、籌碼與 AI 預測達成高度共識！\n\n👉 **行動劇本：** 可於目前價位或支撐 \${r['明日支撐']} 附近分批佈局。嚴守跌破動態停利線 \${r['動態停利']} 停損出場。")
                elif r['收盤價'] < r['VWAP大戶成本']: 
                    st.info(f"❄️ **【弱勢套牢 - 嚴格避開】**\n\n目前股價低於大戶均價 \${r['VWAP大戶成本']}，上方套牢賣壓沉重，資金效率低，請勿進場接刀。")
                else: 
                    st.info(f"⚖️ **【震盪整理 - 靜待表態】**\n\n目前多空不明，缺乏爆發動能。\n\n👉 **行動劇本：** 防守底線為 \${r['防守價']}，建議空手觀望，靜待量能放大。")
                
                st.markdown("---")

                c_t, c_a = st.columns([3, 1])
                c_t.markdown(f"### {r['代號']} {r['名稱']} 戰情報告")
                c_a.metric("AI 預測勝率", r['AI勝率'], help="隨機森林模型根據價量特徵預測下一日收紅機率")

                st.markdown(f"**明日 CDP 區間：** 壓力 \${r['明日壓力']} / 支撐 \${r['明日支撐']}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("收盤價", f"\${r['收盤價']}")
                m2.metric("動態停利線", f"\${r['動態停利']}", help="跌破代表趨勢改變")
                m3.metric("市場輿情", r['新聞情緒'])
                m4.metric("營收 YoY", r['營收動能'])

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("系統評分", f"{r['綜合分數']} ({r['判定']})")
                m6.metric("布林上軌", f"\${r['布林上軌']}", help="股價觸及此線代表短線過熱")
                m7.metric("大戶成本", f"\${r['VWAP大戶成本']}")
                
                chip_txt = "土洋齊買" if r['外資']>0 and r['投信']>0 else "外資偏多" if r['外資']>0 else "投信偏多" if r['投信']>0 else "偏空"
                sub_chip_txt = f"外連買{r['外資連買']}天 / 投連買{r['投信連買']}天" if r['外資連買']>0 or r['投信連買']>0 else f"外:{r['外資']} / 投:{r['投信']}"
                m8.metric("法人連續籌碼", chip_txt, sub_chip_txt)

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
                
                with st.expander("展開查看：策略回測與資金曲線 (近一年)"):
                    backtest_result = run_backtest(r['歷史資料'])
                    b1, b2, b3 = st.columns(3)
                    b1.metric("交易次數", f"{backtest_result['交易次數']} 次")
                    b2.metric("策略勝率", backtest_result['勝率'])
                    b3.metric("累積總報酬", backtest_result['總報酬'])
                    if not backtest_result['曲線'].empty:
                        st.markdown("##### 策略資金累積曲線")
                        st.line_chart(backtest_result['曲線'])

                st.markdown("---")
                st.link_button("進入 Yahoo 新聞中心看全文", r['新聞'], use_container_width=True)
            else:
                st.error("🚨 **系統終止產出報告！** (請參考上方的🕵️‍♂️抓蟲報告釐清原因)")

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
        else: st.error("🚨 掃描失敗。無法獲取該產業的歷史資料，請確認是否為 API 額度限制。")
