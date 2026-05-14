import streamlit as st
from FinMind.data import DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
import time

# 載入機器學習套件
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    st.error("⚠️ 系統偵測到未安裝 scikit-learn！請去 GitHub 的 requirements.txt 加上 'scikit-learn' 後等待重新部署。")

# ==========================================
# 網頁版面與標題設定
# ==========================================
st.set_page_config(page_title="早安晚上好", page_icon="🌅", layout="wide")
st.title("🌅 早安晚上好 (終極神級版)")
st.markdown("內建 **[大盤濾網]**、**[機器學習]**、**[大戶成本]** 與 **[動態停利]** 的機構級量化終端機。")

# ==========================================
# 側邊欄設定
# ==========================================
st.sidebar.header("⚙️ 系統設定")
finmind_token = st.sidebar.text_input("輸入 FinMind Token", type="password", help="留空即為訪客模式（每小時300次額度）")

@st.cache_data(ttl=86400) 
def load_taiwan_stocks():
    dl_temp = DataLoader()
    info = dl_temp.taiwan_stock_info()
    tw_stocks = info[info['stock_id'].str.len() == 4]
    tw_stocks = tw_stocks[tw_stocks['industry_category'].notna()]
    return tw_stocks

# ==========================================
# 🛡️ 核心大腦零：大盤環境濾網 (Market Regime)
# ==========================================
@st.cache_data(ttl=3600)
def get_market_status():
    """抓取加權指數，判斷目前大盤是多頭還是空頭"""
    try:
        twii = yf.Ticker("^TWII").history(period="3mo")
        twii['MA20'] = twii['Close'].rolling(20).mean()
        last_close = twii['Close'].iloc[-1]
        last_ma20 = twii['MA20'].iloc[-1]
        
        if last_close > last_ma20:
            return "📈 多頭格局 (月線上)", 1.2  # 分數加權 1.2 倍
        else:
            return "📉 空頭格局 (破月線)", 0.7  # 分數打 7 折
    except:
        return "⚖️ 震盪不明", 1.0

tw_stocks_df = load_taiwan_stocks()
industry_list = sorted(tw_stocks_df['industry_category'].unique().tolist())
market_status_text, market_multiplier = get_market_status()

# ==========================================
# 🕰️ 核心大腦二：歷史回測引擎
# ==========================================
def run_backtest(df_price):
    position = 0
    buy_price = 0
    trades = []
    
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
            trades.append(profit_pct)
            
    if not trades: return {"交易次數": 0, "勝率": "0%", "總報酬": "0%"}
    
    win_trades = [t for t in trades if t > 0]
    win_rate = (len(win_trades) / len(trades)) * 100
    total_return = (np.prod([1 + t for t in trades]) - 1) * 100
    return {"交易次數": len(trades), "勝率": f"{win_rate:.1f}%", "總報酬": f"{total_return:.1f}%"}

# ==========================================
# 🧠 核心大腦一：單檔究極分析模組
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
        
        # VWAP 大戶均價成本線 (20日)
        df_price['Turnover'] = df_price['close'] * df_price['Volume']
        df_price['VWAP_20'] = df_price['Turnover'].rolling(20).sum() / df_price['Volume'].rolling(20).sum()
        
        # 計算 RSI 供 AI 使用
        delta = df_price['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_price['RSI'] = 100 - (100 / (1 + rs))

        today = df_price.iloc[-1]
        prev = df_price.iloc[-2]

        foreign = df_chip[df_chip['name'] == 'Foreign_Investor'].iloc[-3:]
        foreign_net = int((foreign['buy'].sum() - foreign['sell'].sum()) / 1000)
        trust = df_chip[df_chip['name'] == 'Investment_Trust'].iloc[-3:]
        trust_net = int((trust['buy'].sum() - trust['sell'].sum()) / 1000)

        rev_status = "無資料"
        rev_score = 0
        if is_single_mode:
            try:
                df_rev = dl.taiwan_stock_month_revenue(stock_id=stock_code, start_date='2023-01-01')
                if not df_rev.empty and len(df_rev) >= 13:
                    yoy = ((df_rev.iloc[-1]['revenue'] - df_rev.iloc[-13]['revenue']) / df_rev.iloc[-13]['revenue']) * 100
                    rev_score = 20 if yoy > 20 else (10 if yoy > 0 else -15)
                    rev_status = f"{'🔥 大爆發' if yoy>20 else '📈 成長' if yoy>0 else '📉 衰退'} (YoY {yoy:.1f}%)"
            except: pass

        # CDP 與 ATR 動能
        cdp = (today['max'] + today['min'] + 2 * today['close']) / 4
        tomorrow_high_pressure = 2 * cdp - today['min'] 
        tomorrow_low_support = 2 * cdp - today['max']   

        df_price['H-L'] = df_price['max'] - df_price['min']
        df_price['H-PC'] = abs(df_price['max'] - prev['close'])
        df_price['L-PC'] = abs(df_price['min'] - prev['close'])
        df_price['TR'] = df_price[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr_14 = df_price['TR'].rolling(14).mean().iloc[-1]
        swing_target = today['close'] + (atr_14 * 2.5)
        
        # Chandelier Exit 動態移動停利線 (最高價 - 2.5倍ATR)
        recent_20_high = df_price['max'].rolling(20).max().iloc[-1]
        trailing_stop = recent_20_high - (atr_14 * 2.5)

        # AI 機器學習引擎 (Random Forest)
        ai_prob_str = "未啟動"
        if is_single_mode and len(df_price) > 200:
            try:
                ml_df = df_price[['close', 'Volume', 'MA5', 'MA20', 'RSI']].copy().dropna()
                ml_df['Target'] = (ml_df['close'].shift(-1) > ml_df['close']).astype(int)
                
                train_data = ml_df.dropna()
                X = train_data[['close', 'Volume', 'MA5', 'MA20', 'RSI']]
                y = train_data['Target']
                
                clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                clf.fit(X, y)
                
                latest_features = ml_df[['close', 'Volume', 'MA5', 'MA20', 'RSI']].iloc[-1:]
                prob_up = clf.predict_proba(latest_features)[0][1] * 100
                ai_prob_str = f"🤖 {prob_up:.1f}%"
            except Exception as e:
                ai_prob_str = "模型運算失敗"

        trend_score = 15 if (today['MA5'] > today['MA20'] and today['MA20'] > today['MA60']) else (-15 if today['MA5'] < today['MA20'] else 0)
        vol_score = 25 if (today['Volume'] / prev['Vol_MA5'] >= 2.0 and today['close']>today['open']) else 0
        
        raw_score = 50 + trend_score + vol_score + rev_score
        if pe < 20: raw_score += 15
        if today['close'] > today['VWAP_20']: raw_score += 15
        
        final_score = int(raw_score * market_multiplier)

        action = "觀望"
        if final_score >= 100: action = "🚀 爆發前夕"
        elif final_score >= 80: action = "⭐⭐⭐ 強勢佈局"
        elif final_score >= 60: action = "🟢 偏多觀察"
        elif final_score < 50: action = "🔴 嚴格避開"

        return {
            "代號": stock_code,
            "名稱": stock_name,
            "收盤價": round(today['close'], 1),
            "VWAP大戶成本": round(today['VWAP_20'], 1),
            "明日壓力": round(tomorrow_high_pressure, 1),
            "明日支撐": round(tomorrow_low_support, 1),   
            "波段目標": round(swing_target, 1),
            "動態停利": round(trailing_stop, 1),
            "防守價": round(today['MA20'], 1),
            "綜合分數": final_score,
            "AI勝率": ai_prob_str,
            "營收動能": rev_status,
            "外資": foreign_net,
            "投信": trust_net,
            "判定": action,
            "新聞雷達": f"https://tw.stock.yahoo.com/quote/{stock_code}/news",
            "歷史資料": df_price 
        }
    except Exception as e:
        if is_single_mode: st.warning(f"⚠️ {stock_code} 內部運算中斷: {e}")
        return None

# ==========================================
# 登入 FinMind 與表格設定
# ==========================================
dl = DataLoader()
if finmind_token:
    try: dl.login_by_token(api_token=finmind_token)
    except: pass

column_config = {
    "新聞雷達": st.column_config.LinkColumn("📰 查新聞", display_text="點擊觀看 ↗")
}

# ==========================================
# 🌟 建立分頁標籤 (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["🔍 模式一：單股深度狙擊", "🏭 模式二：特定產業海選"])

# ------------------------------------------
# 分頁一：單股狙擊
# ------------------------------------------
with tab1:
    st.markdown(f"### 📊 今日大盤環境指示燈：**{market_status_text}** (影響系統評分標準)")
    
    col_input, col_opt, col_btn = st.columns([2, 1, 1])
    with col_input:
        target_code = st.text_input("輸入股票代號 (例如：2330)", "2330")
    with col_opt:
        st.write(""); st.write("")
        enable_capital = st.checkbox("開啟資金部位計算")
    with col_btn:
        st.write(""); st.write("")
        btn_single = st.button("⚡ 啟動 AI 深度解析", use_container_width=True)
        
    my_capital = None
    if enable_capital:
         my_capital = st.number_input("💵 預計投入資金 (萬台幣)", value=50, step=10)

    if btn_single:
        target_code = target_code.strip()
        name_match = tw_stocks_df[tw_stocks_df['stock_id'] == target_code]['stock_name']
        
        if name_match.empty: st.error(f"❌ 找不到代號 {target_code}")
        else:
            stock_name = name_match.values[0]
            with st.spinner("AI 隨機森林模型訓練中 & 數據深度解析..."):
                report = analyze_stock(target_code, stock_name, dl, is_single_mode=True)
                
            if report:
                st.markdown("---")
                c_title, c_ai = st.columns([3, 1])
                c_title.markdown(f"## 📊 【{report['代號']} {report['名稱']}】")
                c_ai.info(f"**AI 預測明日上漲機率：\n{report['AI勝率']}**")
                
                st.warning(f"🔔 **短線戰術 (CDP)：** 預估明日若衝到 **${report['明日壓力']}** 將遭遇賣壓；若回檔至 **${report['明日支撐']}** 會有強勁接盤力道。")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📌 收盤價", f"${report['收盤價']}")
                c2.metric("🎯 動能滿足點 (不設限)", f"${report['波段目標']}")
                c3.metric("🏃‍♂️ 動態移動停利線", f"${report['動態停利']}")
                c4.metric("📈 營收動能 (YoY)", report['營收動能'])

                st.markdown("---")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("📊 系統綜合判定", f"{report['綜合分數']}分 ({report['判定']})")
                c6.metric("🛡️ 月線防守", f"${report['防守價']}")
                c7.metric("🏦 VWAP 大戶均價", f"${report['VWAP大戶成本']}")
                
                chip_txt = "土洋齊買👑" if report['外資']>0 and report['投信']>0 else "外資偏多" if report['外資']>0 else "偏空"
                c8.metric("法人動向", chip_txt, f"外:{report['外資']} / 投:{report['投信']}")

                if enable_capital and my_capital:
                    st.markdown("### 🛡️ 資金部位控管建議")
                    risk_money = (my_capital * 10000) * 0.02 
                    price_diff = report['收盤價'] - report['動態停利']
                    
                    if price_diff <= 0:
                        st.warning("⚠️ 目前股價已跌破動態停利線，趨勢轉弱，建議先空手觀望。")
                    else:
                        max_shares = int(risk_money / (price_diff * 1000))
                        st.success(f"✅ **最佳佈局策略：** 建議最多買進 **{max_shares} 張**。跌破動態防守價即停損，虧損控制在 **{risk_money:,.0f} 元** 內。")

                with st.expander("🕰️ 查看近一年歷史策略回測數據"):
                    backtest_result = run_backtest(report['歷史資料'])
                    b1, b2, b3 = st.columns(3)
                    b1.metric("交易次數", f"{backtest_result['交易次數']} 次")
                    b2.metric("策略勝率", backtest_result['勝率'])
                    b3.metric("累積總報酬", backtest_result['總報酬'])

                st.markdown("---")
                st.link_button(f"👉 觀看【{report['名稱']}】最新相關新聞", report['新聞雷達'], type="primary", use_container_width=True)

# ------------------------------------------
# 分頁二：產業海選
# ------------------------------------------
with tab2:
    st.markdown("### 🏭 自動篩選潛力股")
    
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        selected_industry = st.selectbox("1️⃣ 選擇產業板塊：", industry_list, index=industry_list.index("半導體業") if "半導體業" in industry_list else 0)
    with col_sel2:
        market_filter = st.selectbox("2️⃣ 市場別：", ["不限", "上市 (twse)", "上櫃 (tpex)"])
    with col_sel3:
        price_filter = st.selectbox("3️⃣ 股價區間：", ["不限", "100元以下", "100~500元", "500元以上"])
        
    btn_sector = st.button("🚀 啟動掃描", type="primary", use_container_width=True)
    target_stocks = tw_stocks_df[tw_stocks_df['industry_category'] == selected_industry]
    if market_filter == "上市 (twse)": target_stocks = target_stocks[target_stocks['type'] == 'twse']
    elif market_filter == "上櫃 (tpex)": target_stocks = target_stocks[target_stocks['type'] == 'tpex']

    if btn_sector:
        results = []
        total_to_scan = len(target_stocks)
        my_bar = st.progress(0, text="全自動化分析中...")
        
        count = 0
        stock_dict = dict(zip(target_stocks['stock_id'], target_stocks['stock_name']))
        
        for code, name in stock_dict.items():
            count += 1
            my_bar.progress(count / total_to_scan, text=f"掃描: {code} {name} ({count}/{total_to_scan})")
            report = analyze_stock(code, name, dl, is_single_mode=False, price_filter=price_filter)
            if report:
                results.append({
                    "代號": report["代號"], "名稱": report["名稱"], "收盤價": report["收盤價"],
                    "評分": report["綜合分數"], "判定": report["判定"], "新聞雷達": report["新聞雷達"]
                })
            time.sleep(0.1) 
            
        my_bar.empty()
        if results:
            df_all = pd.DataFrame(results).sort_values(by="評分", ascending=False)
            st.subheader(f"🔥 決選名單 (已套用大盤權重) 🔥")
            st.dataframe(df_all, use_container_width=True, hide_index=True, column_config=column_config)
        else:
            st.warning("未能找到符合條件的股票。")
