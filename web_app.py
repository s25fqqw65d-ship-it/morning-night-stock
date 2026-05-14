import streamlit as st
from FinMind.data import DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
import time

# ==========================================
# 網頁版面與標題設定
# ==========================================
st.set_page_config(page_title="早安晚上好", page_icon="🌅", layout="wide")
st.title("🌅 早安晚上好")
st.markdown("內建 **[營收動能]**、**[資金控管]** 與 **[歷史回測]** 的極致量化終端機。")

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

tw_stocks_df = load_taiwan_stocks()
industry_list = sorted(tw_stocks_df['industry_category'].unique().tolist())

# ==========================================
# 🕰️ 核心大腦二：歷史回測引擎
# ==========================================
def run_backtest(df_price):
    capital = 100000 
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
    
    return {
        "交易次數": len(trades),
        "勝率": f"{win_rate:.1f}%",
        "總報酬": f"{total_return:.1f}%"
    }

# ==========================================
# 🧠 核心大腦一：單檔分析模組 (加入 CDP 與 ATR)
# ==========================================
def analyze_stock(stock_code, stock_name, dl, is_single_mode=False, price_filter="不限"):
    try:
        pe = yf.Ticker(f"{stock_code}.TW").info.get('trailingPE', 999)
        df_price = dl.taiwan_stock_daily(stock_id=stock_code, start_date='2023-06-01') 
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
        df_price['Turnover_Millions'] = (df_price['close'] * df_price['Volume']) / 1000
        df_price['Turnover_MA5'] = df_price['Turnover_Millions'].rolling(5).mean()

        if not is_single_mode and df_price.iloc[-2]['Turnover_MA5'] < 20: return None 

        today = df_price.iloc[-1]
        prev = df_price.iloc[-2]

        foreign = df_chip[df_chip['name'] == 'Foreign_Investor'].iloc[-3:]
        foreign_net = int((foreign['buy'].sum() - foreign['sell'].sum()) / 1000)
        trust = df_chip[df_chip['name'] == 'Investment_Trust'].iloc[-3:]
        trust_net = int((trust['buy'].sum() - trust['sell'].sum()) / 1000)

        # 營收引擎
        rev_status = "無資料"
        rev_score = 0
        try:
            if is_single_mode:
                df_rev = dl.taiwan_stock_month_revenue(stock_id=stock_code, start_date='2023-01-01')
                if not df_rev.empty and len(df_rev) >= 13:
                    latest_rev = df_rev.iloc[-1]['revenue']
                    last_yr_rev = df_rev.iloc[-13]['revenue']
                    yoy = ((latest_rev - last_yr_rev) / last_yr_rev) * 100
                    if yoy > 20:
                        rev_score = 20
                        rev_status = f"🔥 營收大爆發 (YoY +{yoy:.1f}%)"
                    elif yoy > 0:
                        rev_score = 10
                        rev_status = f"📈 穩健成長 (YoY +{yoy:.1f}%)"
                    else:
                        rev_score = -15
                        rev_status = f"📉 衰退中 (YoY {yoy:.1f}%)"
        except: pass

        body = abs(today['close'] - today['open'])
        upper_shadow = today['max'] - max(today['open'], today['close'])
        is_red_candle = today['close'] > today['open']
        
        tomorrow_forecast = "↔️ 震盪機率高"
        if is_red_candle and upper_shadow < (body * 0.5) and today['Volume'] > prev['Vol_MA5']: tomorrow_forecast = "🔥 明日看漲 (動能強)"
        elif upper_shadow > body and upper_shadow > (today['close'] * 0.01): tomorrow_forecast = "⚠️ 防開高走低 (上影線長)"
        elif not is_red_candle and today['close'] < today['MA5']: tomorrow_forecast = "📉 明日偏弱 (破五日線)"

        # 🎯 CDP 明日精準價位推測
        cdp = (today['max'] + today['min'] + 2 * today['close']) / 4
        tomorrow_high_pressure = 2 * cdp - today['min'] 
        tomorrow_low_support = 2 * cdp - today['max']   

        # 🎯 ATR 動能波段滿足點 
        df_price['H-L'] = df_price['max'] - df_price['min']
        df_price['H-PC'] = abs(df_price['max'] - prev['close'])
        df_price['L-PC'] = abs(df_price['min'] - prev['close'])
        df_price['TR'] = df_price[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        atr_14 = df_price['TR'].rolling(14).mean().iloc[-1]
        swing_target = today['close'] + (atr_14 * 2.5)

        trend_score = 15 if (today['MA5'] > today['MA20'] and today['MA20'] > today['MA60']) else (-15 if today['MA5'] < today['MA20'] else 0)
        vol_score = 25 if (today['Volume'] / prev['Vol_MA5'] >= 2.0 and is_red_candle) else (10 if today['Volume'] / prev['Vol_MA5'] >= 1.5 and is_red_candle else 0)
        
        score = 50 + trend_score + vol_score + rev_score
        if pe < 20: score += 15
        if today['close'] > today['MA60']: score += 10 
        
        chip_status = "⚖️ 中性"
        if foreign_net < -2000: score -= 30; chip_status = "📉 狂砍"
        elif foreign_net > 1500: score += 20; chip_status = "🔥 底部建倉"

        trust_status = "➖ 觀望"
        if trust_net > 500:
            score += 15; trust_status = f"🤝 投信進駐"
            if foreign_net > 1500: score += 30; trust_status = "👑 土洋齊買"

        action = "觀望"
        if score >= 100: action = "🚀 爆發前夕"
        elif score >= 80: action = "⭐⭐⭐ 強勢佈局"
        elif score >= 65: action = "🟢 偏多觀察"
        elif score < 50: action = "🔴 嚴格避開"

        return {
            "代號": stock_code,
            "名稱": stock_name,
            "收盤價": round(today['close'], 1),
            "明日預測": tomorrow_forecast,
            "明日壓力": round(tomorrow_high_pressure, 1),
            "明日支撐": round(tomorrow_low_support, 1),   
            "波段目標": round(swing_target, 1),
            "防守價": round(today['MA20'], 1),
            "綜合分數": int(score),
            "營收動能": rev_status,
            "外資": f"{chip_status} ({foreign_net})",
            "投信": f"{trust_status} ({trust_net})",
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
tab1, tab2 = st.tabs(["🔍 模式一：單股深度解析", "🏭 模式二：特定產業海選"])

# ------------------------------------------
# 分頁一：單股狙擊
# ------------------------------------------
with tab1:
    st.markdown("### 🔍 輸入股票代號，啟動全方位健檢")
    
    col_input, col_opt, col_btn = st.columns([2, 1, 1])
    with col_input:
        target_code = st.text_input("輸入股票代號 (例如：2330)", "2330")
    with col_opt:
        st.write(""); st.write("")
        enable_capital = st.checkbox("開啟資金部位計算")
    with col_btn:
        st.write(""); st.write("")
        btn_single = st.button("⚡ 啟動解析", use_container_width=True)
        
    my_capital = None
    if enable_capital:
         my_capital = st.number_input("💵 輸入預計投入總資金 (萬台幣)", value=50, step=10)

    if btn_single:
        target_code = target_code.strip()
        name_match = tw_stocks_df[tw_stocks_df['stock_id'] == target_code]['stock_name']
        
        if name_match.empty: st.error(f"❌ 找不到代號 {target_code}")
        else:
            stock_name = name_match.values[0]
            with st.spinner("資料同步中，準備為您產出戰情報告..."):
                report = analyze_stock(target_code, stock_name, dl, is_single_mode=True)
                
            if report:
                st.markdown("---")
                st.markdown(f"## 📊 【{report['代號']} {report['名稱']}】")
                
                st.info(f"### 🎯 明日走勢預測：{report['明日預測']}")
                
                # CDP 預測警告
                st.warning(f"🔔 **短線戰術 (CDP)：** 預估明日若衝到 **${report['明日壓力']}** 將遭遇賣壓；若回檔至 **${report['明日支撐']}** 會有強勁接盤力道。")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📌 收盤價", f"${report['收盤價']}")
                c2.metric("🎯 動能波段滿足點", f"${report['波段目標']}")
                c3.metric("🛡️ 月線防守", f"${report['防守價']}")
                c4.metric("📈 營收動能 (YoY)", report['營收動能'])

                st.markdown("---")
                c5, c6, c7 = st.columns(3)
                c5.metric("📊 系統綜合評分", f"{report['綜合分數']}分 ({report['判定']})")
                c6.metric("法人動向：外資", report['外資'])
                c7.metric("法人動向：投信", report['投信'])

                # 🛡️ 資金控管部位計算機
                if enable_capital and my_capital:
                    st.markdown("### 🛡️ 資金部位控管建議")
                    risk_money = (my_capital * 10000) * 0.02 
                    price_diff = report['收盤價'] - report['防守價']
                    
                    if price_diff <= 0:
                        st.warning("⚠️ 目前股價已跌破月線防守價，技術面偏弱，建議先空手觀望。")
                    else:
                        max_shares = int(risk_money / (price_diff * 1000))
                        if max_shares == 0:
                            st.warning(f"⚠️ 距離防守價過遠 ($ {price_diff:.1f})。若只承擔 {risk_money:,.0f} 元風險，連 1 張都買不起。建議等待拉回再佈局！")
                        else:
                            st.success(f"✅ **最佳佈局策略：** 建議最多買進 **{max_shares} 張**。若不幸跌破防守價停損，您的總虧損將控制在 **{risk_money:,.0f} 元** 左右 (總資金的 2% 以內)。")

                # 🕰️ 歷史回測驗證模組
                with st.expander("🕰️ 點擊展開：查看近一年歷史回測數據"):
                    backtest_result = run_backtest(report['歷史資料'])
                    st.markdown(f"系統針對 {report['名稱']} 過去一年的走勢進行模擬交易回測：")
                    b1, b2, b3 = st.columns(3)
                    b1.metric("模擬交易次數", f"{backtest_result['交易次數']} 次")
                    b2.metric("策略歷史勝率", backtest_result['勝率'])
                    b3.metric("累積總報酬率", backtest_result['總報酬'])

                st.markdown("---")
                st.link_button(f"👉 點我立即觀看【{report['名稱']}】的最新相關新聞", report['新聞雷達'], type="primary", use_container_width=True)

# ------------------------------------------
# 分頁二：產業海選
# ------------------------------------------
with tab2:
    st.markdown("### 🏭 設定條件，自動篩選潛力股")
    
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
        
    st.info(f"⚡ 鎖定【{selected_industry}】符合條件股票，共 {len(target_stocks)} 檔。")

    if btn_sector:
        results = []
        total_to_scan = len(target_stocks)
        my_bar = st.progress(0, text="正在進行全自動化分析...")
        
        count = 0
        stock_dict = dict(zip(target_stocks['stock_id'], target_stocks['stock_name']))
        
        for code, name in stock_dict.items():
            count += 1
            my_bar.progress(count / total_to_scan, text=f"掃描中: {code} {name} ({count}/{total_to_scan})")
            
            report = analyze_stock(code, name, dl, is_single_mode=False, price_filter=price_filter)
            if report:
                table_report = {
                    "代號": report["代號"],
                    "名稱": report["名稱"],
                    "收盤價": report["收盤價"],
                    "明日預測": report["明日預測"],
                    "外資": report["外資"],
                    "投信": report["投信"],
                    "新聞雷達": report["新聞雷達"]
                }
                results.append(table_report)
            time.sleep(0.3) 
            
        my_bar.empty()
        if results:
            df_all = pd.DataFrame(results).sort_values(by="收盤價", ascending=False)
            st.subheader(f"🔥 決選名單 🔥")
            st.dataframe(df_all, use_container_width=True, hide_index=True, column_config=column_config)
        else:
            st.warning("未能找到符合所有條件的股票。")
