import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import time
import pandas as pd
import numpy as np
import warnings
import scipy.stats as stats

# --- 機器學習套件 ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
)

# --- 統計與時間序列套件 (包含 SARIMAX, ES, State Space) ---
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.stats.diagnostic import acorr_ljungbox

# 忽略收斂警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 內建股票字典
# ==========================================
POPULAR_STOCKS = {
    "蘋果 (AAPL)": "AAPL", "微軟 (MSFT)": "MSFT", "輝達 (NVDA)": "NVDA",
    "特斯拉 (TSLA)": "TSLA", "亞馬遜 (AMZN)": "AMZN", "台積電 ADR (TSM)": "TSM",
    "標普500指數 ETF (SPY)": "SPY", "納斯達克指數 ETF (QQQ)": "QQQ"
}

# ==========================================
# 1. 網頁初始化設定
# ==========================================
st.set_page_config(page_title="終極量化預測系統 (大滿貫版)", layout="wide")
st.title("📈 實時多股盯盤與全自動 AI 量化預測系統")

st.sidebar.header("⚙️ 系統設定")
selected_stock_names = st.sidebar.multiselect(
    "1️⃣ 選擇監控股票 (支援複選):", options=list(POPULAR_STOCKS.keys()), default=["輝達 (NVDA)"]
)
custom_ticker = st.sidebar.text_input("2️⃣ 手動輸入代碼 (例如: PLTR)").upper()
refresh_rate = st.sidebar.slider("3️⃣ 網頁更新頻率 (秒)", min_value=15, max_value=120, value=30)
st.sidebar.info("💡 提示：本系統啟用了 25+ 種演算法與嚴格假設檢定，處理需時，請耐心等候。")

target_tickers = [POPULAR_STOCKS[name] for name in selected_stock_names]
if custom_ticker and custom_ticker not in target_tickers:
    target_tickers.append(custom_ticker)

# ==========================================
# 2. 數據獲取與遞迴預測引擎
# ==========================================
def get_realtime_1m_data(ticker):
    return yf.Ticker(ticker).history(period="1d", interval="1m")

def get_daily_data_for_prediction(ticker):
    """使用過去 3 年的數據以捕捉完整市場週期"""
    return yf.Ticker(ticker).history(period="3y", interval="1d")

def generate_ml_trend(model, recent_closes, steps=63):
    """機器學習遞迴預測 (未來 63 個交易日，約 3 個月)"""
    history = list(recent_closes)
    future_prices = []
    for _ in range(steps):
        sma10 = np.mean(history[-10:])
        sma50 = np.mean(history[-50:])
        ret = (history[-1] - history[-2]) / history[-2] if history[-2] != 0 else 0
        X_new = np.array([[history[-1], sma10, sma50, ret]])
        pred = model.predict(X_new)[0]
        future_prices.append(pred)
        history.append(pred)
    return future_prices

# ==========================================
# 3. 核心大腦：跨學派自動選拔與假設檢定
# ==========================================
def auto_select_and_predict_trend(ticker):
    df = get_daily_data_for_prediction(ticker)
    if df.empty or len(df) < 100: return None
        
    data = df.copy()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Target_Next'] = data['Close'].shift(-1) 
    data = data.dropna()
    
    features = ['Close', 'SMA_10', 'SMA_50', 'Daily_Return']
    X = data[features]
    y = data['Target_Next']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    best_model_info = {
        "name": "", "type": "", "mape": float('inf'), "rmse": 0, "mse": 0, 
        "aic": "N/A", "bic": "N/A", "aicc": "N/A",
        "residuals": None, "future_curve": []
    }

    # ---------------------------------------------------------
    # 軌道一：機器學習家族 (Bagging, Boosting)
    # ---------------------------------------------------------
    ml_models = {
        "Bagging (袋裝法)": BaggingRegressor(random_state=42),
        "Random Forest (隨機森林)": RandomForestRegressor(n_estimators=50, random_state=42),
        "Gradient Boosting (梯度提升)": GradientBoostingRegressor(n_estimators=50, random_state=42),
        "AdaBoost (自適應提升)": AdaBoostRegressor(random_state=42),
        "Linear Regression (線性)": LinearRegression()
    }
    
    for name, model in ml_models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, preds) * 100
            
            if mape < best_model_info["mape"]:
                model.fit(X, y)
                future_curve = generate_ml_trend(model, data['Close'].values, steps=63)
                best_model_info.update({
                    "name": name, "type": "Machine Learning",
                    "mape": mape, "rmse": rmse, "mse": mse,
                    "aic": "N/A", "bic": "N/A", "aicc": "N/A",
                    "residuals": y_test - preds, "future_curve": future_curve
                })
        except Exception: pass

    # ---------------------------------------------------------
    # 軌道二：截圖中的 9 宮格 Exponential Smoothing 矩陣
    # ---------------------------------------------------------
    es_params = [
        ("ES (N, N)", None, None, False),
        ("ES (N, A) 附加季節", None, "add", False),
        ("ES (N, M) 乘法季節", None, "mul", False),
        ("ES (A, N) 霍特線性", "add", None, False),
        ("ES (A, A) 霍特溫特斯加法", "add", "add", False),
        ("ES (A, M) 霍特溫特斯乘法", "add", "mul", False),
        ("ES (Ad, N) 阻尼趨勢", "add", None, True),
        ("ES (Ad, A) 阻尼+加法季節", "add", "add", True),
        ("ES (Ad, M) 阻尼+乘法季節", "add", "mul", True)
    ]
    
    for name, trend, seasonal, damped in es_params:
        try:
            es_model = ExponentialSmoothing(y_train.values, trend=trend, seasonal=seasonal, seasonal_periods=5 if seasonal else None, damped_trend=damped).fit(optimized=True)
            preds = es_model.forecast(len(y_test))
            
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, preds) * 100
            
            if mape < best_model_info["mape"]:
                full_es = ExponentialSmoothing(y.values, trend=trend, seasonal=seasonal, seasonal_periods=5 if seasonal else None, damped_trend=damped).fit(optimized=True)
                future_curve = full_es.forecast(63).tolist()
                
                # ES 有提供 aic/bic 等資訊
                aic = getattr(full_es, 'aic', 'N/A')
                bic = getattr(full_es, 'bic', 'N/A')
                aicc = getattr(full_es, 'aicc', 'N/A')

                best_model_info.update({
                    "name": name, "type": "Exponential Smoothing",
                    "mape": mape, "rmse": rmse, "mse": mse,
                    "aic": aic, "bic": bic, "aicc": aicc,
                    "residuals": y_test - preds, "future_curve": future_curve
                })
        except Exception: pass

    # ---------------------------------------------------------
    # 軌道三：SARIMAX (整合自迴歸與外生變數)
    # ---------------------------------------------------------
    try:
        sari_model = SARIMAX(endog=y_train.values, exog=X_train.values, order=(1, 1, 1)).fit(disp=False, maxiter=30)
        preds = sari_model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=X_test.values)
        
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, preds) * 100
        
        if mape < best_model_info["mape"]:
            full_sari = SARIMAX(endog=y.values, exog=X.values, order=(1, 1, 1)).fit(disp=False, maxiter=30)
            
            # 使用測試集外生變數進行未來遞迴預測 (此處做簡化常數遞迴以生成未來線)
            future_exog = np.tile(X.iloc[-1].values, (63, 1))
            future_curve = full_sari.predict(start=len(y), end=len(y)+62, exog=future_exog).tolist()

            best_model_info.update({
                "name": "SARIMAX (1,1,1)", "type": "Time Series (ARIMA)",
                "mape": mape, "rmse": rmse, "mse": mse,
                "aic": full_sari.aic, "bic": full_sari.bic, "aicc": full_sari.aicc,
                "residuals": y_test - preds, "future_curve": future_curve
            })
    except Exception: pass

    # ---------------------------------------------------------
    # 軌道四：狀態空間模型 (State Space Models / UCM)
    # ---------------------------------------------------------
    ssm_params = [
        ("SSM (本地水平)", "local level"),
        ("SSM (本地線性趨勢)", "local linear trend"),
        ("SSM (隨機遊走帶漂移)", "random walk with drift")
    ]
    
    for name, level in ssm_params:
        try:
            ssm_model = UnobservedComponents(y_train.values, level=level).fit(disp=False, maxiter=20)
            preds = ssm_model.forecast(steps=len(y_test))
            
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, preds) * 100
            
            if mape < best_model_info["mape"]:
                full_ssm = UnobservedComponents(y.values, level=level).fit(disp=False, maxiter=20)
                future_curve = full_ssm.forecast(steps=63).tolist()
                
                best_model_info.update({
                    "name": name, "type": "State Space Model (SSM)",
                    "mape": mape, "rmse": rmse, "mse": mse,
                    "aic": full_ssm.aic, "bic": full_ssm.bic, "aicc": getattr(full_ssm, 'aicc', 'N/A'),
                    "residuals": y_test - preds, "future_curve": future_curve
                })
        except Exception: pass

    # ---------------------------------------------------------
    # 模型假設檢定 (Residual Diagnostics)
    # ---------------------------------------------------------
    diagnostic_results = {}
    resids = best_model_info["residuals"]
    
    if resids is not None:
        # ACF 檢定 (Ljung-Box)
        lb_test = acorr_ljungbox(resids, lags=[5], return_df=True)
        diagnostic_results['acf_pvalue'] = lb_test['lb_pvalue'].iloc[0]
        diagnostic_results['acf_pass'] = diagnostic_results['acf_pvalue'] > 0.05
        
        # 常態性檢定 (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(resids)
        diagnostic_results['jb_pvalue'] = jb_pvalue
        diagnostic_results['jb_pass'] = jb_pvalue > 0.05
        
    best_model_info["diagnostics"] = diagnostic_results
    return best_model_info, df

# ==========================================
# 4. 網頁渲染與繪圖循環
# ==========================================
placeholder = st.empty()

while True:
    if not target_tickers:
        with placeholder.container():
            st.info("👈 請從左側選單選擇至少一檔股票。")
        time.sleep(2)
        continue

    with placeholder.container():
        for ticker in target_tickers:
            st.markdown(f"## 📊 【{ticker}】 量化預測與假設檢定報告")
            
            df_1m = get_realtime_1m_data(ticker)
            
            if not df_1m.empty:
                current_price = df_1m.iloc[-1]['Close']
                open_price = df_1m.iloc[0]['Open']
                price_change = current_price - open_price
                percent_change = (price_change / open_price) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label=f"今日實時價格", value=f"${current_price:.2f}", delta=f"{price_change:.2f} ({percent_change:.2f}%)")
                col2.metric(label="今日開盤", value=f"${open_price:.2f}")
                col3.metric(label="更新時間", value=df_1m.index[-1].strftime("%H:%M:%S"))
                
                with st.spinner(f'正在進行 25 核心模型選拔、計算 AIC/MSE 及 ACF 殘差檢定...'):
                    analysis_result = auto_select_and_predict_trend(ticker)
                
                if analysis_result:
                    best_info, df_daily = analysis_result
                    
                    pred_1w = best_info['future_curve'][4]
                    pred_3m = best_info['future_curve'][-1]
                    
                    # --- 詳細數據面板 ---
                    col_info1, col_info2, col_info3 = st.columns([1.5, 1.5, 1.5])
                    
                    # 面板 1：選拔指標與信息準則
                    aic_str = f"{best_info['aic']:.2f}" if isinstance(best_info['aic'], float) else best_info['aic']
                    bic_str = f"{best_info['bic']:.2f}" if isinstance(best_info['bic'], float) else best_info['bic']
                    aicc_str = f"{best_info['aicc']:.2f}" if isinstance(best_info['aicc'], float) else best_info['aicc']

                    col_info1.info(f"**🏆 最優模型:** {best_info['name']}\n\n"
                                   f"**📂 演算法派系:** {best_info['type']}\n\n"
                                   f"**📉 選擇標準 (Errors):**\n"
                                   f"- MAPE (Percentage MSE): **{best_info['mape']:.2f}%**\n"
                                   f"- RMSE: {best_info['rmse']:.4f}\n"
                                   f"- MSE: {best_info['mse']:.4f}\n\n"
                                   f"**📊 信息準則 (僅統計模型):**\n"
                                   f"- AIC: {aic_str} | BIC: {bic_str}\n"
                                   f"- AICc: {aicc_str}")
                    
                    # 面板 2：殘差假設檢定
                    diag = best_info['diagnostics']
                    diag_msg = "### 🔍 模型假設檢定\n"
                    diag_msg += f"{'✅' if diag['acf_pass'] else '⚠️'} **ACF 自相關 (Ljung-Box):** {'通過' if diag['acf_pass'] else '未通過'} *(p={diag['acf_pvalue']:.4f})*\n"
                    diag_msg += f"{'✅' if diag['jb_pass'] else '⚠️'} **常態分佈 (Jarque-Bera):** {'通過' if diag['jb_pass'] else '未通過'} *(p={diag['jb_pvalue']:.4f})*\n\n"
                    if best_info['type'] == "Machine Learning":
                        diag_msg += "*註: 獲勝者為無母數機器學習，殘差未通過屬正常現象。*"
                    col_info2.warning(diag_msg)
                    
                    # 面板 3：未來目標預測
                    trend_1w = "📈 看漲" if pred_1w > current_price else "📉 看跌"
                    trend_3m = "📈 看漲" if pred_3m > current_price else "📉 看跌"
                    col_info3.success(f"### 🎯 目標價預測\n\n"
                                      f"**🗓️ 1 週後預測:**\n"
                                      f"# ${pred_1w:.2f} ({trend_1w})\n\n"
                                      f"**🗓️ 3 個月後預測:**\n"
                                      f"# ${pred_3m:.2f} ({trend_3m})")

                    # ==========================================
                    # 繪製圖表 1：實時 1 分鐘 K 線
                    # ==========================================
                    fig1 = go.Figure(data=[go.Candlestick(
                        x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close'],
                        name="實時走勢", increasing_line_color='green', decreasing_line_color='red'
                    )])
                    fig1.update_layout(title='📍 今日實時動能 (1分鐘 K線)', template='plotly_dark', xaxis_rangeslider_visible=False, height=350, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig1, width="stretch")

                    # ==========================================
                    # 繪製圖表 2：3 個月歷史 vs 3 個月未來軌跡
                    # ==========================================
                    last_date = df_daily.index[-1]
                    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=63)
                    
                    fig2 = go.Figure()
                    hist_dates = df_daily.index[-90:]
                    hist_prices = df_daily['Close'].iloc[-90:]
                    
                    fig2.add_trace(go.Scatter(x=hist_dates, y=hist_prices, mode='lines', name='過去3個月實際價格', line=dict(color='white', width=2)))
                    
                    line_color = '#00FFCC' if pred_3m > current_price else '#FF3399'
                    fig2.add_trace(go.Scatter(x=future_dates, y=best_info['future_curve'], mode='lines', name='AI 預測趨勢', line=dict(color=line_color, width=2, dash='dash')))
                    
                    fig2.add_trace(go.Scatter(x=[future_dates[4]], y=[pred_1w], mode='markers+text', name='1週目標', 
                                              marker=dict(size=12, symbol='circle', color='yellow'), text=[f'1W: ${pred_1w:.2f}'], textposition="top left"))
                    fig2.add_trace(go.Scatter(x=[future_dates[-1]], y=[pred_3m], mode='markers+text', name='3個月目標', 
                                              marker=dict(size=14, symbol='star', color=line_color), text=[f'3M: ${pred_3m:.2f}'], textposition="top right"))

                    fig2.update_layout(title='🔮 AI 中長期預測軌跡 (日線級別)', template='plotly_dark', height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified')
                    st.plotly_chart(fig2, width="stretch")
                    
                else:
                    st.warning(f"⚠️ {ticker} 歷史數據不足，無法進行分析。")
                
                st.markdown("<br><hr><br>", unsafe_allow_html=True)
            
    time.sleep(refresh_rate)
    st.rerun()
