import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
load_model = tf.keras.models.load_model
from sklearn.preprocessing import MinMaxScaler


# ===== 1. Kết nối với MT5 =====
def connect_mt5():
    if not mt5.initialize():
        print("Kết nối MT5 thất bại!")
        mt5.shutdown()
    else:
        print("✅ Kết nối MT5 thành công!")


# ===== 2. Lấy dữ liệu từ MT5 =====
def get_latest_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, n=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


# ===== 3. Tiền xử lý dữ liệu =====
def preprocess_data(df, scaler):
    df['price_change'] = df['close'].pct_change()
    df['high_low_range'] = df['high'] - df['low']
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df.dropna(inplace=True)

    scaled_data = scaler.transform(df[['close', 'price_change', 'high_low_range', 'sma_10', 'sma_50']])
    X_input = np.array(scaled_data).reshape(1, scaled_data.shape[0], scaled_data.shape[1])
    return X_input


# ===== 4. Dự đoán giá bằng mô hình LSTM =====
def predict_price(model, df, scaler):
    X_input = preprocess_data(df, scaler)
    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform([[predicted_price[0][0], 0, 0, 0, 0]])[0][0]
    return predicted_price


# ===== 5. Gửi lệnh giao dịch đến MT5 =====
def place_trade(action, symbol="EURUSD", lot=0.1, sl_pips=50, tp_pips=100):
    mt5.initialize()
    price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": price - sl_pips * 0.0001 if action == "BUY" else price + sl_pips * 0.0001,
        "tp": price + tp_pips * 0.0001 if action == "BUY" else price - tp_pips * 0.0001,
        "deviation": 20,
        "magic": 1001,
        "comment": "AI Trading",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    mt5.shutdown()
    return result


# ===== 6. Chạy mô hình và giao dịch =====
def run_trading():
    connect_mt5()
    model = load_model("lstm_model3.keras")
    data = get_latest_data("EURUSD", mt5.TIMEFRAME_M1, 100)

    scaler = MinMaxScaler()
    scaler.fit(data[['close', 'price_change', 'high_low_range', 'sma_10', 'sma_50']])

    predicted_price = predict_price(model, data, scaler)
    latest_close = data.iloc[-1]['close']

    if predicted_price > latest_close:
        place_trade("BUY")
    elif predicted_price < latest_close:
        place_trade("SELL")


# ===== Chạy chương trình =====
if __name__ == "__main__":
    run_trading()
