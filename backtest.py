import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
import ta
import pickle

# ===================== 1️⃣ Load mô hình & Scaler =====================
model = tf.keras.models.load_model("best_lstm_model.keras")

# Load scaler đã lưu từ file pickle
try:
    with open("scaler.pkl", "rb") as f:
        scaler_X, scaler_y = pickle.load(f)
except FileNotFoundError:
    print("⚠️ Không tìm thấy scaler.pkl! Hãy chắc chắn đã lưu khi train.")
    exit()


# ===================== 2️⃣ Lấy dữ liệu từ MT5 =====================
def get_mt5_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, num_bars=100):
    if not mt5.initialize():
        print("⚠️ Lỗi kết nối MT5!")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("⚠️ Không có dữ liệu từ MT5!")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


# ===================== 3️⃣ Tiền xử lý dữ liệu =====================
def preprocess_data(df):
    df['high_low_range'] = df['high'] - df['low']
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bollinger_high'] = bb.bollinger_hband()
    df['bollinger_low'] = bb.bollinger_lband()

    df.dropna(inplace=True)
    return df


# ===================== 4️⃣ Chuẩn bị dữ liệu đầu vào =====================
def create_sequences(df, sequence_length=50):
    features = ['close', 'high_low_range', 'sma_10', 'sma_50', 'rsi', 'macd', 'macd_signal', 'bollinger_high',
                'bollinger_low']

    # Kiểm tra đủ đặc trưng chưa
    if not all(feature in df.columns for feature in features):
        missing_features = [f for f in features if f not in df.columns]
        print(f"⚠️ Thiếu các đặc trưng sau: {missing_features}")
        return None

    # Lấy dữ liệu mới nhất
    df_recent = df[features].iloc[-sequence_length:].values

    # Kiểm tra đủ số lượng cột chưa
    if df_recent.shape[0] < sequence_length:
        print(f"⚠️ Không đủ {sequence_length} dữ liệu gần nhất để dự đoán!")
        return None

    # Chuẩn hóa dữ liệu
    df_scaled = scaler_X.transform(df_recent)
    X = df_scaled.reshape(1, sequence_length, len(features))
    return X


# ===================== 5️⃣ Dự đoán giá tiếp theo =====================
def predict():
    df = get_mt5_data()
    if df is None:
        return None

    df = preprocess_data(df)
    X = create_sequences(df)

    if X is None:
        return None

    # Dự đoán
    prediction_scaled = model.predict(X)[0][0]

    # Chuyển giá trị về giá trị thực
    predicted_price = scaler_y.inverse_transform([[prediction_scaled]])[0][0]
    return predicted_price


# ===================== 6️⃣ Chạy script dự đoán =====================
if __name__ == "__main__":
    predicted_price = predict()
    if predicted_price is not None:
        print(f"🔮 Dự đoán giá tiếp theo: {predicted_price:.5f}")
    else:
        print("⚠️ Không thể thực hiện dự đoán!")
