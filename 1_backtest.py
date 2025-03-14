import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
load_model = tf.keras.models.load_model
from sklearn.preprocessing import MinMaxScaler


# ✅ Kết nối MT5 và lấy dữ liệu backtest
def get_mt5_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, num_bars=1000):
    if not mt5.initialize():
        print("Lỗi kết nối MT5!", mt5.last_error())
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("Không lấy được dữ liệu!")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


# ✅ Xử lý dữ liệu (Giữ nguyên như khi train)
import ta


def preprocess_data(df):
    df['price_change'] = df['close'] - df['open']
    df['high_low_range'] = df['high'] - df['low']

    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_slope'] = df['sma_10'] - df['sma_50']

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    bollinger = ta.volatility.BollingerBands(df['close'], window=20)
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()

    df.dropna(inplace=True)
    return df


# ✅ Tạo chuỗi thời gian (Dùng đúng sequence_length và features)
def create_sequences(df, sequence_length=20):
    features = ['close', 'price_change', 'high_low_range', 'sma_10', 'sma_50', 'sma_slope',
                'rsi', 'macd', 'macd_signal', 'bollinger_high', 'bollinger_low']

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[features].iloc[i:i + sequence_length].values)
        y.append(df['close'].iloc[i + sequence_length])

    return np.array(X), np.array(y)


# ✅ Load mô hình đã train
model = load_model("lstm_model.keras")  # Đảm bảo model đã lưu đúng định dạng

# ✅ Chuẩn bị dữ liệu backtest
df = get_mt5_data()
if df is not None:
    df = preprocess_data(df)
    X_test, y_test = create_sequences(df)

    # Chuẩn hóa dữ liệu (Dùng MinMaxScaler y như khi train)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_test = scaler_X.fit_transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_test = scaler_y.fit_transform(y_test.reshape(-1, 1))

    # ✅ Dự đoán giá
    y_pred = model.predict(X_test)

    # Chuyển về giá trị thực tế
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_test_real = scaler_y.inverse_transform(y_test)

    # ✅ Hiển thị kết quả
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label="Thực tế", color='blue')
    plt.plot(y_pred_real, label="Dự đoán", color='red')
    plt.legend()
    plt.title("Backtest LSTM Dự Đoán Giá EUR/USD")
    plt.show()

    print("✅ Backtest hoàn tất!")
