
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import ta

Sequential = tf.keras.models.Sequential

# 1️⃣ Kết nối MetaTrader 5 & lấy dữ liệu EUR/USD
def get_mt5_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, num_bars=10000):
    if not mt5.initialize():
        print("Kết nối MT5 thất bại!", mt5.last_error())
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("Không có dữ liệu từ MT5!")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


# 2️⃣ Tiền xử lý dữ liệu & thêm chỉ báo kỹ thuật
def preprocess_data(df):
    df['price_change'] = df['close'] - df['open']
    df['high_low_range'] = df['high'] - df['low']

    # Trung bình động
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_slope'] = df['sma_10'] - df['sma_50']

    # Chỉ báo RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Chỉ báo MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20)
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()

    df.dropna(inplace=True)  # Xóa các dòng NaN
    return df


# 3️⃣ Tạo chuỗi dữ liệu thời gian
def create_sequences(df, sequence_length=30):
    features = ['close', 'price_change', 'high_low_range', 'sma_10', 'sma_50', 'sma_slope',
                'rsi', 'macd', 'macd_signal', 'bollinger_high', 'bollinger_low']

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[features].iloc[i:i + sequence_length].values)
        y.append(df['close'].iloc[i + sequence_length])

    return np.array(X), np.array(y)


# 4️⃣ Chia dữ liệu & chuẩn hóa bằng MinMaxScaler
def prepare_training_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# 5️⃣ Xây dựng mô hình LSTM
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


# 6️⃣ Chạy toàn bộ quy trình
df = get_mt5_data()
if df is not None:
    df = preprocess_data(df)
    X, y = create_sequences(df)
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_training_data(X, y)

    # Xây dựng mô hình
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    # Huấn luyện mô hình
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

    # 7️⃣ Dự đoán & đánh giá mô hình
    y_pred = model.predict(X_test)

    # Chuyển đổi về giá trị thực
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_test_real = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    print(f"📉 MSE: {mse:.6f}, RMSE: {rmse:.6f}")

    # 📊 Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label="Thực tế", color='blue')
    plt.plot(y_pred_real, label="Dự đoán", color='red')
    plt.legend()
    plt.title("Dự đoán giá EUR/USD bằng LSTM")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá EUR/USD")
    plt.show()

    print("✅ Huấn luyện hoàn tất!")

    # Sau khi huấn luyện xong
    model.save("lstm_model.h5")
    print("✅ Mô hình đã được lưu thành công!")
    # Load mô hình đã lưu






