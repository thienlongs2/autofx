import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential
Sequential = tf.keras.models.Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import ta
import pickle

# ===================== 1️⃣ Lấy dữ liệu từ MT5 =====================
def get_mt5_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, num_bars=5000):
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

# ===================== 2️⃣ Tiền xử lý dữ liệu =====================
def preprocess_data(df):
    df['high_low_range'] = df['high'] - df['low']
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    bollinger = ta.volatility.BollingerBands(df['close'], window=20)
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()

    df.dropna(inplace=True)
    return df

# ===================== 3️⃣ Chuẩn hóa & tạo sequences =====================
def create_sequences(df, sequence_length=50):
    features = ['close', 'high_low_range', 'sma_10', 'sma_50', 'rsi', 'macd', 'macd_signal', 'bollinger_high', 'bollinger_low']

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[['close']])

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(X_scaled[i:i + sequence_length])
        y.append(y_scaled[i + sequence_length])

    # Lưu scaler để dùng cho dự đoán sau này
    with open("scaler.pkl", "wb") as f:
        pickle.dump((scaler_X, scaler_y), f)

    return np.array(X), np.array(y), scaler_X, scaler_y

# ===================== 4️⃣ Xây dựng mô hình LSTM =====================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')
    return model

# ===================== 5️⃣ Huấn luyện mô hình =====================
df = get_mt5_data()
if df is not None:
    df = preprocess_data(df)
    X, y, scaler_X, scaler_y = create_sequences(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    # Callback tối ưu
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    checkpoint = ModelCheckpoint("best_lstm_model.keras", save_best_only=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr, checkpoint])

    # ===================== 6️⃣ Đánh giá mô hình =====================
    model = tf.keras.models.load_model("best_lstm_model.keras")  # Load model tốt nhất

    y_pred = model.predict(X_test)
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_test_real = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    print(f"📉 MSE: {mse:.6f}, RMSE: {rmse:.6f}")

    # 📊 Biểu đồ loss
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Biểu đồ Loss khi huấn luyện")
    plt.show()

    # 📊 Biểu đồ dự đoán
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label="Giá thực tế", color='blue')
    plt.plot(y_pred_real, label="Giá dự đoán", color='red')
    plt.legend()
    plt.title("Dự đoán giá EUR/USD bằng LSTM")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá EUR/USD")
    plt.show()

    print("✅ Huấn luyện hoàn tất! Mô hình tốt nhất đã được lưu.")

