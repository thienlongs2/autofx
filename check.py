import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("best_lstm_model.keras")

# Test với dữ liệu giả định
test_input = np.random.rand(1, 50, 9)  # 50 bước thời gian, 9 đặc trưng
prediction = model.predict(test_input)
print("✅ Output của model:", prediction)
