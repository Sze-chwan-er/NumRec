# 导入所需的库
from os import environ
environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 加载模型
model = load_model('digit_recognition_model.h5')

# 加载并预处理图像
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# 识别图像
def predict_digit(img_path):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

# 测试模型
img_path = 'C:/Users/Batman/Documents/coding/python/newnum/6.png'  # 替换为你的图片路径
predicted_digit = predict_digit(img_path)
print(f'The predicted digit is: {predicted_digit}')
