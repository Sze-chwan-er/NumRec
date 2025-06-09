# 导入所需的库
# 使用PlaidML作为Keras的后端
import plaidml.keras
plaidml.keras.install_backend()
import numpy as np
from keras.models import load_model
from PIL import Image,ImageOps

# 使用PlaidML作为Keras的后端
import plaidml.keras
plaidml.keras.install_backend()

# 加载训练好的模型
#model = load_model('mnist_model.h5')
model = load_model('digit_recognition_model.h5')

# 定义一个函数来预处理图像并识别数字
def recognize_image(image_path):
    img = Image.open(image_path).convert('L')#转灰度
    # 反色（如果你的图片是白底黑字，MNIST是黑底白字）
    img = ImageOps.invert(img)
    img = img.resize((28, 28))#调整图像大小

    # 将图像数据归一化并调整形状
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # 使用模型进行预测
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    # 将标签10和11转换回1和7
    if predicted_class == 10:
        predicted_class = 1
    elif predicted_class == 11:
        predicted_class = 7
    
    return predicted_class

# 测试函数
image_path = 'C:/Users/Batman/Documents/coding/python/newnum/9.png'
result = recognize_image(image_path)
print(f'识别结果: {result}')
