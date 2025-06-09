# 导入所需的库
# 使用PlaidML作为Keras的后端
import plaidml.keras
plaidml.keras.install_backend()
from PIL import Image,ImageOps
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam

#图像风格统一化预处理函数

def preprocess_hard_sample(image_path):
    """将图像转为mnist风格(黑底白字+居中+缩放+28x28)"""
    img = Image.open(image_path).convert('L') #转灰度
    img = ImageOps.invert(img) #黑底白字
    img = ImageOps.autocontrast(img) #增强对比度（可选）

    #缩小图像至20x20以内，保持原始比例
    img.thumbnail((20, 20)),Image.ANTIALTAS
    
    #创建28x28黑底画布
    canvas = Image.new('L', (28, 28), (0))
    upper_left = ((28 - img.size[0]) // 2, (28 - img_size[1]) // 2)
    canvas.paste(img, upper_left)

    return  np.array(canvs)

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#加载困难样本
hard_samples_dir = 'C:/Users/Batman/Documents/coding/python/hard-samples'
hard_imgs = []
hard_labels = []

#遍历文件夹
for filename in os.listdir(hard_samples_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    label = int(filename.split('_')[0])  # 文件名如 "7_001.jpg"
    img_path = os.path.join(hard_samples_dir, filename)
    processed_img = preprocess_hard_sample(img_path)
    hard_imgs.append(processed_img)
    hard_labels.append(label)


# 转换为numpy数组
hard_imgs = np.array(hard_imgs).astype('float32') / 255.0
hard_imgs = hard_imgs.reshape(-1, 28, 28, 1)
hard_labels = np.array(hard_labels)

# 原始 MNIST 预处理
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)

# 合并数据
x_train = np.concatenate([x_train, hard_imgs], axis=0)
y_train = np.concatenate([y_train, hard_labels], axis=0)

# 将标签进行one-hot编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 将图像数据归一化并调整形状
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# 构建卷积神经网络模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),#使用dropout正则化，可适当降低防止欠拟合
    Dense(10, activation='softmax')
])

# 编译模型
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
optimizer = Adam(lr=0.001)  # 可以试试 0.0005 或 0.005
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.2)

# 保存模型
model.save('mnist_model.h5')

# 绘制训练和验证的学习曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='training accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('training/validation accuracy curve')
plt.legend()

# 绘制训练和验证的损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('training/validation loss curve')
plt.legend()
plt.show()
