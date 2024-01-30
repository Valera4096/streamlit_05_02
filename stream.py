import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize


url = st.text_input('Вставьте URL ссылку с изображением', 'https://dobrovserdce.ru/images/2022/11/02/kot%20Fedya_large.jpeg')
image = io.imread(url)

age = st.slider('На сколько сжать фото?', 0, 100, 50)

compressed_channels = []
top_k = int(image.shape[0] - image.shape[0]/100 * age)  

if len(image.shape) == 3:
    
    for i in range(3):  # Проходим по каждому цветовому каналу
        U, sing_vals, V = np.linalg.svd(image[:,:,i])  
        # Создаем диагональную матрицу с топ-K сингулярными значениями
        sigma = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
        np.fill_diagonal(sigma, sing_vals)
        
        trunc_U = U[:, :top_k]
        trunc_sigma = sigma[:top_k, :top_k]
        trunc_V = V[:top_k, :]
        # Формула для восстановления сжатого изображения из усеченных матриц U, sigma, V
        compressed_channel = trunc_U @ trunc_sigma @ trunc_V
        compressed_channels.append(compressed_channel)

    # Объединяем сжатые цветовые каналы в одно цветное изображение
    compressed_image = np.stack(compressed_channels, axis=-1).astype('uint8')
else:      
    U, sing_vals, V = np.linalg.svd(image)
    sigma = np.zeros_like(image, dtype=np.float64)

    np.fill_diagonal(sigma, sing_vals)

    top_k = int(image.shape[0] - image.shape[0]/100 * age)

    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]
    compressed_image = trunc_U @ trunc_sigma @ trunc_V

fig1, ax1 = plt.subplots(figsize=(15, 20))
ax1.imshow(image, cmap='gray')

fig2, ax2 = plt.subplots(figsize=(15, 20))
ax2.imshow(compressed_image, cmap='gray')

st.pyplot(fig1)
st.pyplot(fig2)

st.write(f'Фото сжато на {age} %')



