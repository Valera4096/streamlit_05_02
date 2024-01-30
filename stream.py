import streamlit as st
# import numpy as np 
# from skimage import io
# import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize


url = st.text_input('Вставьте URL ссылку с изображением', 'https://dobrovserdce.ru/images/2022/11/02/kot%20Fedya_large.jpeg')
image = io.imread(url)

age = st.slider('На сколько сжать фото?', 0, 100, 50)

if len(image.shape) == 3:
    gray_image = rgb2gray(image)
else:
    gray_image = image
    
U, sing_vals, V = np.linalg.svd(gray_image)
sigma = np.zeros_like(gray_image, dtype=np.float64)

np.fill_diagonal(sigma, sing_vals)

top_k = int(image.shape[0] - image.shape[0]/100 * age)

trunc_U = U[:, :top_k]
trunc_sigma = sigma[:top_k, :top_k]
trunc_V = V[:top_k, :]
trunc_image = trunc_U @ trunc_sigma @ trunc_V

fig1, ax1 = plt.subplots(figsize=(15, 20))
ax1.imshow(image, cmap='gray')

fig2, ax2 = plt.subplots(figsize=(15, 20))
ax2.imshow(trunc_image, cmap='gray')
st.pyplot(fig1)
st.pyplot(fig2)
Proch = 1 - age/666
st.write(f'Фото сжато на {age} %')



