import streamlit as st
import numpy as np 
from skimage import io
import matplotlib.pyplot as plt

age = st.slider('На сколько сжать фото?', 0, 666, 666)

url = 'https://cs.pikabu.ru/post_img/big/2013/03/17/6/1363508611_1596589037.jpg'
image = io.imread(url)

U, sing_vals, V = np.linalg.svd(image)

sigma = np.zeros_like(image, dtype=np.float64)

np.fill_diagonal(sigma, sing_vals)

top_k = age

trunc_U = U[:, :top_k]
trunc_sigma = sigma[:top_k, :top_k]
trunc_V = V[:top_k, :]

trunc_image = trunc_U @ trunc_sigma @ trunc_V
# fig, ax = plt.subplots(1, 2, figsize=(15, 20))
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title('Исходное изобржение')
# ax[1].imshow(trunc_image, cmap='gray')
# ax[1].set_title(f'Изображение на top {top_k} сингулярных чисел');
# st.pyplot(fig)
fig1, ax1 = plt.subplots(figsize=(15, 20))
ax1.imshow(image, cmap='gray')

fig2, ax2 = plt.subplots(figsize=(15, 20))
ax2.imshow(trunc_image, cmap='gray')
st.pyplot(fig1)
st.pyplot(fig2)
Proch = 1 - age/666
st.write(f'Фото сжато на {int(Proch * 100)} %')