
![[Region Based Segmentation.png]]
- Сегментация на основе порогового значения (**threshold**)
```python
gray = rgb2gray(image)
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
# Попиксельный проход по изображению закраживание пикселя в цвет класса
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')
```

+Быстрые расчёты
+При высокой контрастности  легко различить объекты
-Когда нет существенной разницы в оттенках серого или есть перекрытие значений пикселей в градациях серого, становится очень трудно получить точные сегменты.