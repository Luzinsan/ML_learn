# Clustering-based Image Segmentation
[[k-means clustering]]
Для работы с k-means нужно предварительно:
- стандартизировать значения в интервал [0,1] 
- конвертировать в shape`(height*width, channels)`
```python
pic = plt.imread('1.jpeg')/255
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)
```

- хорошо работает на небольшом наборе данных
- алгоритм основан на вычислении расстояний и применим только к выпуклым наборам данных 
- просматривает все пиксели (семплы) на каждой итерации -> затрачиваемое время слишком велико
