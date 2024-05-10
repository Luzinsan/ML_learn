# Edge Detection Segmentation
> Суть: обнаружение ребер объектов (edges) -> определение границ объектов
> Для этого используются ***фильтры*** и ***свёртки*** (filters и [[Convolutions]])

## Weight matrix
### Sobel Operator
> Имеет две матрицы весов - для обнаружения горизонтальных и вертикальных границ (detect edges)

Sobel filter (horizontal) = $S_H = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\  -1 & -2 & -1 \end{bmatrix}$

Sobel filter (vertical) = $S_V = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\  -1 & 0 & 1 \end{bmatrix}$
### Laplace Operator
> Также позволяет обнаруживать как горизонтальные, так и вертикальные границы, но состоит из одном матрицы

Laplas filter = $L = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\  1 & 1 & 1 \end{bmatrix}$
#### Способ применения
```python
kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
plt.imshow(out_l, cmap='gray')
```
