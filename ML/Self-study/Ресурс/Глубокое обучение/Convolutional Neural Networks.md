Требования при разработке архитектуры нейронной сети, пригодной для Computer Vision:
1. Принцип трансляционной инвариантности (_translation invariance_, _translation equivariance_) - в первых слоях сеть должна одинаково реагировать на один и тот же patch, независимо от того, где он появляется на изображении (the children’s game “Where’s Waldo”)
   ![[Where’s Waldo.png|400]]
2. Принцип локальности - первые слои должны быть сосредоточены на локальных регионах, не принимая внимание на содержимое изображений в отдалённых регионах. Далее эти локальные представления можно объединить в более крупные до уровня всего изображения
3. Более глубокие слои должны быть способны улавливать детали изображения на более дальнем расстоянии, аналогично зрению более высокого порядка.

# Свёрточный слой
![[Convolution.png|500]]
1. Пусть есть $[\mathbf{X}]_{i, j}$ и $[\mathbf{H}]_{i, j}$  - это пиксели в позиции $(i,j)$ входного изображения (одноканального). Для выражения полносвязанного слоя нужно, чтобы каждый из скрытых нейронов получали данные от каждого из входных пикселей => нужно использовать четырехмерную матрицу весов $\mathbf{W}$ и смещение $\mathbf{U}$: $$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}$$
- Между $\mathbf{X}$ и $\mathbf{H}$ существует взаимно-однозначное соответствие, поэтому допустили переход от $\mathbf{W}$ к $\mathbf{V}$ ($k=i+a$ и $l=j+b$)
	- Общее кол-во параметров при заданном изображении $1000 \times 1000$: $10^{12}$
2. Применение принципа *1.трансляционной инвариантности* => сдвиг значений во входных данных $\mathbf{X}$ должен привести к соответствующему сдвигу во входном представлении $\mathbf{H}$ => $\mathbf{V}$ и $\mathbf{U}$ не зависят от $(i,j)$ => определение $\mathbf{H}$ упрощается: $$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
- получившееся выражение - ***свёртка*** 
	- Общее кол-во параметров теперь: $4 \times 10^6$ и зависимость $a,b \in (-1000,1000)$ 
3. Применение принципа *2.локальности* => нет необходимости смотреть всё изображение, чтобы оценить пиксель $(i,j)$, достаточно рассматривать какую-то неотдалённую область от $(i,j)$, т.е. ограничить $|a|>\Delta$ и $|b|>\Delta$: $$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
- получивлееся выражение - ***свёрточный слой***
- *Свёрточные нейронные сети (CNN - Convolutional Neural Network)* - это особое семейство нейронных сетей, содержащих свёрточные слои.
- $\mathbf{V}$ - ***ядро свёртки*** (convolution kernel), фильтр (filter) или матрица весов, окно свёртки (convolution window) - является обучаемым параметров
	- Общее кол-во параметров теперь: $4 \times \Delta ^2$
	- Значение $\Delta$ обычно меньше $10$
### Вычислительная сложность
- Image $(h \times w)$, kernel $(k \times k)$: $\mathcal{O}(h * w * k^2)$
- Добавляем каналы $c_i, c_o$: $\mathcal{O}(h*w*k^2*c_i*c_o)$

## Channels
На самом деле, рассматривая цветные изображения, получаем формулу свёрточного слоя: $$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$
### Для получения двумерного выходного тензора:
  ![[Convolution Layer multi-input.png]]
	- Срез по каждому каналу => выполнение cross-correlation => суммирование результатов по каждому каналу и получение двумерного тензора:
```python
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```
### Для получения трёхмерного выходного тензора
- Пусть $c_i$ и $c_o$ -  кол-во входных и выходных каналов соответственно 
- $k_h$ и $k_w$ высота и ширина ядра. 
- Чтобы получить вывод с несколькими каналами, мы можем создать тензор ядра формы $c_i \times k_h \times k_w$ для *каждого выходного канала*. 
- Объединяем их по каждому выходному каналу => форма ядра: $c_o \times c_i \times k_h \times k_w$. 
- В cross-correlation результат для каждого выходного канала вычисляется на основе ядра свертки, соответствующего этому выходному каналу, и принимает входные данные со *всех каналов* во входном тензоре.
```python
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are
    # stacked together
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```
### 1 x 1 Convolutional Layer
![[Convolution Layer.png]]
> Применяется для получения свёртки вдоль измерения канала (например: из 3-канального изображения в 2-канальное)

```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
```

---
Свёрточный слой - несовсем корректное название -> выражаемую операцию правильнее называть операцией взаимной корреляции (***cross-correlation operation***) (но всё же это разные операции...)
- *Форма выходного слоя* определяется формой входного тензора $n_h \times n_w$ и формой ядра свёртки $k_h \times k_w$: $$(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1).$$
- ***Выходной слой*** называют картой признаков (*feature map*)
- Для некоторого элемента выходного слоя после свёртки его восприимчивым полем (*receptive field*) будут все элементы во входном окне свёртки. Поэтому, если элементу карты признаков требуется большее рецептивное поле для обнаружения большего контекста, то для этого строят более глубокие сети.
### From Scratch:
- операция свёртки одного окна
```python
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
```
- Свёртка всего тензора:
```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```
- Обучение ядра свёртки:
```python
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
        
# epoch 2, loss 16.481
# epoch 4, loss 5.069
# epoch 6, loss 1.794
# epoch 8, loss 0.688
# epoch 10, loss 0.274

conv2d.weight.data.reshape((1, 2))
# tensor([[ 1.0398, -0.9328]])
```

# Техники контроля размера выходных данных
## Padding (Заполнение)  - increase shape
- Решает проблему: потеря информации на границах исходного изображения после свёртки -> пиксели по периметру не используются, и проблема усугубляется при использовании нескольких свёрточных слоёв: ![[Padding reason.png]]
- Решение: добавление дополнительных пикселей (нули) вокруг границы изображения: ![[Padding.png]]
- При добавлении $p_h$ и $p_w$ пикселей по вертикали и горизонтали (по краям)($\times 2$) соответственно, выходная форма будет иметь вид: $$(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+1).$$
- Если требуется, чтобы форма выхода и входа была одинакова: $p_\textrm{h}=k_\textrm{h}-1$, $p_\textrm{w}=k_\textrm{w}-1$
- Значение размера *ядра свёртки* обычно устанавливается *нечетное* (легче сохранить форму входного изображения при использовании padding, т.к. кол-во пикселей со всех сторон заполняется **одинаковое**)
- Указание разных значений padding'а ядра свёртки:
  `nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))`

### Пример сохранения размерности:
```python
# We define a helper function to calculate convolutions. It initializes the
# convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
# torch.Size([8, 8])
```

## Stride (Шаг) - reduce shape
- Решает проблему: избыточное качество изображения.
- Применяется: для повышения скорости вычислений, для понижения дискретизации; если ядро свертки велико.
- Пример техники stride с шагом 3 по вертикали и 2 по горизонтали:
   ![[Stride.png]]
- Выходная форма при stride по высоте $s_h$ и ширине $s_w$ имеет вид:  $$\lfloor(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+s_\textrm{h})/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+s_\textrm{w})/s_\textrm{w}\rfloor.$$
- Если $p_h=k_h-1, p_w=k_w-1$ то выходная форма упрощается: $\lfloor(n_\textrm{h}+s_\textrm{h}-1)/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}+s_\textrm{w}-1)/s_\textrm{w}\rfloor$
- Если получается целочисленные значения в следующем выражении, то это есть выходная форма: $(n_\textrm{h}/s_\textrm{h}) \times (n_\textrm{w}/s_\textrm{w})$
### Using Example pytorch:
```python
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
# torch.Size([4, 4])

conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
# torch.Size([2, 2])
```
# Pooling
> **Цели:** снижает чувствительность свёрточных слоёв кк местоположению и пространственно понижает дискретизацию представлений
> **Принцип:** состоит из окна фиксированной формы (***pooling window***)(популярный выбор: $2 \times 2$), которое скользит по всем областям входных данных в соответствие с его шагом, вычисляя один выходной результат для каждой области. Не содержит параметров (***нет ядра***). А оператор объединения является нелинейным: *max-pooling* (более предпочтительный) или *average pooling*.
> *(Функция активации с мире computer vision)*

- ***stride*** по умолчанию равен размеру pooling window
- При обработке многоканальных входных данных pooling layer объединяет каждый входной канал отдельно, а не суммирует входные данные по каналам, как в сверточном слое.
### Examples:
- Implementation:
```python
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
# tensor([[4., 5.],
#         [7., 8.]])
```
- Multiple channels:
```python
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]],

         [[ 1.,  2.,  3.,  4.],
          [ 5.,  6.,  7.,  8.],
          [ 9., 10., 11., 12.],
          [13., 14., 15., 16.]]]])
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
tensor([[[[ 5.,  7.],
          [13., 15.]],

         [[ 6.,  8.],
          [14., 16.]]]])
```

## Maximum Pooling
![[MaxPooling.png]]
- `pool2d = nn.MaxPool2d(3)` - max-pooling слой с размером pooling window (3,3), stride=(3,3)
## Average Pooling