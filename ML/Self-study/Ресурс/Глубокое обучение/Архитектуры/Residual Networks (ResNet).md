# Residual Block
- Остаточный блок $g(x)=f(x)-x$ 
- Если тождественное отображение $f(x)=x$ является желаемым базовым отображением, то остаточное отображение (веса и смещения последнего слоя) нужно свести к нулю $g(x)=0$
- Оператор сложения - *остаточное соединение* 
![[Usual & Residual Block.png]]

> ResNet имеет структуру, как у [[Networks Using Blocks (VGG)|VGG]] 
> Остаточный блок имеет две $3\times 3$ свертки с одинаковым количеством выходных каналов
> После каждого сверточного слоя следует слой [[Batch Normalization]] и [[Функции активации#ReLU|ReLU]]
> x добавляется перед последним ReLU и требуется, чтобы размерность x и выхода сверточного слоя совпадала (для согласованной операции остаточного соединения)
> Для изменения количества каналов можно ввести свертку $1\times 1$ 
![[Residual Block & with conv1-1.png]]
```python
class Residual(nn.Module):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

- Сохранение размерности:
```python
blk = Residual(3)
X = torch.randn(4, 3, 6, 6)
blk(X).shape
--------------------------
torch.Size([4, 3, 6, 6])
```
- Именьшение размерности и увеличение кол-ва признаков в 2 раза:
```python
blk = Residual(6, use_1x1conv=True, strides=2)
blk(X).shape
--------------------------
torch.Size([4, 6, 3, 3])
```
# ResNet Model
- Первые 2 слоя такие же, как и у [[GoogLeNet]]: $7\times 7$ сверточный слой с 64 выходными каналами (stride=2), max-pooling (k_s=$3\times 3$, stride=2) 
- Отличие: после каждого сверточного слоя добавляется *Batch Normalization* 
```python
class ResNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```
- В то время как GoogLeNet использует 4 Inception блока, ResNet использует 4 Residual blocks (остаточных блока), каждый из которых использует *несколько* остаточных блоков с одинаковым количеством выходных каналов.
- Кол-во каналов в 1-м модуле совпадает с кол-вом входных каналов
- В первом residual block для каждого из последующих модулей кол-во каналов удваивается, а размерность уменьшается вдвое (за счет stride=2)
```python
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels))
    return nn.Sequential(*blk)
```

![[ResNet.png|1000]]
- В каждом модуле имеется 4 сверточных слоя (не считая $1\times 1$) + $7\times 7$ сверточный слой вначале и fully connected слой в конце получается 18 слоев => в данном случае представлена ResNet-18 модель
- Структура ResNet проще и легче поддается модицикациям
```python
class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                       lr, num_classes)

ResNet18().layer_summary((1, 1, 96, 96))
-----------------------------------------------------------------
Sequential output shape:     torch.Size([1, 64, 24, 24])
Sequential output shape:     torch.Size([1, 64, 24, 24])
Sequential output shape:     torch.Size([1, 128, 12, 12])
Sequential output shape:     torch.Size([1, 256, 6, 6])
Sequential output shape:     torch.Size([1, 512, 3, 3])
Sequential output shape:     torch.Size([1, 10])
```

# ResNeXt
```python
class ResNeXtBlock(nn.Module):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)

-----------Using---------------
blk = ResNeXtBlock(32, 16, 1)
X = torch.randn(4, 32, 96, 96)
blk(X).shape
-------------------------------
torch.Size([4, 32, 96, 96])
```
![[ResNeXT groups.png]]

- Основная идея: использование множества независимых групп, где в каждой ветви применяется одно и то же преобразование
- В конце стоит сгруппированная свертка: каждая группа имеет размерность $c_i/g$, выход которой размерности $c_o/g$ 
- Вычислительные затраты по сравнению с GoogLeNet $O(c_i * c_o)$ уменьшаются до $\mathcal{O}(g*\frac{c_i}{g} * \frac{c_o}{g}) = \mathcal{O} (\frac{c_i*c_o}{g})$ 
- Количество параметров для генерации выходных данных также уменьшается с $c_i\times c_o$ до $g$ матриц размером $\frac{c_i}{g}\times\frac{c_o}{g}$ 