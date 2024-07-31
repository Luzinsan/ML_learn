![[DenseNet 1.png]]
Характеризуется:
- соединительным паттерном - каждый слой связывается со всеми предыдущими слоями
- операцией конкатенации (в отличие от оператора сложения в [[Residual Networks (ResNet)|ResNet]]). 
  ResNet & DenseNet: ![[ResNet block & DenseNet block.png]]
- ResNet раскладывает f на простой линейный член $\mathbf{x}$ и более сложный нелинейный $g(\mathbf{x})$ ($f(\mathbf{x})=\mathbf{x}+g(\mathbf{x})$), тогда как DenseNet собирает информацию за пределами двух членов -> т.е. выполняется наложение x с его значениями после применения все более сложной последовательности функций: $\mathbf{x} \to \left[\mathbf{x},f_1(\mathbf{x}),f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), f_3\left(\left[\mathbf{x},f_1\left(\mathbf{x}\right), f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right)\right]\right), \ldots\right]$
- Все эти признаки в конечном счете объединяются в MLP, который сокращает кол-во признаков и граф зависимости между признаками становится достаточно плотным (последний слой цепочки плотно связан со всеми предыдущими слоями): 
  ![[Dense connections (DenseNet).png]]
> Основные компоненты: dense blocks (объединение входов и выходов) и transition layers (контроль количества каналов)

### Dense Blocks
```python
def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))
```
- (из модифицированной версии [[Residual Networks (ResNet)#Residual Block|Residual blocks]]) - состоит из [[Batch Normalization]], [[Функции активации#ReLU|ReLU]] и [[Convolutions|сверточного слоя]]
```python
class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X
```
- блоки свертки используют одинаковое кол-во выходных каналов
- во время прямого распространения объединяются образы и прообразы каждого блока свертки в измерении каналов
```python
blk = DenseBlock(2, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
---------------------------
torch.Size([4, 23, 8, 8])
```
- количество выходных каналов: 23 = 3 + 10 + 10 - кол-во каналов блока свертки контролирует рост количества выходных каналов относительно количества входных каналов - *growth rate* (темп роста)

### Transition Layers
```python
def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```
- используется для контроллирования сложности модели
- Уменьшает: кол-во **каналов** за счет свертки $1\times 1$, **размерность** за счет average pooling (stride=2)
```python
blk = transition_block(10)
blk(Y).shape
---------------------------
torch.Size([4, 10, 4, 4])
```
- Итого: у transition_block задается выходное кол-во каналов, а размерность входа уменьшается вдвое.

## DenseNet Model
![[DenseNet 2.png]]
```python
class DenseNet(d2l.Classifier):
	def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4),
	             lr=0.1, num_classes=10):
	    super(DenseNet, self).__init__()
	    self.save_hyperparameters()
	    self.net = nn.Sequential(self.b1())
	    for i, num_convs in enumerate(arch):
	        self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
	                                                          growth_rate))
	        # The number of output channels in the previous dense block
	        num_channels += num_convs * growth_rate
	        # A transition layer that halves the number of channels is added
	        # between the dense blocks
	        if i != len(arch) - 1:
	            num_channels //= 2
	            self.net.add_module(f'tran_blk{i+1}', transition_block(
	                num_channels))
	    self.net.add_module('last', nn.Sequential(
	        nn.LazyBatchNorm2d(), nn.ReLU(),
	        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
	        nn.LazyLinear(num_classes)))
	    self.net.apply(d2l.init_cnn)

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

> !Довольно ресурсозатратна: по памяти и времени