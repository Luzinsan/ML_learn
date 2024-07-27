> Имеет значительно меньше параметров, чем [[AlexNet]] и [[Networks Using Blocks (VGG)]]

Принцип: использование
1. $1 \times 1$ свёрток для добавления локальных нелинейностей (аналог линейного слоя, только с сохранением пространственной структуры)
2. global Average Pooling для объединения по всем элементам в последнем слое представления
---
NiN использует те же начальные размеры свертки, что и [[AlexNet]] (он был предложен вскоре после этого). Размеры ядра равны $11\times 11, 5 \times 5, 3 \times 3$, соответственно, а количество выходных каналов соответствует количеству выходных каналов *AlexNet*. 
За каждым блоком NiN следует слой с $MaxPooling$ с `stride=2`, kernel_size=($3\times 2$).

> NiN полностью избегает полносвязных слоёв.
> Вместо этого NiN использует NiN-блоки с числом выходных каналов в последнем слое =кол-ву классов, после которого следует GlobalAveragePooling (возвращает вектор logits)
# Architecture
![[NiN net.png]]
```python
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())

```

```python
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
        self.net.apply(d2l.init_cnn)
```