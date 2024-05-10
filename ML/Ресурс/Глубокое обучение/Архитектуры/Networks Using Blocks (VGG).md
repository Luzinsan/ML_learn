> VGG - Visual Geometry Group - Группа визуальной геометрии Оксфордского университета. 
> Они в своей сети VGG впервые применили идею использования повторяющихся структур - ***блоков*** (Simonyan and Zisserman, 2014).
> Такой подход доказал, что более глубокие и узкие сети более эффективны.
> Отличие от AlexNet - VGG net состоит из блоков слоёв, тогда как AlexNet спроектирован индивидуально
> ***Является чрезмерно ресурсозатратным***

VGG состоит из последовательности:
- свёрточных слоёв (kernel_size=$3\times 3$) с padding=1 для поддержания разрешения
- нелинейной функции активации ReLU
- Слоя pooling - MaxPooling (kernel_size=$2 \times 2$) с stride=2 для уменьшения разрешения (*downsampling*)

```python
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```
# VGG's architecture
![[VGG.png]]
Состоит из двух частей:
- Блок VGG: свёрточные слои с функцией активацией ReLU, завершающийся слоем pooling - оставляет размерность неизменной
- Полносвязные слои (как в [[AlexNet]])
```python
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
```
# Виды VGG
- Оригинальная архитектура: 5 сверточных блоков: первые 2 по 1 сверточному слою, следующие 3 по 2 сверточных слоя. Первый блок имеет 64 выходных канала, число каналов в последующих блоках увеличивается в 2 раза (до 512 в последнем блоке). Итого: 8 сверточных слоёв и 3 полносвязых -> VGG-11.
  `VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary((1, 1, 224, 224))
```plaintext
Sequential output shape:     torch.Size([1, 64, 112, 112])
Sequential output shape:     torch.Size([1, 128, 56, 56])
Sequential output shape:     torch.Size([1, 256, 28, 28])
Sequential output shape:     torch.Size([1, 512, 14, 14])
Sequential output shape:     torch.Size([1, 512, 7, 7])
Flatten output shape:        torch.Size([1, 25088])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 10])
```
- 