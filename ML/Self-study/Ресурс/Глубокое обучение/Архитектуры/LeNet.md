> Одна из первых опубликованных CNN. Решала задачу распознавания рукописных цифр.
> Модель предложена Yann LeCun (Ян Лекун, и названа в честь него)

# Архитектура
![[LeNet.png]]
- Convolutional encoder: 2 свёрточных слоя (kernel=$5\times 5$, padding=2, f=sigmoid activation), 2 average pooling:
	- 1 conv layer - 6 выходных каналов (feature maps)
	- 2 conv layer - 16 feature maps
	- polling layers: kernel=$2\times 2$, stride=2, уменьшает размерность в 4 раза
	- Shape выходных данные этой части: (batch_size, num_channels, height, width) - должны быть преобразованы (flatten) в shape=(batch_size, flatten_vector)
- Dence block: 3 полносвязных слоя:
	- 120, 84, 10 num_outputs (10 - кол-во предсказываемых классов)
- Слой активации: Гауссов activation layer (но в реализации d2l - softmax)
- Loss функция: Cross-Entropy Loss
## Сжатое обозначение для LeNet
![[LeNet simple.png]]

### Пример реализации
```python
def init_cnn(module):  #@save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes))
            
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = torch.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 1, 28, 28))

#Conv2d output shape:         torch.Size([1, 6, 28, 28])
#Sigmoid output shape:        torch.Size([1, 6, 28, 28])
#AvgPool2d output shape:      torch.Size([1, 6, 14, 14])
#Conv2d output shape:         torch.Size([1, 16, 10, 10])
#Sigmoid output shape:        torch.Size([1, 16, 10, 10])
#AvgPool2d output shape:      torch.Size([1, 16, 5, 5])
#Flatten output shape:        torch.Size([1, 400])
#Linear output shape:         torch.Size([1, 120])
#Sigmoid output shape:        torch.Size([1, 120])
#Linear output shape:         torch.Size([1, 84])
#Sigmoid output shape:        torch.Size([1, 84])
#Linear output shape:         torch.Size([1, 10])
```

```python
class ParamLeNet(d2l.Classifier):
    def __init__(self, convs, linears, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        for conv in convs:
            layers.append(nn.LazyConv2d(conv[0], kernel_size=conv[1],
                                        padding=conv[2]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Flatten())
        for linear in linears:
            layers.append(nn.LazyLinear(linear))
            layers.append(nn.ReLU())
        layers.append(nn.LazyLinear(num_classes))
        self.net = nn.Sequential(*layers)

# Adjust the convolution window size (kernel_size)
data = d2l.FashionMNIST(batch_size=256)
convs_list = [[[6,i,(i-1)//2],[16,i,0]] for i in [1,2,3,4,5,6,7,8,9,10]]
# convs_list = [[[6,11,5],[16,11,0]],[[6,5,2],[16,5,0]],[[6,3,1],[16,3,0]]]
acc_list = []
for convs in convs_list:
    hparams = {'convs':convs, 'linears':[120, 84]}
    model = ParamLeNet(**hparams)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
    y_hat = model(data.val.data.type(torch.float32).unsqueeze(dim=1))
    acc_list.append(model.accuracy(y_hat,data.val.targets).item())


d2l.plot(list(range(len(acc_list))),acc_list,'conv window','acc')


pic = data.val.data[:2,:].type(torch.float32).unsqueeze(dim=1)
d2l.show_images(pic.squeeze(),1,2)


d2l.show_images(model.net[:2](pic).squeeze().detach().numpy().reshape(-1,28,28),4,8)
```