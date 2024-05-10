
<center><p>Первая современная CNN (Алекс Крижевский)</p>
<p>Показала хорошие результаты на соревновании ImageNet (2012)</p>
<p>Является эволюционным улучшением [[LeNet]]</p></center>

--- start-multi-column: ID_ac7q
```column-settings
Number of Columns: 2
Largest Column: standard
```
# Representation Learning
![[Feature Maps.png]]
- Использует функцию активации - ReLU
- Использует Dropout в качестве регуляризации
- Dence (полносвязные линейные слои) занимают до 1ГБ памяти
- Очень ресурсозатратна, хоть и не склонна к переобучению: последние 2 скрытых слоя - $6400 \times 4096$ и $4096 \times 4096$ => 164 МБ и 81 MFLOP 

--- column-break ---

# Design
![[AlexNet.png]]

--- end-multi-column
## Architecture's Code
```python
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
```

