> https://d2l.ai/chapter_convolutional-modern/googlenet.html
> https://arxiv.org/pdf/1409.4842

GoogLeNet (2014) - выиграл конкурс ImageNet Challenge
- Различают stem (основа, прием данных; 2/3 свертки, которые извлекают низкоуровневые признаки), body (обработка данных; набор сверточных блоков), head (прогнозирование; сопоставляются полученные к данному моменту признаки с требуемыми выходами согласно задаче)
- Основное отличие -  Inception Block ![[Inception.png]] 
	- состоит из 4-х параллельных ветвей
	- первые 3 ветви используют сверточные слои с ядрами $1\times 1$, $3\times 3$ и $5\times 5$ для извлечения информации из разных пространственных размеров
	- две средние ветви также добавляют $1\times 1$ свертку, чтобы уменьшить количество каналов и снизить сложность модели. 
	- четвертая ветвь использует $3\times 3$ max-pooling слой, за которым следует $1 \times 1$ сверточный слой, чтобы уменьшить количество каналов (привести к единообразию)
```python
class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
```
![[GoogLeNet.png|800]]
- GoogLeNet использует стек из 9 Inception блоков, сгруппированных в три группы с max-pooling слоями (уменьшает размерность) между ними и average-pooling слоем в конце для генерации оценок.
```python
class GoogleNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
	def b2(self):
	    return nn.Sequential(
	        nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
	        nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
	        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
	
	def b3(self):
	    return nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
	                         Inception(128, (128, 192), (32, 96), 64),
	                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
	def b4(self):
	    return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
	                         Inception(160, (112, 224), (24, 64), 64),
	                         Inception(128, (128, 256), (24, 64), 64),
	                         Inception(112, (144, 288), (32, 64), 64),
	                         Inception(256, (160, 320), (32, 128), 128),
	                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
	def b5(self):
	    return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
	                         Inception(384, (192, 384), (48, 128), 128),
	                         nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
	                     
	def __init__(self, lr=0.1, num_classes=10):
	    super(GoogleNet, self).__init__()
	    self.save_hyperparameters()
	    self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
	                             self.b5(), nn.LazyLinear(num_classes))
	    self.net.apply(d2l.init_cnn)
```