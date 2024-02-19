> Метод увеличения обучающей выборки, в котором выполняют серию случайных изменений обучающих *изображений*
```python
%matplotlib inline
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
```
---
- Вспомогательная функция для применения и отображения трансформаций:
```python
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

## Flipping and Cropping - Отражение и обрезка
- `torchvision.transforms.RandomHorizontalFlip()` - отображает изображения влево и вправо с вероятностью 50%
- `torchvision.transforms.RandomVerticalFlip()` - отображает изображения вверх и вниз с вероятностью 50%
- `torchvision.transforms.RandomResizedCrop(size, scale, ratio)` - обрезает случайную часть изображения и меняет её размер до заданного
	- `size` - размер выходного изображения
	- `scale` - нижняя и верхняя границы обрезаемом площади области в процентах
	- `ratio` - нижняя и верхняя границы случайного соотношения сторон кадра (crop) перед изменением размера
## Changing Colors - Изменение цветов
- `torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)` - случайным образом изменяет яркость, контрастность, насыщенность и оттенок изображения
	- `brightness, contrast, saturation` - насколько сильно колебать яркость/контраст/насыщенность изображения: `[max(0, 1-brightness), 1+brightness]`(у остальных аналогично)
	- `hue` - насколько сильно колебать оттенок изображения: `[-hue, hue]/[min,max]`. Значение hue должно быть в отрезке `[0,0.5]` или `-0.5 <= min <= max <= 0.5`.
## Комбинирование нескольких методов
- `torchvision.transforms.Compose([smt_aug1, smt_aug2, ...])` - комбинирование нескольких методов аугментации изображений