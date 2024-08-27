> Предположения линейной регрессии:
> 1. Связь между features (covariates) $\mathbf{x}$ и labels (targets) $y$ приблизительно линейная => условное среднее $E[Y \mid X=\mathbf{x}]$ может выражено как взвешенная сумма признаков $\mathbf{x}$, целевое значение y при этом может отклоняться за счёт шума (распределения Гаусса) - сопоставление производится как ***[[аффинное преобразование]]***

# Model
$$\hat{y} = w_1 x_1 + \cdots + w_d x_d + b.$$
> где $w_i$ - веса - определяют влияние каждого признака на предсказание (prediction)
> $b$ - смещение ($bias$, offset, intercept) - значение оценки, когда все объекты равны нулю
> 	Модель представляет собой [[аффинное преобразование]] входных признаков, которое характеризуется линейным преобразованием признаков через взвешенную сумму в сочетании с переносом через добавленное смещение
> Компактное представление для одного наблюдения: $$\hat{\mathbf{y}} = \mathbf{w}^T \mathbf{x} + b$$где $\mathbf{w} = \{w_1, ..., w_d\} \in \mathbb{R}^d$ 
>  $\mathbf{x} = \{x_1,...,x_d\} \in \mathbb{R}^d$ 
>  $\mathbf{w}^T \mathbf{x}$ - скалярное произведение (dot product [[Скалярное произведение и двойственность]])
>  ---
>  Распространяя на все объекты получаем матрично-векторное произведение:$$\hat{y}=\mathbf{X} \mathbf{w} + b$$
>  где $\mathbf{X} \in \mathbb{R}^{n\times d}$ - *design matrix* - (каждая строка - наблюдения, каждый столбец - признак)
>  $\mathbf{w} = \{w_1, ..., w_d\} \in \mathbb{R}^d$
>  $b \in \mathbb{R}$ - применяется [[torch#Broadcasting|broadcasting]] чтобы расширить до размера результата $\mathbf{X} \mathbf{w}$ 
>  $\hat{y} \in \mathbb{R}^n$ 

> Т.о. цель линейной регрессии - найти вектор весов $\mathbf{w}$ и смещение $b$ такие, чтобы при заданных характеристиках новых примеров данных $\mathbf{X}$ (взятых их того же распределения, что и при обучении), предсказания отличались от фактических значений $\hat{y}$ с наименьшей ошибкой.

# Поиск параметров модели - Fitting Model
## Функция потерь - Loss Function
> Loss функция:
> - Количественно определяет расстояние между реальными данными и прогнозируемыми значениями
> - Представляет собой неотрицательное число, где "чем меньше, тем лучше"
> - Для задач регрессии наиболее распространённая - среднеквадратичная ошибка (MSE)

### Среднеквадратичная ошибка
> Рассматривая одно наблюдение:$$l^{(i)}(\mathbf{w},b) = \frac{1}{2}(\hat{y}^{i} - y^{(i)})^2$$
> $\frac{1}{2}$ - константа уходит при дифференцировании
> Fitting (аппроксимация) модели линейной регрессии к одномерным данным:![[Linear Regression - Loss function.png]]
>  ! Большая разница между $\hat{y}^{(i)}$ и $y^{(i)}$ приводит к большему значению loss функции (из-за квадратичной формы функции) => модель стремиться избегать больших ошибок => модель чувствительна к выбросам.
>  ---
>  Распространяя на n наблюдений, оценка усредняется:$$L(\mathbf{w},b)=\frac{1}{n}\sum^{n}_{i=1}l^{(i)}(\mathbf{w},b) = \frac{1}{n}\sum^{n}_{i=1}\frac{1}{2}(\mathbf{w}^T\mathbf{x}^{(i)}+b-y^{(i)})^2$$
>  При обучении мы ищем параметры:$$\mathbf{w}^*, b^* = {argmin}_{\mathbf{w},b}L(\mathbf{w},b)$$
>  

# Аналитическое решение
[[СЛАУ]]
> Задача линейной регрессии - задача оптимизации.
> 	Столбец смещения b включается в столбец весов $\mathbf{w}$  путём дополнения столбца к матрице $\mathbf{X}$ состоящего из единиц
> Минимизация: $$||\mathbf{y} - \mathbf{X}\mathbf{w}||^2$$Нахождение минимума - есть нахождение градиента loss функции:
> $$\begin{aligned}\partial_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 = 2 \mathbf{X}^\top (\mathbf{X} \mathbf{w} - \mathbf{y}) = 0\textrm{ and hence }\mathbf{X}^\top \mathbf{y} = \mathbf{X}^\top \mathbf{X} \mathbf{w}.\end{aligned}$$Выполнима, если $\mathbf{X}$ имеет полный ранг ([[Пространство столбцов и ранг]])(все признаки линейно-независимы) => т.е. $\mathbf{X}$ - обратимая матрица => поверхность функции потерь имеет единственную критическую точку - минимум loss функции.

# Оптимизация
## Minibatch Stochastic Gradient Descent
> Итеративное уменьшение ошибки путём обновления параметров в направлении, которое постепенно уменьшает loss функцию.
> 1. Если брать производную loss функции, вычисленную для каждого отдельного примера, то нужно будет пройти весь набор данных, прежде чем сделать одно обновление. Даже если шаг будет очень большим, вычисления будут очень медленными. *(по одному)*
> 2. Процессоры намного быстрее умножают и складывают числа, чем перемещают данные из основной памяти в кэш процессора => эффективнее выполнить матрично-векторное произведение, чем соответствующие векторные произведения. Но для большого объёма данных это затратно *(весь пакет)*
> Решение: промежуточная стратегия - обработка мини-партий (*minibatch* - гиперпараметр $\eta$)
> ---
> Параметры выбора размера пакета зависит от: объёма памяти, кол-ва ускорителей, выбора слоёв, размера набора данных.
> Число от 32 до 256.
> ***Число кратное степени 2***
> ---
> Процесс обновления весов:
> 1. Инициализируются параметры модели - случайным образом 
> 2. Итеративно выбираются случайные minibatch'и
> 3. Параметры обновляются в направлении отрицательного градиента$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$
> 	Пояснение: на каждой итерации $t$ случайно выбирается minibatch $\mathcal{B}_t$. Вычисляется градиент loss функции на minibatch по параметрам модели $\mathbf{w}$. Градиент умножается на скорость обучения learning rate $\eta$ (гиперпараметр). Полученный вектор вычитается из текущего вектора весов $\mathbf{w}$ (рассматривается градиент в обратном направлении).
> Для квадратичных потерь и аффинных преобразований имеется разложение в замкнутой форме:$$\begin{aligned} \mathbf{w} & \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) && = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)\\ b &\leftarrow b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_b l^{(i)}(\mathbf{w}, b) && = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$

Гиперпараметр - параметр, определённый пользователем, который не меняется в цикле обучения.
> может быть настроен автоматически с помощью методов - [[Байесовская оптимизация]] 

Качество решения оценивается на validation set (проверочном наборе данных).

# Linear Regression as a Neural Network
![[Linear Regression.png]]
Входные данные сети: x_1, ..., x_d - где d - размерность признакового пространства во сходном слое.
Выход сети o_1.
Все входные значения заданы и есть только 1 вычисляемый нейрон => линейную регрессию можно рассматривать как однослойную полносвязную нейронную сеть (single-layer fully connected neural network)

# Использование LazyLinear
```python
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
    
    def forward(self, X):
	    return self.net(X)

	def loss(self, y_hat, y):
	    fn = nn.MSELoss()
	    return fn(y_hat, y)

	def configure_optimizers(self):
	    return torch.optim.SGD(self.parameters(), self.lr)

	def get_w_b(self):
	    return (self.net.weight.data, self.net.bias.data)
```
## Обучение
```python
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)


w, b = model.get_w_b()

print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
# error in estimating w: tensor([ 0.0094, -0.0030])
# error in estimating b: tensor([0.0137])
```

## Weight Decay
- Настраиваем регуляризацию только для весов
```python
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```
- Использование:
```python
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```