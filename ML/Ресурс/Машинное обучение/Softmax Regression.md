> [[аффинное преобразование]]

# Классификация
> Категориальные признаки обязательно кодируются методом - one-hot encoding
## Linear Model
> У такой модели существует несколько выходных данных - аффинных функций - представляющие оценки условных вероятностей (по одной на класс)
> Пример: 4 признака, 3 класса => 12 значений коэффициентов весов, 3 скаляра-смещений:$$\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}$$

---
> Соответствующая схема нейронной сети - полносвязная однослойная нейронная сеть.
> ![[Softmax Regression.png]]
В обозначениях объектов линейной алгебры (без оптимизации): $$\mathbf{o}=\mathbf{W}\mathbf{x} + \mathbf{b},$$где $\mathbf{W}$ - матрица весовых коэффициентов (каждая строка матрицы - веса для выхода $\mathbf{o_i}$) ($W \in \mathcal{R}^{q\times d}$)
$\mathbf{x}$ - вектор признаков ($x \in \mathcal{R}^d$)
$\mathbf{b}$ - вектор коэффициентов смещений ($b \in \mathcal{R}^{q}$)
$q$ - кол-во классов
$d$ - кол-во признаков

# Softmax - функция активации
**Цель**: преобразовать данные выходного слоя сети в дискретные распределения вероятностей.
**Решение**: $P(y = i) \propto \exp o_i$ - сведение условной вероятности к экспоненциальной функции:
 ![[Экспоненциальная.png|300]]
> Обеспечивает: 
> - вероятность условного класса увеличивается с увеличением $\mathbf{o_i}$
> - является монотонной
> - все вероятности неотрицательны
> Требование: в сумме вероятности должны составлять $1$ => каждое значение делится на сумму всех вероятностей (нормализация)

**$Softmax$ функция**:$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \quad \textrm{where}\quad \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}.$$
> В экспоненциальной функции при очень больших значениях o может возникнуть переполнение.
> Решение: вычесть из текущих выходов $o_i$ и $o_j$ максимальное значение $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ $$\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} = \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.$$
> 

Т.о., наиболее вероятный класс $\hat{y}$ соответствует наибольшей координате вектора $\mathbf{o}$, т.к. операция $softmax$ сохраняет порядок своих аргументов: 
$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

# Оптимизация
## Векторизация - Minibatch SGD

> [[Linear Regression#Minibatch Stochastic Gradient Descent]]

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
где $\mathbf{X} \in \mathcal{R}^{n \times d}$
$\mathbf{W} \in \mathcal{R}^{d \times q}$
$\mathbf{b} \in \mathcal{R}^{1 \times q}$

# Loss Function - Cross-entropy loss
> Log-Likelihood - логарифмическое правдоподобие

Цель: сравнить оценки модели $\hat{y}$ с реальными метками $\mathbf{y}$: $P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).$
Допущение: поскольку каждая метка класса вычисляется независимо от соответствующего распределения, и максимизация произведения слагаемых затруднительна => вычисляется отрицательный логарифм - *задача минимизации отрицательного логарифма правдоподобия*: 
$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$
- В срезе одного объекта, loss функция $l$ (***Cross-entropy loss*** [[Энтропия]]): $$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
- Подставляя $softmax$ функцию в функцию потерь, получаем: 
$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j \\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

> > Учитывая возможность переполнения $o_i$ получим следующее: $$\log \hat{y}_j =\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).$$
>
> Рассматривая частную производную по выходу $\mathbf{o_i}$: 
  $$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$
> Т.о., приходим к тому, что градиент loss функции - это разница между вероятностями, назначенными нашей моделью (преобразованные с помощью $softmax$) и истинными метками классов (в виде one-hot encoding меток).

# Accuracy
> Accuracy (точность) - это доля правильных прогнозов.
> В задачах классификации она более показательна (но не дифференциируема)

**Вычисление:**
- Если $\hat{y}$ - это матрица, то предполагается, что второе измерение хранит оценки прогнозирования для каждого класса
- Используется $argmax$ для получения натболее вероятного класса в виде индекса
- Сравнивается предсказанный класс $\hat{y}$ с истинной меткой $y$ напрямую (оператором `==`)
- Оператор `==` чувствителен к типам данных, поэтому $\hat{y}$ преобразуется в `type(y)`
- Результат - тензор, содержащий записи 0($False$) и 1($True$)
- При необходимости оценка усредняется (mean)
# Using with framework
```python
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```
## Обучение
```python
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```