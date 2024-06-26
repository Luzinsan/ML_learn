> Выбор схемы инициализации влияет на поддержание числовой стабильности, а также связан с выбором нелинейной функции активации => всё это влияет на скорость сходимости алгоритма
> Неправильный выбор может привести к: взрывному или исчезающему градиенту во время тренировки.

> Пусть имеется глубокая сеть:
> - $L$ - кол-во слоёв
> - $\mathbf{x}$ - входной вектор
> - $\mathbf{o}$ - выходной вектор
> - $l$ - слой, определённый преобразованием $f_l$ параметризованным весами $\mathbf{W}^{(l)}$
> - $h^{(l)}$ - выход скрытого слоя ($h^{(0)}=\mathbf{x}$)
> > $$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ and thus } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$
> 
> Допустим, все входные и выходные данные - вектора. Тогда градиент вектора $\mathbf{o}$ можно записать относительно параметров $\mathbf{W}^{(l)}$ так: $$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$
> > Этот градиент является произведением $L-1$ матриц $\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$ и вектора градиента $\mathbf{v}^{(l)}$. 
> > Проблемы: проблема численного недобора (при перемножении большого кол-во вероятностей)(решают переходом на логарифмическую сетку, чтобы сместить давление с мантиссы на показатель степени); матрицы имеют большое кол-во собственных значений (маленьких или больших) и из произведение тоже может быть очень маленьким или очень большим.

# Vanishing Gradients - Исчезающие градиенты
Сигмоидная функция: [[Многослойный Перцептрон#Sigmoid Function]]
![[Vanishing Gradients.png]]
- Проблема: градиент сигмоиды исчезает, когда её входные данные слишком велики, или слишком малы; ситуация усугубляется, когда в модели много слоёв.
- [[Функции активации#ReLU|ReLU]] в свою очередь более стабильный (но менее точный) - считается выбором по-умолчанию

# Exploding Gradients - Взрывные градиенты
- Значения, расчитанные при комбинации матриц (на примере нормального распределения с дисперсией=1), получились либо очень большими, либо очень маленькими.
- Когда это происходит из-за инициализации глубокой сети, у нас нет шансов добиться сходимости оптимизатора градиентного спуска
```python
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
# a single matrix
#  tensor([[-0.8755, -1.2171,  1.3316,  0.1357],
#         [ 0.4399,  1.4073, -1.9131, -0.4608],
#         [-2.1420,  0.3643, -0.5267,  1.0277],
#         [-0.1734, -0.7549,  2.3024,  1.3085]])
# after multiplying 100 matrices
#  tensor([[-2.9185e+23,  1.3915e+25, -1.1865e+25,  1.4354e+24],
#         [ 4.9142e+23, -2.3430e+25,  1.9979e+25, -2.4169e+24],
#         [ 2.6578e+23, -1.2672e+25,  1.0805e+25, -1.3072e+24],
#         [-5.2223e+23,  2.4899e+25, -2.1231e+25,  2.5684e+24]])
```
# Нарушение симметрии
# Инициализация параметров
- Грамотный выбор инициализации решает вышеупомянутые проблемы
- + подходящая регуляризация - повышают стабильность
## Инициализация по-умолчанию
- Нормальное распределение
- Является методом инициализации по-умолчанию
- Хорош в задачах среднего размера

## Xavier init - Инициализация Ксавье
### Forward Propagation
- Пусть есть элемент выходного слоя: $$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$
- Все веса взяты из распределения (необязательно гауссова) с E=0 (средним=0) и дисперсией $\sigma^2$
- Все входные данные слоя имеют E=0 и дисперсию $\gamma^2$ 
- И пусть эти распределения независимы
- Тогда среднее выходного слоя: $$\begin{aligned} E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$
- И дисперсия выходного слоя: $$\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}$$
> Один из способов сохранить фиксированную дисперсию - установить $n_\textrm{in} \sigma^2 = 1$

### Backpropagation
- Градиенты распространяются от выходного слоя -> дисперсия градиентов может резко увеличиться только если $n_\textrm{out} \sigma^2 = 1$ 
- Возникает дилемма:  $n_\textrm{in} \sigma^2 = 1$ или  $n_\textrm{out} \sigma^2 = 1$. 
- Компромисс - пытаемся удовлетворить: $$\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}$$

### Основные моменты
- Инициализация Ксавье выбирает веса из:
	- распределения Гаусса: $E=0$, $Var = \sigma^2=\frac{2}{n_{in} + n_{out}}$
	- равномерного распределения: $U(-a,a)$, $Var=\frac{a^2}{3} =>^{\sigma^2}$: $$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$