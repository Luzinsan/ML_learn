> Forward Propagation и Backpropagation - это то, что происходит под капотом обучения (подбора) нейронной сети. 
# Forward Propagation (Forward Pass)
> Forward Propagation - это вычисление и хранение промежуточных переменных (включая выходной слой) для нейронной сети в порядке от входного слоя к выходному.

На примере [[Многослойный Перцептрон]]: 
*Computation graph*:
![[Computational Graph.png]]
1. $\mathbf{x} \in \mathcal{R}^d$, $\mathbf{W}^{(1)} \in \mathcal{R}^{h \times d}$
2. $\mathbf{z} = \mathbf{W}^{(1)}\mathbf{x}$, $\mathbf{z} \in \mathcal{R}^h$
3. $\phi$ - функция активации
4. $\mathbf{h} = \phi(\mathbf{z})$ - скрытый слой после активации (промежуточная переменная)
5. $\mathbf{W}^{(2)} \in \mathcal{R}^{q \times h}$ 
6. $\mathbf{o} = \mathbf{W}^{(2)}\mathbf{h}$, $\mathbf{o} \in \mathcal{R}^{q}$  - выходной слой
7. $\mathcal{l}$ - функция потерь, $y$ - истинная метка
8. $L = \mathcal{l}(\mathbf{o}, y)$ - значение функции потерь для одного наблюдения
9. $s = \frac{\lambda}{2}\left(||\mathbf{W}^{(1)}||^2_F + ||\mathbf{W}^{(2)}||^2_F\right)$ - значение регуляризации ([[Нормы#Норма Фробениуса]])
10. $J = L + s$ - значение **целевой функции** (регуляризованная *loss* функция)
# Backward Propagation (Backpropagation)
> Сводится к расчёту градиента параметров сети.
> Метод обходит всю сеть в обратном порядке, от выходного слоя к входному, в соответствии с [[Calculus - Дифференциальное исчисление#Chain Rule - Цепное правило|цепным правилом]] и сохраняет любые промежуточные переменные ([[Calculus - Дифференциальное исчисление#Частные производные и градиенты|частные производные]]), необходимые при расчёте градиента по некоторым параметрам.

> Пусть есть функции $\mathsf{Y}=f(\mathsf{X})$ и $\mathsf{Z}=g(\mathsf{Y})$ тогда производная $\mathsf{Z}$ по отношению к $\mathsf{X}$:
> $$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \textrm{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$
> Цель backpropagation для MLP - это вычисление градиентов $\frac{\partial J}{\partial \mathbf{W}^{(1)}}$ и $\frac{\partial J}{\partial \mathbf{W}^{(2)}}$. Для этого применяется цепное правило и по-очереди вычисляется градиент по каждой переменной. При этом порядок вычислений *обратный*.

1. Градиент целевой функции $J$ относительно значения loss функции $L$ и значения регуляризации $s$:
   $$\frac{\partial J}{\partial L} = 1 \; \textrm{and} \; \frac{\partial J}{\partial s} = 1.$$
2. Градиент целевой функции $J$ относительно переменной выходного слоя $\mathbf{o}$ (с цепным правилом):
   $$\frac{\partial J}{\partial \mathbf{o}}
= \textrm{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.$$
3. Градиенты члена регуляризации $s$ по обоим параметрам ($\mathbf{W}^{(1)}$ и $\mathbf{W}^{(2)}$): 
   $$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)} \; \textrm{and} \; \frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$
4. Градиент целевой функции $J$ относительно параметром модели, ближайших к выходному слою: ($\mathbf{W}^{(2)}$): $$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
5. Градиент целевой функции $J$ в сторону $\mathbf{W}^{(1)}$ производится вдоль выходного слоя $\mathbf{o}$ до скрытого слоя $\mathbf{h}$ 
   $$\frac{\partial J}{\partial \mathbf{h}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.$$
6. Функция активации $\phi$ применяется поэлементно => вычисление градиента целеввой переменной $J$ по слою $\mathbf{z}$ требует использования поэлементного оператора умножения $\odot$: $$\frac{\partial J}{\partial \mathbf{z}}= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).$$
7. Градиент целевой функции $J$ по параметрам $\mathbf{W}^{(1)}$:$$\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.$$

### Заметки
- После инициализации параметров модели чередуется *forward propagation* и *backpropagation*, обновляя параметры модели с использованием градиентов, заданных *backpropagation*
- *Backpropagation* повторно использует сохранённые промежуточные значения *forward propagation*, чтобы избежать дублирования вычислений
- => поэтому обучение требует больше памяти: размер промежуточных значений пропорционален кол-ву своёв, кол-ву нейронов, размеру пакета => обучение более глубоких сетей с использованием пакетов большего размера легче приводит к ошибкам нехватки памяти.