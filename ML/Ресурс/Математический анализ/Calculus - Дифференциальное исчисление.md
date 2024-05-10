> Производная - это скорость изменения функции по отношению к изменениям её аргументов - *наклон касательной функции в каждой точке аргумента*.
> Для $f: \mathbb{R} \rightarrow \mathbb{R}$, производная от f в точке x определяется: $$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$
> Когда $f'(x)$ существует, говорят, что в $f$ дифференцируем при любом $x \in \mathbb{R}$.
> Но не всегда $f(x)$ дифференцируем (accuracy, AUC), поэтому дифференцируют его суррогат (похожую функцию)
> Нотации (выражения $y=f(x)$):$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$
## Правила композиции дифференцируемых функций:
$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{Constant multiple rule} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{Sum rule} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{Product rule} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{Quotient rule} \end{aligned}$$
# Частные производные и градиенты
> Пусть дана функция $y = f(x_1, x_2, \ldots, x_n)$ - функция с $n$ переменными. Частная производная функции $y$ по $x_i$ аргументу есть:$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$
> Однако всё, кроме $x_i$ можно рассматривать как константы, дифференцируя только по $x_i$. Условные обозначения в таком случае: $$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$
> Частные производные многомерной функции по всем переменным (направлениям) представляются в виде компонент вектора - ***градиента функции***.
> $$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots \partial_{x_n} f(\mathbf{x})\right]^\top.$$

> **Правила дифференциации многомерных функций:**
> 	Для всех $\mathbf{A} \in \mathbb{R}^{m \times n}$ выполняется $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ и $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} = \mathbf{A}$.
> 	Для квадратных матриц $\mathbf{A} \in \mathbb{R}^{n \times n}$ выполняется $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ и, в частности, $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.
> Аналогично, для любой матрицы $\mathbf{X}$, имеем $\nabla_{\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\mathbf{X}$.

### Chain Rule - Цепное правило
>Функции одной переменной: 
> 	Пусть $y=f(g(x))$ и вложенные функции $y=f(u)$, $u=g(x)$ - дифференцируемые функции. Тогда: $$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$
> Многомерные функции:
> 	Пусть $y=f(u)$, где $u=(u_1,.u_2,...,u_m)$ ($u=g(x)$) где каждый $u_i=g_i(x)$ (где $x=(x_1,x_2,...,x_n)$). Тогда: $$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \ \textrm{ and so } \ \nabla_{\mathbf{x}} y = \mathbf{A} \nabla_{\mathbf{u}} y,$$