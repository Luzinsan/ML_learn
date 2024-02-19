### Инструменты
[[plotly]]
[[seaborn]]
# Стадии визульного анализа данных
## Одномерная визуализация (Uni-variate visualization)
> Рассматривает один признак за раз. Интересует распределение значений.
### Количественные признаки (Quantitative features)
#### Гистограммы и графики плотности (Histograms and density plots)
- Встроенными средствами pandas - группировка значений в ячейки одинакового диапазона (гистограммы) -> даёт информацию о типе распределения признака: гауссов (нормальное), экспоненциальное и т.д. -> различные методы ML предполагают определённый тип растределения
```python
features = ["Total day minutes", "Total intl calls"]
df[features].hist(figsize=(10, 4))
```
- Графики плотности (_Kernel Density Plots_ - Density) - сглаженная версия гистограмм, но независимая от частоты разбиения (bins):
```python
df[features].plot(
    kind="density", subplots=True, 
    layout=(1, 2), sharex=False, figsize=(10, 4)
)```
- График распределения наблюдения + график плотности (density): высота столбцов нормирована и показывает плотность, а не кол-во примеров в каждом интервале
```python
sns.histplot(df["Total intl calls"], stat='density', 
			 kde=True, linewidth=0, bins=20)
```
#### Box Plot
```python
sns.boxplot(x="Total intl calls", data=df)
```
- Коробка показывает межквартильный размах $(IQR=Q3-Q1)$: 25 $(Q1)$ и 75 $(Q3)$ процентили, линия внутри коробки - медиана (50%)
- Отрезки по бокам коробки (усы): разброс точек данных в интервале $(Q1 - 1.5*IQR, Q3 + 1.5*IQR)$
- Точки - выбросы (*outliers*)
#### Violin Plot
```python
_, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.boxplot(data=df["Total intl calls"], ax=axes[0])
sns.violinplot(data=df["Total intl calls"], ax=axes[1])
```
- Box Plot показывает определенную статистику, касающуюся отдельных примеров в наборе данных, тогда как "скат" концентрируется на сглаженном распределении информации в целом
### Категориальные и бинарные признаки (Categorical and binary features)
> Если значения категориального (или количественного) признака упорядочены -> порядковый признак (_ordinal_)
#### Bar Blot
> Графическое представление таблицы частот

 - `df['Column'].value_counts(normalize=True).plot(kind='bar')` - встроенными средствами pandas
 - С помощью seaborn (не путать с barplot - используется для представления статистики числовой переменной, **сгруппированной** по категориальному признаку):
 ```python
 _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

sns.countplot(x="Churn", data=df, ax=axes[0])
sns.countplot(x="Customer service calls", data=df, ax=axes[1]);
```
---
## Многомерная визуализация (Многомерная визуализация)
> Позволяют увидеть взаимосвязь между двумя и более различными переменными на одном рисунке

### Quantitative vs. Quantitative (Количественный с количественным)
#### Correlation matrix (Корреляционная матрица)
```python
corr_matrix = df[numerical].corr()
sns.heatmap(corr_matrix);
```
- `corr()` вычисляет корреляции между каждой парой ***числовых*** признаков.
- `heatmap()` отображает матрицу с цветовой кодировкой для предоставленныз значений
- После изучения матрицы корреляций следует избавиться от зависимых признаков
#### Scatter plot (Диаграмма рассеяния, Точечная диаграмма)
```python
plt.scatter(df["Total day minutes"], df["Total night minutes"])
```
- с помощью matplotlib
- отображает значения 2х числовых признака в виде координат, отображая точку в двумерном пространстве 
- если результат похож на эллипс - то переменные не коррелируют
```python
sns.jointplot(x="Total day minutes", y="Total night minutes", 
			  data=df, kind="scatter")
```
- с помощью seaborn
- `joinplot()` строит 2 гистограммы
- `kind='kde' `- позволяет получить сглаженную версию распределения
#### Scatterplot matrix (Матрица рассеяния)
```python
%config InlineBackend.figure_format = 'png'
sns.pairplot(df[numerical])
```
- Ее диагональ содержит распределения соответствующих переменных, а диаграммы рассеяния для каждой пары переменных заполняют остальную часть матрицы.
- `%config InlineBackend.figure_format = 'png'` - полезно, если числовых признаков много
### Quantitative vs. Categorical
```python
sns.lmplot(
    x="Total day minutes", y="Total night minutes", data=df, hue="Churn", fit_reg=False
)
```
- диаграмма рассеяния, в которой точки окрашиваются в разные цвета, представляя собой разные значения категориальной переменной
```python
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 7))
for idx, feat in enumerate(numerical):
	ax = axes[idx // 4, idx % 4]
	sns.boxplot(x="Churn", y=feat, data=df, ax=ax)
	ax.set_xlabel("")
	ax.set_ylabel(feat)

fig.tight_layout()
```

```python
_, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

sns.boxplot(x="Churn", y="Total day minutes", data=df, ax=axes[0])
sns.violinplot(x="Churn", y="Total day minutes", data=df, ax=axes[1]);
```
- Box Plots, разбитые по категориальному признаку
- Здесь можно увидеть расхождения распределений числовых признаков в зависимости от целевого признака
```python
sns.catplot(
    x="Churn",
    y="Total day minutes",
    col="Customer service calls",
    data=df[df["Customer service calls"] < 8],
    kind="box",
    col_wrap=4,
    height=3,
    aspect=0.8,
);
```
- Проанализировать количественный признак сразу по двум категориальным переменным
```python
platform_genre_sales = (
    df.pivot_table(
        index="Platform", columns="Genre", values="Global_Sales",
        aggfunc=sum).fillna(0).applymap(float)
)
sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=0.5)
```
- Тепловая карта также позволяет проанализировать распределение числового признака по двух категориальным признакам
### Categorical vs. Categorical
```python
sns.countplot(x="Customer service calls", hue="Churn", data=df);
```
- Частотный график, разбитый по категориальному признаку -> параметр hue


> Можно попробовать использовать crosstab(), pivot_table()

# Визуализация всего набора данных
## t-SNE
t-SNE - это нелинейный метод (многообразоное обучение - Manifold Learning), предназначенный для уменьшения размерности данных, для последующего анализа данных, но с уже меньшей размерностью.
- t-SNE - _t-distributed Stochastic Neighbor Embedding_ - t-распределенное встраивание стохастических соседей
- Основная идея: найти проекцию многомерного пространства признаков на плоскость (или трехмерную гиперплоскость, но она почти всегда двумерная) такую, что те точки, которые были далеко друг от друга в исходном n-мерном пространстве, в конечном итоге окажутся далеко друг от друга на плоскости. Те, кто изначально были близки, останутся близкими друг другу.
```python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
```
*смотри пример на https://mlcourse.ai/book/topic02/topic02_visual_data_analysis.html#t-sne или в topic_2.ipynb*
- Большие вычислительные затраты для больших датасетов (что чаще всего встречается) -> [Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
