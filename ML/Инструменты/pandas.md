# Настройка 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html#pandas.set_option
- `pd.set_option("display.precision", 2)` - устрановить кол-во знаков после запятой
- `pd.set_option('display.max_columns', 100)` - настройка максимального количества отображаемых столбцов 
- `pd.set_option('display.max_rows', 100)` - настройка максимального количества отображаемых строк
- `pd.set_option('display.max_colwidth', 100)` - текст в ячейке отражается полностью вне зависимости от длины
- `df.T` - транспонирование фрейма данных

### Базовые методы и атрибуты для первичного изучения данных
- `df.shape` - кортеж `[(кол-во строк, кол-во столбцов)]`
- `df.columns` - объект `Index([названия столбцов])`
- `df.info()` - информация: тип объекта, `RangeIndex`: кол-во записей, индексы с 0 до n. Кол-во столбцов (всего m столбцов:
  \# - Колонка - Кол-во non-null значений - `dtype` тип данных столбца
  `Dtypes`: типы(кол-во столбцов этого типа данных)
  Занимаемая память.
- `df.describe()` - основные статистические характеристики
	- `include=['int64','float64']` или `include=np.number`:  кол-во непропущенных значений, среднее арифметическое, стандартное отклонение, минимальное значение, 25, 75 квартиль, 50% - медиана, максимальное значение
	- `include=['object','bool']`:  кол-во непропущенных значений, кол-во уникальных значений, мода, частота моды
	- Можно вызвать для конкретного столбца статистические характеристики методами:
		- `mix()` 
		- `max()`
		- `mean()`
		- `mode()`
		- `median()`
	- `percentiles=[]` - отобразить всё, кроме *25%* и *75%*
- `df['column'].value_counts()` -  таблица частот значений признака(ов) - позволяет определить баланс классов в признаке(ах)
	- `normalize=True` - отобразить в процентном соотношении 

### Методы индексации (сначала всегда идут строки)
- `df.loc[]` - индексация по `[имени/срезу строк, имени/срезу столбцов]`
- `df.iloc[]` - индексация по `[серийному номеру/срезу номеров строк, серийному номеру/срезу номеров столбцов]`
- `df[start_row:end_row]` - получить срез данных по строкам. Столбцы указать нельзя.
- `df[P(df['column'])]` - где `P` - это функция, возвращающая булевый список. Называется *Булевой индексацией* фрейма данных

### Преобразующие методы
- `df["target"].astype("int64")` - в задачах бинарной классификации, если целевая переменная true/false, её нужно конвертировать в домен 0/1
- `df['Column'].map({old_value: new_value})` - заменяет значения в указанном столбце с `old_value` на `new_value`. Если значение в словаре не указано, оно заменяется на **Nan**
  В случае с передачей словаря работает только с `Series`. 
  Если использовать функцию, можно применять и к `DataFrame`
- `df.replace({old_value: new_value})` - различие с `map()` - `replace()` ничего не делает со значениями, которых нет в словаре. 
  Ещё можно указывать так: `df.replace({'Column1': dict_replace})`
  ***Устареет в будущих версиях!***
  - `df.insert(loc=index_col, column="New_col", value=pd.Series)` - позволяет вставить новый столбец в указанное `index_col` место
  - `df.drop()` - удаление строк (`axis=0` по умолчанию) или столбцов
	  - `columns=[]` или `axis=1` - удаляет указанные столбцы
	  - `[row_names]` - удаляет указанные строки
- `pd.get_dummies(df)` - `one-hot-encoding` в pandas
	- `dummy_na=True` - позволяет воспринимать` pd.NaN` как отдельную категорию
- `df.fillna(value)` - замена значений `pd.NaN` на значение `value`
  
## Feature Extraction
- `df['New_Col'] = df['Col1'] + df['Col2']` - комбинация столбцов для получения нового признака
- `df['Column'].isin(['value1', 'value2', ...])` - возвращает Series булевых значений
- `df["Column"].str.startswith("Start")` - возвращает Series булевых значений

### Анализ
- `df.select_dtypes(include=np.number)` - получить фрейм данных с признаками конкретных типов
-  `df.sort_values(by="column", ascending=False)` - получить фрейм данных, отсортированный по указанной колонке(ам) (`ascending=False` - по убыванию)
	- `df.sort_values(by=['Churn','Total day charge'], ascending=[True, False])`
- `df['Column'].apply(lambda row: P(row))` - применить к каждой строке/столбцу функцию P, возвращает объект Series. по умолчанию axis=1 (перебирает по строкам)
- `df.groupby(by=grouping_columns)[columns_to_show].function()` - `grouping_columns` группируемые колонки, которые становатся новыми **индексами**.
	- `function()` -> `describe()`, `agg(['count','mean','median'])`

### Сводные таблицы
- `pd.crosstab(df["Column1"], df["Column2"])` - возвращает таблицу сопряженности (**contingency table**) - показывает многомерное распределение частот категориальных переменных в табличной форме. В частности, это позволяет нам увидеть распределение одной переменной в зависимости от другой, просматривая столбец или строку.
	- `normilize=True`
- `df.pivot_table()` - возвращает сводную таблицу
	- `values` – список колонок для расчета статистики
	- `index` – список колонок, по которым можно группировать данные
	- `aggfunc` — какую статистику нам нужно посчитать для групп: "count", "std", "mean", "min", "median", "max"
	- `margins=True` - отобращить общее кол-во
	- Пример: `data.pivot_table(values=['age'], index=['race','sex'], aggfunc=['mean','median','min','max'], margins=True)`

## Выведенные рецепты
- Сформировать pd.Series, где:
	- индексы - названия столбцов, 
	- значения - уникальные значения в столбцах
```python
data.select_dtypes("object").apply(lambda col: col.unique(), axis=0)
```
- `df['Column'].value_counts(normalize=True).plot(kind='bar')` - отображение частотного графика (для выявления сбалансированности классов)
```python
df_uniques = pd.melt(
    frame=df,
    value_vars=["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"],
)
df_uniques = (
    pd.DataFrame(df_uniques.groupby(["variable", "value"])["value"].count())
    .sort_index(level=[0, 1])
    .rename(columns={"value": "count"})
    .reset_index()
)

sns.catplot(
    x="variable", y="count", hue="value", data=df_uniques, kind="bar"
)
plt.xticks(rotation='vertical');
```
- График распределения значений категориальных признаков (value - значения в этих признаках)
## Визуальный анализ
- `series.plot()`
	- `kind=['bar']`