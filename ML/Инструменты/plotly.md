# Начальная настройка
```python
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
from IPython.display import display, IFrame

init_notebook_mode(connected=True)

def plotly_depict_figure_as_iframe(fig, title="", width=800, height=500,
  plot_path='../../_static/plotly_htmls/'):
  """
  This is a helper method to visualizae PLotly plots as Iframes in a Jupyter book.
  If you are running `jupyter-notebook`, you can just use iplot(fig).
  """

  # in a Jupyter Notebook, the following should work
  #iplot(fig, show_link=False)

  # in a Jupyter Book, we save a plot offline and then render it with IFrame
  fig_path_path = f"{plot_path}/{title}.html"
  plot(fig, filename=fig_path_path, show_link=False, auto_open=False);
  display(IFrame(fig_path_path, width=width, height=height))
```
# Express
# Graph Object
`fig = go.Figure()` - инициализация области отрисовки
`fig.add_trace(Graph_object)` - добавить объект в область
## Line
 `go.Scatter(x,y,name)` - формирует объект `Graph_object`, передаваемые значения x,y - массивы np.array или pd.Series. name указывает название линии (trace).

Документация по тому, как сделать subplots (множество графиков в одном figure): https://plotly.com/python/subplots/

## Рецепты
- Отображение `value_counts` столбца
```python
px.bar(df['sex'].value_counts().reset_index(), x='sex', y='count',
labels={'sex': 'Sex', 'count': 'Count'},
title='Sex Distribution', color='sex')
```
### Line plot
```python
years_df = (
    df.groupby("Year_of_Release")[["Global_Sales"]]
    .sum()
    .join(df.groupby("Year_of_Release")[["Name"]].count())
)
years_df.columns = ["Global_Sales", "Number_of_Games"]

# Create a line (trace) for the global sales
trace0 = go.Scatter(x=years_df.index, y=years_df["Global_Sales"], name="Global Sales")

# Create a line (trace) for the number of games released
trace1 = go.Scatter(
    x=years_df.index, y=years_df["Number_of_Games"], name="Number of games released"
)

# Define the data array
data = [trace0, trace1]

# Set the title
layout = {"title": "Statistics for video games"}

# Create a Figure and plot it
fig = go.Figure(data=data, layout=layout)

# in a Jupyter Notebook, the following should work
#iplot(fig, show_link=False)

# in a Jupyter Book, we save a plot offline and then render it with IFrame
plotly_depict_figure_as_iframe(fig, title="topic2_part2_plot1")
```

> Сохранить график в html
> 
> 
```python
# commented out as it produces a large in size file
#plotly.offline.plot(fig, filename="years_stats.html", show_link=False, auto_open=False);
```
### Bar chart

##### OpenML
```python
# Do calculations and prepare the dataset
platforms_df = (
    df.groupby("Platform")[["Global_Sales"]]
    .sum()
    .join(df.groupby("Platform")[["Name"]].count())
)
platforms_df.columns = ["Global_Sales", "Number_of_Games"]
platforms_df.sort_values("Global_Sales", ascending=False, inplace=True)

# Create a bar for the global sales
trace0 = go.Bar(
    x=platforms_df.index, y=platforms_df["Global_Sales"], name="Global Sales"
)

# Create a bar for the number of games released
trace1 = go.Bar(
    x=platforms_df.index,
    y=platforms_df["Number_of_Games"],
    name="Number of games released",
)

# Get together the data and style objects
data = [trace0, trace1]
layout = {"title": "Market share by gaming platform"}

# Create a `Figure` and plot it
fig = go.Figure(data=data, layout=layout)
# in a Jupyter Notebook, the following should work
#iplot(fig, show_link=False)

# in a Jupyter Book, we save a plot offline and then render it with IFrame
plotly_depict_figure_as_iframe(fig, title="topic2_part2_plot2")
```
##### Рецепты
```python
df_uniques = pd.melt(
    frame=df,
    value_vars=["gender", "cholesterol", "gluc", "smoke", "alco", "active"],
    id_vars=["cardio"],
)
df_uniques = (
    pd.DataFrame(df_uniques.groupby(["variable", "value", 
                                    #  "target" # Разбиваем по таргету
                                     ])["value"].count())
    .sort_index(level=[0, 1])
    .rename(columns={"value": "count"})
    .reset_index()
)

px.bar(df_uniques, x="variable", y="count", color="value", 
    #    facet_col="target"
       )
```
- Красивый вывод распределения уникальных значений, разбитый по целевой переменной (закомменчено)
### Box Plot
```python
data = []

# Create a box trace for each genre in our dataset
for genre in df.Genre.unique():
    data.append(go.Box(y=df[df.Genre == genre].Critic_Score, name=genre))

# Visualize
# in a Jupyter Notebook, the following should work
#iplot(data, show_link=False)

# in a Jupyter Book, we save a plot offline and then render it with IFrame
plotly_depict_figure_as_iframe(data, title="topic2_part2_plot3")
```