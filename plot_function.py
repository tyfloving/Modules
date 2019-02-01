import plotly.offline as offline
import plotly.plotly as py
import plotly.graph_objs as go
offline.init_notebook_mode(connected=False)

# 1. 使用最好看的画图的包(plotly)的步骤
def plotly_function(data, x='', y='', choose_clos=''):
  cols_sample = data.choose_clos.sample(1).values[0]
  print(f"the samples is:{cols_sample}!!!!!")
  temp = data[data.choose_clos == cols_sample].reset_index(drop=True)
  trace0 = go.Scatter(x=temp[x].values.tolist(),
                      y=temp[y].values.tolist(),
                      name='dispatch',
                      mode='lines+markers'
                     )
  layout = go.Layout(
                    xaxis=dict(title=x),
                    yaxis=dict(title=y)
                    )
  datas = go.Data([trace0])
  fig = go.Figure(data=datas, layout=layout)
  offline.iplot(fig, show_link=False)

# 2. 使用seabron进行画图的步骤
  
  
  
