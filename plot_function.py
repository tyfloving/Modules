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

# 2. 使用seabron进行corr相关系数的画图的步骤
import seaborn as sns
import matplotlib.pyplot as plt
def plot_heat(train):
  corr = train.corr()#计算各变量的相关性系数
  xticks = list(corr.index)#x轴标签
  yticks = list(corr.index)#y轴标签
  fig = plt.figure(figsize=(13,10))
  ax1 = fig.add_subplot(1, 1, 1)
  sns.heatmap(corr, annot=True, cmap="rainbow",ax=ax1,linewidths=5, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
  ax1.set_xticklabels(xticks, rotation=35, fontsize=15)
  ax1.set_yticklabels(yticks, rotation=0, fontsize=15)
  plt.show()
  
# 3. 使用pandas自带的画图函数进行优雅的画图的方法
import matplotlib.pyplot as plt
# 坐标轴上显示中文的设置方法
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ax1=df2[["好客户数","坏客户数"]].plot.bar(figsize=(10,5))
ax1.set_xticklabels(df2.index,rotation=15)
ax1.set_ylabel("客户数")
ax1.set_title("年龄与好坏客户数分布图")

# 4. matplotlib两种不同的方式的画图操作手段
4.1 面向对像的写法
    #!/usr/bin/python
    #coding: utf-8
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.arange(0, 100)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(x, x)
    ax2 = fig.add_subplot(222)
    ax2.plot(x, -x)
    ax3 = fig.add_subplot(223)
    ax3.plot(x, x ** 2)
    ax4 = fig.add_subplot(224)
    ax4.plot(x, np.log(x))
    plt.show()
4.2 pyplot的写法
    x = np.arange(0, 100)
    plt.subplot(221)
    plt.plot(x, x)
    plt.subplot(222)
    plt.plot(x, -x)
    plt.subplot(223)
    plt.plot(x, x ** 2)
    plt.subplot(224)
    plt.plot(x, np.log(x))
    plt.show()

  
  
