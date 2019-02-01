import plotly.offline as offline
import plotly.plotly as py
import plotly.graph_objs as go
offline.init_notebook_mode(connected=False)

# 1. 使用最好看的画图的包的步骤

sampe = result.zone_sampe.sample(1).values[0]
print(f"the zonecode is:{sampe}!!!")
result_test = result[result.zone_sampe == sampe]
# result_test

trace0 = go.Scatter(x=result_test.columns.tolist()[1:-1],
                    y=result_test.iloc[0,1:-1].values,
                    name='dispatch',
                    mode='lines+markers'
)



layout = go.Layout(
                    xaxis=dict(title='日期'),
                    yaxis=dict(title='派件时长')
)
data = go.Data([trace0])
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, show_link=False)
