#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import seaborn as sn
import plotly.offline as offline
offline.init_notebook_mode(connected=False)
from plotly import tools
import plotly.graph_objs as go
import os
from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)
CSS = """#notebook div.output_subarea {max-width:100%;}""" #changes output_subarea width to 100% (from 100% - 14ex)
HTML('<style>{}</style>'.format(CSS))


# In[2]:


df_predict = pd.read_excel('./sf预测数据.xlsx', dtype=({'future_date': 'object'}))
df_envent = pd.read_excel('./sbux-store-events_new.xlsx', dtype=({'event_date': 'object'}))


# In[3]:


df_predict = df_predict[['store_number', 'item_code', 'future_date', 'quantile_0.5', 'real', 'price', 'category']]
df_predict['quantile_0.5'] = abs(df_predict['quantile_0.5'].round(3))
df_predict['real'] = df_predict['real'].round(3)
# df_predict['quantile_0.5'] = df_predict['quantile_0.5'] * df_predict['price']
# df_predict['real'] = df_predict['real'] * df_predict['price']
df_predict['smape'] = abs(df_predict['quantile_0.5'] - df_predict.real) / (abs(df_predict.real) + abs(df_predict['quantile_0.5']))
df_predict.fillna(0, inplace=True)
df_predict.rename({'future_date':'date'}, axis=1, inplace=True)
df_predict['date'] = df_predict['date'].astype('str')


# In[4]:


df_envent['date'] = df_envent.apply(lambda x: x['order date'] if pd.isnull(x['event_date']) else x['event_date'], axis=1)
df_envent['flag'] = df_envent.apply(lambda x: 0 if pd.isnull(x['event_date']) else 1, axis=1)
df_envent = df_envent.drop_duplicates(['GlobalStoreNumber','date']).reset_index(drop=True)


# In[5]:


envent_result = df_envent[['GlobalStoreNumber', 'date', '数据来源', 'event', 'remark', 'flag', 'Store feedback']]
envent_result.rename({'GlobalStoreNumber':'store_number', 'Store feedback':'store_feedback'
                      }, axis=1, inplace=True)
envent_result['date'] = envent_result['date'].apply(lambda x: str(x)[:10])
tmp = df_envent[['GlobalStoreNumber','Store me']].drop_duplicates()
store_name_dict = dict(zip(tmp.GlobalStoreNumber.tolist(), tmp['Store me']))

predict_result = df_predict[df_predict.store_number.isin(envent_result.store_number.unique().tolist())].reset_index(drop=True)
plot_result = predict_result.merge(envent_result, on=['store_number', 'date'], how='left')
plot_result['store_name'] = plot_result.store_number.map(store_name_dict)


# # 门店事件对预测结果的影响分析
# 
# ## 数据解释
# - 800616: 全脂牛奶
# - 800639：脱脂牛奶
# - 900800343：豆奶
# - 11029543： 5磅咖啡豆-浓缩烘培
# - 11076980： 3磅咖啡豆-冷萃综合
# 
# ## 评价指标SMAPE
# <center>$SMAPE=\frac{(|y_i|-|y_j|)}{(|y_i|+|y_j|)}$

# In[16]:


def plot_store(df_test):
    store_name = df_test.store_name.unique()[0]
    store_id = df_test.store_number.unique()[0]
    print(f'门店:{store_name}: {store_id}')
    tmp1 = df_test.query('item_code == 800616').sort_values('date').reset_index(drop=True)
    tmp1_1 = tmp1[tmp1['flag'].notnull()].sort_values('date').reset_index(drop=True)

    tmp2 = df_test.query('item_code == 800639').sort_values('date').reset_index(drop=True)
    tmp2_1 = tmp2[tmp2['flag'].notnull()].sort_values('date').reset_index(drop=True)

    tmp3 = df_test.query('item_code == 900800343').sort_values('date').reset_index(drop=True)
    tmp3_1 = tmp3[tmp3['flag'].notnull()].sort_values('date').reset_index(drop=True)

    tmp4 = df_test.query('item_code == 11029543').sort_values('date').reset_index(drop=True)
    tmp4_1 = tmp4[tmp4['flag'].notnull()].sort_values('date').reset_index(drop=True)

    tmp5 = df_test.query('item_code == 11076980').sort_values('date').reset_index(drop=True)
    tmp5_1 = tmp5[tmp5['flag'].notnull()].sort_values('date').reset_index(drop=True)

    trace0 = go.Scatter(
                x = pd.to_datetime(tmp1['date'].tolist()),
                y = tmp1['smape'].tolist(),
                mode = 'lines+markers',
                name = 'smape',
                text = ['<b>数据来源:</b>'+ str(tmp1.loc[i, '数据来源']) + '<br>'
                        + '<b>event:</b>' + str(tmp1.loc[i, 'event']) + '<br>'
                        + '<b>remark:</b>' + str(tmp1.loc[i, 'remark']) + '<br>'
                        + '<b>smape:</b>' + str(round(tmp1.loc[i, 'smape'], 3)) + '<br>'
                        + '<b>store_back:</b>' + str(tmp1.loc[i, 'store_feedback'])
                        for i in range(tmp1.shape[0])],
                hoverinfo = 'text',
                line = dict(color='#85144B')
    )
    trace0_0 = go.Scatter(
                x = pd.to_datetime(tmp1_1['date'].tolist()),
                y = tmp1_1['smape'].tolist(),
                mode = 'markers',
                name = 'event',
                text = [None for i in range(tmp1_1.shape[0])],
                hoverinfo = 'text',
                marker = dict(
                        color = '#ff7500',
                        size = 13
                )
    )
    trace1 = go.Scatter(
                    x = pd.to_datetime(tmp1['date'].tolist()),
                    y = tmp1['quantile_0.5'].tolist(),
                    xaxis='x',
                    yaxis='y2',
                    mode = 'lines+markers',
                    name = 'predict',
                    line = dict(color='red', dash='dash')
        )
    trace2 = go.Scatter(
                x = pd.to_datetime(tmp1['date'].tolist()),
                y = tmp1['real'].tolist(),
                xaxis='x',
                yaxis='y2',
                mode = 'lines+markers',
                name = 'real',
                line = dict(color='blue')
    )

    trace3 = go.Scatter(
                x = pd.to_datetime(tmp2['date'].tolist()),
                y = tmp2['smape'].tolist(),
                mode = 'lines+markers',
                name = 'smape',
                text = ['<b>数据来源:</b>'+ str(tmp2.loc[i, '数据来源']) + '<br>'
                            + '<b>event:</b>' + str(tmp2.loc[i, 'event']) + '<br>'
                            + '<b>remark:</b>' + str(tmp2.loc[i, 'remark']) + '<br>'
                            + '<b>smape:</b>' + str(round(tmp2.loc[i, 'smape'], 3)) + '<br>'
                            + '<b>store_back:</b>' + str(tmp2.loc[i, 'store_feedback'])
                            for i in range(tmp2.shape[0])],
                hoverinfo = 'text',
                line = dict(color='#85144B')
    )
    trace3_0 = go.Scatter(
                x = pd.to_datetime(tmp2_1['date'].tolist()),
                y = tmp2_1['smape'].tolist(),
                mode = 'markers',
                name = 'event',
                text = [None for i in range(tmp2_1.shape[0])],
                hoverinfo = 'text',
                marker = dict(
                        color = '#ff7500',
                        size = 13
                )
    )
    trace4 = go.Scatter(
                    x = pd.to_datetime(tmp2['date'].tolist()),
                    y = tmp2['quantile_0.5'].tolist(),
                    xaxis='x',
                    yaxis='y2',
                    mode = 'lines+markers',
                    name = 'predict',
                    line = dict(color='red', dash='dash')
        )
    trace5 = go.Scatter(
                x = pd.to_datetime(tmp2['date'].tolist()),
                y = tmp2['real'].tolist(),
                xaxis='x',
                yaxis='y2',
                mode = 'lines+markers',
                name = 'real',
                line = dict(color='blue')
    )
    trace6 = go.Scatter(
                x = pd.to_datetime(tmp3['date'].tolist()),
                y = tmp3['smape'].tolist(),
                mode = 'lines+markers',
                name = 'smape',
                text = ['<b>数据来源:</b>'+ str(tmp3.loc[i, '数据来源']) + '<br>'
                                + '<b>event:</b>' + str(tmp3.loc[i, 'event']) + '<br>'
                                + '<b>remark:</b>' + str(tmp3.loc[i, 'remark']) + '<br>'
                                + '<b>smape:</b>' + str(round(tmp3.loc[i, 'smape'], 3)) + '<br>'
                                + '<b>store_back:</b>' + str(tmp3.loc[i, 'store_feedback'])
                                for i in range(tmp3.shape[0])],
                hoverinfo = 'text',
                line = dict(color='#85144B')
    )
    trace6_0 = go.Scatter(
                x = pd.to_datetime(tmp3_1['date'].tolist()),
                y = tmp3_1['smape'].tolist(),
                mode = 'markers',
                name = 'event',
                text = [None for i in range(tmp3_1.shape[0])],
                hoverinfo = 'text',
                marker = dict(
                        color = '#ff7500',
                        size = 13
                )
    )
    trace7 = go.Scatter(
                    x = pd.to_datetime(tmp3['date'].tolist()),
                    y = tmp3['quantile_0.5'].tolist(),
                    xaxis='x',
                    yaxis='y2',
                    mode = 'lines+markers',
                    name = 'predict',
                    line = dict(color='red', dash='dash')
        )
    trace8 = go.Scatter(
                x = pd.to_datetime(tmp3['date'].tolist()),
                y = tmp3['real'].tolist(),
                xaxis='x',
                yaxis='y2',
                mode = 'lines+markers',
                name = 'real',
                line = dict(color='blue')
    )
    trace9 = go.Scatter(
                x = pd.to_datetime(tmp4['date'].tolist()),
                y = tmp4['smape'].tolist(),
                mode = 'lines+markers',
                name = 'smape',
                text = ['<b>数据来源:</b>'+ str(tmp4.loc[i, '数据来源']) + '<br>'
                        + '<b>event:</b>' + str(tmp4.loc[i, 'event']) + '<br>'
                        + '<b>remark:</b>' + str(tmp4.loc[i, 'remark']) + '<br>'
                        + '<b>smape:</b>' + str(round(tmp4.loc[i, 'smape'], 3)) +'<br>'
                        + '<b>store_back:</b>' + str(tmp4.loc[i, 'store_feedback'])
                        for i in range(tmp4.shape[0])],
                hoverinfo = 'text',
                line = dict(color='#85144B')
    )
    trace9_0 = go.Scatter(
                x = pd.to_datetime(tmp4_1['date'].tolist()),
                y = tmp4_1['smape'].tolist(),
                mode = 'markers',
                name = 'event',
                text = [None for i in range(tmp4_1.shape[0])],
                hoverinfo = 'text',
                marker = dict(
                        color = '#ff7500',
                        size = 13
                )
    )
    trace10 = go.Scatter(
                    x = pd.to_datetime(tmp4['date'].tolist()),
                    y = tmp4['quantile_0.5'].tolist(),
                    xaxis='x',
                    yaxis='y2',
                    mode = 'lines+markers',
                    name = 'predict',
                    line = dict(color='red', dash='dash')
        )
    trace11 = go.Scatter(
                x = pd.to_datetime(tmp4['date'].tolist()),
                y = tmp4['real'].tolist(),
                xaxis='x',
                yaxis='y2',
                mode = 'lines+markers',
                name = 'real',
                line = dict(color='blue')
    )
    trace12 = go.Scatter(
                x = pd.to_datetime(tmp5['date'].tolist()),
                y = tmp5['smape'].tolist(),
                mode = 'lines+markers',
                name = 'smape',
                text = ['<b>数据来源:</b>'+ str(tmp5.loc[i, '数据来源']) + '<br>'
                        + '<b>event:</b>' + str(tmp5.loc[i, 'event']) + '<br>'
                        + '<b>remark:</b>' + str(tmp5.loc[i, 'remark']) + '<br>'
                        + '<b>smape:</b>' + str(round(tmp5.loc[i, 'smape'], 3)) + '<br>'
                        + '<b>store_back:</b>' + str(tmp5.loc[i, 'store_feedback'])
                        for i in range(tmp5.shape[0])],
                hoverinfo = 'text',
                line = dict(color='#85144B')
    )
    trace12_0 = go.Scatter(
                x = pd.to_datetime(tmp5_1['date'].tolist()),
                y = tmp5_1['smape'].tolist(),
                mode = 'markers',
                name = 'event',
                text = [None for i in range(tmp5_1.shape[0])],
                hoverinfo = 'text',
                marker = dict(
                        color = '#ff7500',
                        size = 13
                )
    )
    trace13 = go.Scatter(
                    x = pd.to_datetime(tmp5['date'].tolist()),
                    y = tmp5['quantile_0.5'].tolist(),
                    xaxis='x',
                    yaxis='y2',
                    mode = 'lines+markers',
                    name = 'predict',
                    line = dict(color='red', dash='dash')
        )
    trace14 = go.Scatter(
                x = pd.to_datetime(tmp5['date'].tolist()),
                y = tmp5['real'].tolist(),
                xaxis='x',
                yaxis='y2',
                mode = 'lines+markers',
                name = 'real',
                line = dict(color='blue')
    )


    data = [trace0, trace0_0, trace1, trace2, trace3,trace3_0, trace4, trace5, trace6,trace6_0, trace7, trace8, 
            trace9, trace9_0, trace10, trace11, trace12, trace12_0, trace13, trace14]


    updatemenus=list([
        dict(
            buttons=list([
                 dict(
                    args=[{'visible': [True,True, True,True,
                                       False, False, False, False,
                                       False, False, False, False,
                                       False, False, False, False,
                                       False, False, False, False]},
                          {'title': store_name + ': 全脂牛奶(800616)'}],
                    label='全脂牛奶',
                    method='update'
                ),

                dict(
                    args=[{'visible': [ False, False, False,False,
                                        True, True, True, True,
                                       False, False, False, False,
                                       False, False, False,False,
                                       False, False, False, False]},
                          {'title': store_name + ':脱脂牛奶(800639)'}],
                    label='脱脂牛奶',
                    method='update'
                ),
                dict(
                    args=[{'visible': [False, False, False, False,
                                       False, False, False, False,
                                       True, True, True, True,
                                       False, False, False, False,
                                       False, False, False, False]},
                          {'title': store_name + ':豆奶(900800343)'}],
                    label='豆奶',
                    method='update'
                ),
                dict(
                    args=[{'visible': [False, False, False, False,
                                       False, False, False, False,
                                       False, False, False, False,
                                       True, True, True, True,
                                       False, False, False, False]},
                          {'title': store_name + ':5磅咖啡豆-浓缩烘培(11029543)'}],
                    label='5磅咖啡豆-浓缩',
                    method='update'
                ),
                 dict(
                    args=[{'visible': [False, False, False,False,
                                       False, False, False,False,
                                       False, False, False,False,
                                       False, False, False,False,
                                       True, True, True, True]},
                          {'title': store_name + ':3磅咖啡豆-冷萃综合(11076980)'}],
                    label='3磅咖啡豆-冷萃',
                    method='update'
                ),
            ]),
            direction = 'down',
            pad = {'r': 0, 't': 0},
            showactive = True,
        ),
    ])

    layout = go.Layout(
    #     title='深圳前海湾店',
        dict(updatemenus=updatemenus),
    #     hiddenlabels = (1,2),
    #     hovermode='closest',
    #     hoverlabel=
    #             dict(
    #                     font=dict(size=20),namelength=-1),
    #     showlegend=True,
        xaxis=dict(
            title='日期',
        ),
        yaxis=dict(
            title='误差(SMAPE_%)',
            domain=[0, 0.45],
            autorange=False,
            range=[0, 1.2],
            type='linear'
        ),
        yaxis2=dict(
            title='业务量(L/kg)',
            domain=[0.55, 1]
        )
    )

    fig = go.FigureWidget(data=data, layout=layout)
    offline.iplot(fig)


# In[17]:


need_sku = [800616, 800639, 900800343, 11029543, 11076980]
for i in plot_result.store_number.unique().tolist():   
    df_tmp = plot_result[(plot_result.store_number == i) & (plot_result.item_code.isin(need_sku))].reset_index(drop=True)
    plot_store(df_test=df_tmp)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




