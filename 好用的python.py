import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')
import plotly.offline as offline
import plotly.plotly as py
import plotly.graph_objs as go
%matplotlib inline
offline.init_notebook_mode(connected=True)

# 2.  pandas中读入中文名字的文件的方法
pd.read_csv('filename_path', encoding='utf-8', engine='python')
如果文件刚开始不是utf-8编码方式，就需要将文件通过norepadd++编辑器将其打开，  然后改为以utf-8的编码方式即可。

# 3. 需要对pandas中的Dataframe数据要一行一行的转为字典的好的写法：
for (index,row) in data.iterrows():
    if row[0] not in order_dict:
        order_dict[row[0]]=[]
    order_dict[row[0]].append((row[3],row[4]))

or others ways
result = {}
for i in data.columns:
    result[i] = zip(data['a'], data['b'])


# 4.  将pandas中的数据转换为dict的好的方法
step1: 将Dataframe中要设置为字典中的key的列设置为索引，可以将多列设置为索引     
arc_cost_mean.set_index(['start','end'])，
 将dict中的值变为一个values，一列
step2: 然后将设置完索引后的结果进行转置接着再进行to_dict()操作
arc_cost_mean.set_index(['start','end']).T.to_dict(）
如果要对转换后的字典进行还原的话就可以使用这个方法：				    
pd.DataFrame(arc_cost_dict).T.reset_index()

# 直接list的写法
[i if len(i) > 0 else i-1 for i in data.columns]

# 改变index_name的方法
cols = df.index.values
df = df.T
df.columns = cols

#  list的直接相加的方法
['a'] + ['b'] + ['c']

# 当对pandas进行操作是
index的索引号的大小


# parallel processing 
def powers(x):
return x**2, x**3, x**4
This function can be called on any iterable, for example, a list.
>>> y = [1.0, 2.0, 3.0, 4.0]
>>> list(map(powers, y))
[(1.0, 1.0, 1.0), (4.0, 8.0, 16.0), (9.0, 27.0, 81.0), (16.0, 64.0, 256.0)]

# 牛逼一点的if else
data = None
data + 'hello' if data else "world"

# pandas 采样操作
pandas.['columns_name'].sample(1).values[0]

# 各列之间是否存在查找操作

#快速整体提取相关特征的操作
agg_funcs = {'mean':np.mean, 'sum':np.sum, 'min':np.min, 'max':np.max, 'std':np.std}
temp = train.groupby(by=['prefix','title'])['label'].agg(agg_funcs).reset_index()


# numpy 与pandas的结合操作处理多列组合逻辑操作, 当使用apply时记得调节参数axis=1，不然会报错
df['label'] = np.where((df_result_test['协议结束时间_x_x'] == df_result_test['协议结束时间_x_y'])
                     & (df_result_test['协议结束时间_y'] > df_result_test['协议结束时间_x_x']) ,
                     df_result_test['协议结束时间_y'], df_result_test['协议结束时间_x_x'])

df['label'] = df.apply(lambda x : x['协议结束时间_y'] if (x['协议结束时间_x_x'] == x['协议结束时间_x_y'])
                     & (x['协议结束时间_y'] > x['协议结束时间_x_x']) else
                     x['协议结束时间_x_x'], axis=1)


# 这两种操作方式是一样的，但是上面的那种方式是通过空间换取时间的方法。还可以使用groupby这种forloop的方式进行

# pandas groupby的骚操作
df.groupby('a')['b'].transform(func)  (func = 'count', 'max', 'min'), 
# 这样会保持和原来一样的行数，不会进行行数的减少。

# pandas中对两列数据进行操作的方法以及 (True, False)*1 可以变为1,0的操作方法，以及pandas的两列数据进行函数式操作的方法
val['tinq']=val[['prefix','title']].apply(lambda x:(x[1] in x[0])*1, axis=1)
val['max_pred_in_title']=val[['query_prediction','title']].apply(lambda x : max_pred_in_title(x[0],x[1]),axis=1)
其中max_pred_in_title(x[0], x[1])  就是函数的表达形式，其中（x[0], x[1]）为函数的输入数据

# 直接将一个DataFrame类型的数据操作结果转换为一个字典
sub_pred_list=val['query_prediction'].apply(lambda x : pred_order(x)).apply(lambda x: {'pred_score_'+str(i): pred for i, pred in enumerate(x)})


# pandas中以columns的类型进行选择
numerical = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
temp = val.select_dtypes(include=numerics)


# pandas判断后者去掉某列数字中的其它杂乱的类型数据
df['a'] = pd.to_numeric(df['a'], errors='coerce')

# pandas两张表的操作时间复杂度的比较
df = pd.DataFrame([[1,3]], columns=list('AB'))
df2 = pd.DataFrame([[5,7]], columns=list('AB'))
df = pd.concat([df, df2]).reset_index(drop=True)

df = pd.DataFrame([[1, 3]], columns=list('AB'))
dicts = {'A':5, 'B':7}
df = df.append(dicts,ignore_index=True)
# 其中后面那个的时间复杂度要比前面一个的时间复杂度要低一点

# 对pandas中混杂的数据进行操作，先转为str，然后根据实际的情况，适机而动，比如对str进行增删查改操作！
tt = pd.DataFrame({'a':[1,2,'aa','d'], 'b':[3,2,'h','5']})
tt.a.str.isnumeric()
0      NaN
1      NaN
2    False
3    False



# pandas对某列数据按照其它的列进行分组统计个数

df = pd.DataFrame({
            "key1":     ["a", "a", "b", "b", "a"],
            "key2":     ["one", "two", "one", "two", "one"],
            "data1":    np.random.randn(5),
            "data2":    np.random.randn(5)
        })
 
# 按照key1, key2分组, 对data1列计数
key12 = df["data1"].groupby([df["key1"], df["key2"]]).count().unstack()


# pandas对空值的处理方法
# 显示DataFrame中有NaN值的数据
df_gounei[df_gounei.isnull().values == True]
# 判断一行或者一列中是否有空值的方法,其中axis参数用于控制是统计行还是统计列
df.isnull().any(axis=1)
# 判断列数据中的空值，是判断一个还是判断一列数据中是否为nan, 返回值为bool值类型，
# 一个值的判断方法：
np.isnan(df.loc[1,'columns_name'])
# 一列值得判断方法：
pd.is_null(df['columns_name'])

# python中分箱操作代码
test = pd.DataFrame([[1,2],[3,4], [4,9],[6,25], [3,6],[1,9]], columns=list('ab'))
	a	b
0	1	2
1	3	4
2	4	9
3	6	25
4	3	6
5	1	9
test.b.apply(lambda x: 0 if x <=7 else 2 if x>=24 else 1)
# 连环骚操作
test.b.apply(lambda x:0 if x<=7 else 2 if x<=15 else 3 if x<=24 else 1)
# 进行区间的映射，如果小于7为0，7-24之间为2， 大于24则为1

# pandas中删除相同列的方法
df = df.T.drop_duplicates().T

# numpy中将多维的数据转换为一维的数据, 其中二者的区别在于flatten()是对数据进行一份拷贝，而ravel是对数据进行修改
a = np.array([[1 , 2] , [3 , 4]])
b = a.flatten()
c = a.ravel()
