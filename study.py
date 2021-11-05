import numpy as np
import pandas as pd

test_df = pd.read_csv("Titanic/test.csv")
train_df = pd.read_csv("Titanic/train.csv")

# 将Y看成分类型变量，X：性别，船舱，港口 分别对Y一对一列联卡方独立检验

a = train_df[['Sex', 'Survived']]
# 取一列或行做Series
a = train_df.loc[1]  # 首选
a = train_df.iloc[1]

a = train_df['Sex']
a = train_df.loc[:, 'Sex']  # 首选
a = train_df.iloc[:, 1]
# 取多行或多列做子df
# 列
a = train_df[['Sex', 'Survived']]
a = train_df.loc[:, ['Sex', 'Survived']]  # 首选
a = train_df.iloc[:, 1:3]
a = train_df.iloc[:, [1, 2, 4]]
# 行
a = train_df[:3]
a = train_df.iloc[[1, 3, 4]]
a = train_df.iloc[:3]
a = train_df.loc[[1, 3, 4]]
a = train_df.loc[:3]

# 取一个值如何取
train_df['Sex'][1]
train_df.loc[1]['Sex']  # 首选
