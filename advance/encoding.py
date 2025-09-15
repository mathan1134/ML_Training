# encoding
#  easy to understand machine why that was not understand text so
# reduce bias feature uses and relationship datas


# label encoding

# from sklearn.preprocessing import LabelEncoder

# colors1=["red","blue","yellow","white","pink"]
# colors2=["red","blue","red","white","pink"]
# encoder=LabelEncoder()
# encode=encoder.fit_transform(colors1)
# encode1=encoder.fit_transform(colors2)
# print(encode,encode1)


# one hot encoding

# from sklearn.preprocessing import OneHotEncoder
# import numpy as np

# color=np.array(["red","blue","green"]).reshape(-1,1)
# encode=OneHotEncoder(sparse=False)
# encoded=encode.fit_transform(color)

# print(encoded)


# ordinery encoding/mapping

# import pandas as pd

# sizes=pd.Series(["small","medium","large"])
# assign={"small":0,"medium":1,"large":2}
# ordin=sizes.map(assign)
# print(ordin)


# target

# import pandas as pd

# df=pd.DataFrame({
#     "City": ["Paris", "London", "Paris", "Delhi"],
#     "Size": ["Small", "Large", "Medium", "Small"]
# })


# # one_hot-encoding
# one_hot=pd.get_dummies(df,columns=["City"])
# print(one_hot)

# # ordinery
# size_map = {"Small": 0, "Medium": 1, "Large": 2}
# df["Size_encoded"] = df["Size"].map(size_map)
# print(df)