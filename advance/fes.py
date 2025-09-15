                           # Feature Engineering & Scaling

# feature engineering 
# Feature Engineering = create new or transformed features → improve model accuracy.



import pandas as pd

# Example Titanic-like dataset
data = pd.DataFrame({
    'Name': ['John Smith', 'Alice Brown', 'Tom Lee'],
    'Age': [22, 35, 58],
    'Cabin': ['C85', None, 'E46'],
    'Ticket': ['113803', '373450', '330911']
})

print("Original Data:\n", data)

# ---- Feature Engineering ----

data['Title'] = data['Name'].str.split(" ").str[0]


data['AgeGroup'] = pd.cut(data['Age'], bins=[0,18,40,60,100], labels=['Teen','Young','Adult','Senior'])

data['HasCabin'] = data['Cabin'].notnull().astype(int)

data['TicketLength'] = data['Ticket'].apply(len)

print("\nAfter Feature Engineering:\n", data)





                    #  scalling

                            #   normalization
# Features must be bounded (e.g., pixel values, neural networks, KNN, clustering).
# easy to fix so 0 to 1

# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# X=np.array([[1, 10],
#               [2, 20],
#               [3, 30],
#               [4, 40]], dtype=float)

# scaller=MinMaxScaler()
# x_normalization=scaller.fit_transform(X)

# print("orginal_data :",X)
# print("normalization data :",x_normalization)


# standardlization

# Centering Around 0 for identify 0.34,-values ,0.35 like that

# import numpy as np
# from sklearn.preprocessing import StandardScaler

# X=np.array([[1, 10],
#               [2, 20],
#               [3, 30],
#               [4, 40]], dtype=float)

# scaller=StandardScaler()
# st_scaller=scaller.fit_transform(X)

# print("ordeinary data :",X)
# print("standarxziation: ",st_scaller)