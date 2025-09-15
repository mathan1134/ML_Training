#                             matplotlip

# import matplotlib.pyplot as plt

# X=[1,2,3,4,5]
# y=[1,4,6,16,25]

#                plot line
# plt.plot(X,y, color="green")
# plt.title("line plot" ,color="red")
# plt.xlabel("x_label")
# plt.ylabel("y_lael")
# plt.show()


#            scatter    (points)
# plt.scatter(X,y,color="red")             
# plt.title("scatter")
# plt.xlabel("x_label")
# plt.ylabel("y_label")
# plt.show()

#                              bar chart


# catgs=["a","b","c","d"]
# values=[1,2,3,4]

# plt.bar(catgs,values,color="red")
# plt.plot(catgs,values,color="green")
# plt.title("bar")
# plt.show()


#                                         histogram
# data=[1,2,2,5,5,5,5,3,3,3,4,4,5,5,5,5,6,7]
# plt.hist(data,bins=5,color="green",edgecolor="black")
# plt.title("histogram")
# plt.show()



#                               piechart
# sizes = [30, 25, 20, 25]
# labels = ['A', 'B', 'C', 'D']

# plt.pie(sizes, labels=labels, autopct='%1.1f%%')
# plt.title("Pie Chart")
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# data = pd.DataFrame({'Year':[2020,2021,2022],'Sales':[100,150,200]})
# plt.plot(data['Year'], data['Sales'])
# plt.show()

#          marker (round)
# x = [1,2,3,4,5]
# y = [2,4,6,8,10]
# plt.plot(x, y, color='purple', linestyle='--', marker='o')
# # plt.plot(x, y)
# plt.grid(True)   # show grid
# plt.show()
# plt.show()



#                                                     seaborn


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# data=[1, 2, 2, 3, 4, 4, 5, 6]
# data2=[1,1,1,1,1,3,4,5,6,6,6,6,6,6,8,9]

# sns.histplot(data)
# plt.savefig("first")
# plt.show()
# plt.clf()

# sns.histplot(data2)
# plt.savefig("second")
# plt.show()
# plt.clf()



                            # scatterplot
df=pd.DataFrame({
    "x":[1,2,3,4,5],
    "y":[5,4,3,2,1],
    "cate":["a","b","c","d","e"]
})

# sns.scatterplot(x="x",y="y",hue="cate",data=df)
# plt.show()



#                      line,plot,bar

# sns.lineplot(x="x",y="y",data=df)
# sns.barplot(x="cate",y="y",data=df)
# sns.boxplot(x="cate",y="y",data=df)
# plt.show()


#                   heatmap

# import numpy as np

# matri=np.array([[1,2,3],[4,5,6],[7,8,9]])
# sns.heatmap(matri, annot=True, cmap="coolwarm")
# plt.show()


