# pd.DataFrame() → create

# read_csv, to_csv → load/save

# head, info, describe → inspect

# loc, iloc → select

# groupby → summarize

# merge → combine


import pandas as pd


# data = {
#     "name": ["Mathan", "John", "Sara"],
#     "age": [25, 30, 22],
#     "city": ["Chennai", "Delhi", "Mumbai"]
# }

# df = pd.DataFrame(data)
# print(df)


# #                                            read write
# df = pd.read_csv("data.csv")
# df.to_csv("data.csv",index=False)  



                #                     data handle
# df.head()        # first 5 rows
# df.tail()        # last 5 rows
# df.info()        # summary (columns, datatypes)
# df.describe()    # statistics for numeric columns
# df.shape         # (rows, columns)


#                                        select data

# df['name']                # one column
# df[['name', 'city']]      # multiple columns
# df.iloc[0]                # first row (by index)
# df.iloc[0:2]              # first two rows
# df.loc[0, 'name']         # row 0, column 'name'


# df1 = pd.DataFrame({"id":[1,2,3], "name":["A","B","C"]})
# df2 = pd.DataFrame({"id":[1,2,4], "score":[90,85,88]})

# merged = pd.merge(df1, df2, on="id", how="outer")   # join
# print(merged)



            #                             filters
# print(df[df['age'] > 25])# rows where age > 25
# print(df[df['city'] == "Mumbai"])          # filter by city
# df[(df['age'] > 20) & (df['city']=="Delhi")]   # multiple conditions


#                 add  and update
# df['country'] = "India"          # add new column
# df['age_plus_5'] = df['age']+5   # derived column
# df['age'] = df['age']+15


#                              sort and drop





data = {
    "name": ["Mathan", "John", "Sara", "Alex", "Ravi","johny"],
    "age": [25, 30, 22, 30, 25,None],
    "city": ["Chennai", "Delhi", "Mumbai", "Delhi", "Chennai","tck"]
}

df = pd.DataFrame(data)

# df=df.sort_values(by='age', ascending=True)   # sort by age                               ul
# df.drop('city', axis=1, inplace=True)       # drop column
# df.drop(0, axis=0, inplace=True)            # drop first row


#                              group aggregate
# df=df.groupby('city')['age'].mean()   # average age per city
# df=df.groupby('city').size()          # count per city



# df.isnull().sum()                  # check missing values
# df.fillna(0, inplace=True)         # replace NaN with 0
# df.dropna(inplace=True)            # drop rows with NaN


# print(df)


#                                                join,merge,edit

# df1 = pd.DataFrame({"id":[1,2,3], "name":["A","B","C"]})
# df2 = pd.DataFrame({"id":[1,2,4], "score":[90,85,88]})

# merged = pd.merge(df1, df2, on="id", how="outer")   # join
# merged["score"].fillna(0, inplace=True)        


# print(merged)


#                                           function apply

# df['age_doubled'] = df['age'].apply(lambda x: x*2)

# print(df)