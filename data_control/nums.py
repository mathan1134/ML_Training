import numpy as np

# arr=[1,2,3,4,5,9]
# # print(arr)
# a=np.array(arr)                     
# # print(a)

# print(a.shape)    # list cant shape
# print(a.dtype)

#           array initialization

# zeros=np.zeros((2,3))
# ones=np.ones((4,5))
# rand=np.random.rand(2,3)
# print(rand)


#               reshape & indexing

# mat=np.arange(0,15).reshape(5,3)
# print(mat)
# # print(mat[0,:])     #[0:1] also and row
# # print(mat[:,2])     # column
# print(mat[2,2])       # find or select



#                                   vectorize operater

# a=np.array([1,2,3])
# b=np.array([4,5,6])

# print(a+b)    # add
# print(b*a)
# print(np.dot(a,b))    # multiple and add


#                                matriz operations

# a=np.array([[1,2],[4,5]])
# b=np.array([[3,2],[8,9]])
# # print(b.T)    # transform  b and a
# print(np.matmul(a,b))                                               #algib..   ---| 



#                                     statiscial function

# data=np.random.randint(1,100,size=10)
# # print(np.mean(data))                                                 #randaom take 10 and divide 10
# # print(np.median(data))             
# # print(np.std(data)) 
# print(np.max(data)) 
# print(np.min(data)) 
# # print(data)


#                                          boolean masking
# a=np.arange(10)
# mask=a>5
# # print(mask)
# print(a[mask])

#                                   broadcasting example

# x=np.array([8,2,3])
# y=np.array([[8],[3],[7]])


#                 # x -->copy 823,823,823
#                 # y ----> 888 333 777

# print(x+y)                                                    