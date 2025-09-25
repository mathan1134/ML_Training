# if True:      # indentation
#     print("Indented block")


#     #    data types
# x=10  #integer
# y="string"  # string

# # control flow
# for i in range (4):
#     print(i)

#                                       # function
# name="john"
# def greatm_name(name):
#     return f"hello {name}"
# print(greatm_name(name))

#                                      # class
# class person:
#     def __init__(self,name):
#         self.name=name
# per_name=person("mathan")       

# print(per_name.name)


#     #  n version
# class person:
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age
#     def greet(self):
#         print(f"my name :{self.name}")
#     def s_age(self):
#         print(f"my age :{self.age}") 
        

# p1=person("mathan",25)  
# p2=person("john",20)

# p1.greet()
# p1.s_age()
# p2.greet()
# p2.s_age()
        
        

#                              # import modules
# import math
# print(math.sqrt(16))



#                                   error handling           basic

# def risky_operation():
#     print(100/0)         #error

#     # print(1+1)      #right

# try:
#     risky_operation()
# except:
#     print("error acured")    
# finally:
#     print("code executed")    



#                                 raise exception


# def found():
#   x=-4
#   if x < 4:
#     raise ValueError("value error")
# try:
#     found() 
# except ValueError as e:
#    print("error showing: ",e)    
# finally:
#     print("code executed")    




       #                                   Data types

# list=[1,2,3,4,5]
# set={1,2,3,4}
# dict={"a":1,"b":2,"c":3}
# tuple=(12,34,56)
# boolean="yes" or "no"
# none="none"

#                                         oops concept
#                                        class & object


# class car:    #class
#     def __init__(self,model,brand):
#         self.model=model 
#         self.brand=brand
#     def dis_dts(self)    :
#         print(f"car detls :{self.model} {self.brand}")

# dtl=car("800","maruthi")      #object
# dtl.dis_dts()



#                                          enscapsulation
#                                          puplic,rivate

# class bankaccount:
#     def __init__(self,balance):
#         self.__balance=balance

#     def deposite(self,amount)    :
#         self.__balance += amount

#     def g_balnce(self)   :
#         return self.__balance
    
# account=bankaccount(1000)    
# account.deposite(500)
# print(account.g_balnce())


#                             polymorpisam(running sub lass main class not run)


# class animal:
#     def speak(self):
#         print("animal can't speak")

# class dog(animal):
#     def speak(self):
#         print("dog barks")

# doggy=dog()        
# doggy.speak()




#                                       inheritance

# class bird:
#     def fly(self)   :
#         print("bird can fly")

# class peguin(bird):
#     def fly(self):
#         print("penguin can't fly")

# bird=bird()        
# peguin=peguin()

# bird.fly()
# peguin.fly()






#                                     notes
#  Class = Blueprint

# Object = Instance of class

# self = Current object reference

# Inheritance = Reuse parent class features

# Polymorphism = Same method, different behavior

# Encapsulation = Hide sensitive data (__private)

# Abstraction = Hide unnecessary details, show only essentials