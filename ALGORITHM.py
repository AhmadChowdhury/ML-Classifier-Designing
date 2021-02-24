#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[30]:


# load csv data
iris = pd.read_table("C:/Users/User/Desktop/Online class/Anaconda/iris.txt", sep=',')
iris.info()


# In[31]:


iris


# In[32]:


iris["species"].value_counts()


# In[33]:


###----------------------------------------- Extract Classes ----------------------------------------------------------------###
First=iris[iris.species=='Iris-setosa']
Second=iris[iris.species=='Iris-versicolor']
Third=iris[iris.species=='Iris-virginica']


# In[34]:


########################################## Select any two class ############################################################### 


# In[35]:


'''    First='Iris-setosa'  Second='Iris-versicolor'  Third='Iris-virginica'     '''
A=Second
B=Third

labelA='Iris-versicolor'
labelB='Iris-virginica'


AB=A.append(B)


# In[36]:


######################################## Select any two features ############################################################## 


# In[37]:


#'sepal_length'	'sepal_width'	'petal_length'	'petal_width'
feature1='petal_width'
feature2='petal_length'


# In[38]:


iris["species"].value_counts()


# In[39]:


sns.set(style='whitegrid')
sns.FacetGrid(iris, hue="species",size=8)   .map(plt.scatter,"petal_length", "petal_width")   .add_legend()
plt.title('ALL CLASSES')
plt.show()


# In[40]:


###----------------------------------------- Plot any two features ---------------------------------------------------------###
sns.set(style='whitegrid')
sns.FacetGrid(AB, hue="species",size=8)   .map(plt.scatter,feature1, feature2)   .add_legend()
plt.title('SELECTED CLASSES')
plt.show()


# In[43]:


###---------------------------------------------- Inputs --------------------------------------------------------------------###
# class A
XA=np.array([])
YA=np.array([])

XA=np.append(XA,[A[feature1]])
YA=np.append(YA,[A[feature2]])

# class B
XB=np.array([])
YB=np.array([])

XB=np.append(XB,[B[feature1]])
YB=np.append(YB,[B[feature2]])

##############################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics

###---------------------------------------------- Functions -----------------------------------------------------------------###
def connector(x1,y1,k,x2,y2,n):
    slope= (y1-y2)/(x1-x2)
    midpoint=np.array([(x2*k+x1*n)/(k+n),(y2*k+y1*n)/(k+n)])
    return slope,midpoint

def bisector(slope,point,angle):
    theta=math.degrees(math.atan(slope))+int(angle)
    m=math.tan(math.radians(theta))
    c=point[1]-point[0]*m
    return(m,c)

def margin(point,m,c):
    y=point[1]
    x=point[0]
    value=y-m*x-c
    if value>=0:
        sign= 1
    else:
        sign= -1
    return sign

###---------------------------------------------- Inputs --------------------------------------------------------------------###
x=XA
y=YA

x1=XB
y1=YB

###---------------------------------------------- Connector -----------------------------------------------------------------###

k=max(statistics.stdev(x),statistics.stdev(y))
n=max(statistics.stdev(x1),statistics.stdev(y1))

X=np.mean(x)
Y=np.mean(y)
X1=np.mean(x1)
Y1=np.mean(y1)

(slope, midpoint) = connector(X, Y, k, X1, Y1, n)
print(midpoint)
accuracy_max=0

###---------------------------------------------- Decision line -------------------------------------------------------------###
for angle in range(0,360):
    wrong=0;
    class1_false = 0
    class2_false = 0


    (m,c)=bisector(slope,midpoint,angle)

    #First set Calc
    point=np.array([X,Y])
    primesign=margin(point,m,c)
    primesign1=primesign
    for i in range(len(x)):
        point=np.array([x[i],y[i]])
        if margin(point,m,c)!=primesign:
            wrong=wrong+1
            class1_false=class1_false+1

    #2nd set Calc
    point=np.array([X1,Y1])
    primesign=margin(point,m,c)
    for i in range(len(x1)):
        point=np.array([x1[i],y1[i]])
        if margin(point,m,c)!=primesign:
            wrong=wrong+1
            class2_false = class2_false + 1

    total=len(x)+len(x1)
    accuracy=(total-wrong)/total*100
    if accuracy>accuracy_max:
        accuracy_max=accuracy
        M=m
        C=c
        angle_with_connector=int(angle)
        class2_False=class2_false
        class1_False=class1_false

###-------------------------------------------------- Results ---------------------------------------------------------------###
print('accuracy_max          :', accuracy_max)
print(labelA,'_False:',class1_False)
print(labelB,'_False :',class2_False)
print('total                 :',total)
###---------------------------------------------------- Plot ----------------------------------------------------------------###
t= np.arange(-1000,1000,.01)
plt.figure()

ry=min(statistics.stdev(y),statistics.stdev(y1))
rx=min(statistics.stdev(x),statistics.stdev(x1))

plt.ylim(min(min(y),min(y1))-ry, max(max(y),max(y1))+ry)
plt.xlim(min(min(x),min(x1))-rx, max(max(x),max(x1))+rx)
plt.scatter(x,y,label=labelA)
plt.scatter(x1,y1,label=labelB)
plt.plot(t, M*t+C,'k')
plt.title('Scatter Plot')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()
plt.rcParams['figure.figsize']=[12,8]
plt.show()


# In[ ]:




