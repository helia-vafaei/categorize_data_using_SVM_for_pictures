import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 


dataset = ["book" , "watch" , "cup"]
data = []
for i in range(len(dataset)):
    path1 = "\\".join(["img",dataset[i]])      # p = img\watch , img\book
    for img in os.listdir(path1):    #img -> watch14.jpg    
        path2 = "\\".join([path1,img])  #img\watch\watch14.jpg
        img_read = cv2.imread(path2 , 0)    #pixels
        try:
            img_read = cv2.resize(img_read , (170,170))
            img_read = np.array(img_read).flatten()
            data.append([img_read , i])    
        except Exception:
            pass
x = []
y = []
for item1,item2 in data:
    x.append(item1)
    y.append(item2) 

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


my_tree = SVC(kernel='linear')   #sigmoid 7 , poly 76 , rbf 69 , linear 76
my_tree.fit(X_train, Y_train)
my_Y = my_tree.predict(X_test) 

def possibly(Y_test, my_Y):
    sum=0
    n=len(Y_test)
    for i in range(n):
        if Y_test[i] == my_Y[i]:
            sum+=1
    return sum/n   

poss = possibly(Y_test, my_Y)   
print(poss) 