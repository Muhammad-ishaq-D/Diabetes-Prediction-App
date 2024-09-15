import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score, pair_confusion_matrix
import seaborn as sns
from tkinter import TkVersion
from tkinter import messagebox
from tkinter import *

dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

def predict():
     
    glucose=entry1.get()
    bp=entry2.get()
    st=entry3.get()
    insulin=entry4.get()
    age=entry5.get()

    prediction = classifier.predict(sc.transform([[glucose,bp,st,insulin,age]]))          
    if(prediction[0]==0):
        messagebox.showinfo("","This person is Non-Diabetic")
    else :
        messagebox.showinfo("","This person is Diabetic") 
def accuracy_train():     
    X_train_prediction=classifier.predict(X_train)
    training_data_accuracy_score=accuracy_score(y_train,X_train_prediction)   
    messagebox.showinfo("",f"Accuracy Score of testing data : {training_data_accuracy_score * 100} %")
    
def accuracy_test(): 
    X_test_prediction=classifier.predict(X_test)
    testing_data_accuracy_score=accuracy_score(y_test,X_test_prediction)
    messagebox.showinfo("",f"Accuracy Score of testing data : {testing_data_accuracy_score * 100} %")
def histogram():
    dataset.hist(bins = 20,figsize= (15,15) )
    plt.show()
def pie_chart():
    labels= 'Diabetic','Not Diabetic'
    plt.pie(dataset['Outcome'].value_counts(),labels=labels,autopct='%0.02f%%')
    plt.legend()
    plt.show()
def conf_matrix():
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm,annot=True,fmt='g',cmap='Blues',xticklabels=['Yes','No'],yticklabels=['Yes','No'])
    plt.xlabel('Actual',fontsize=14)
    plt.ylabel('Prediction',fontsize=14)
    plt.title("Confusion Matrix",fontsize=18)
    plt.show()
def s_plot():
    corr_columns = ['Glucose',"BloodPressure","SkinThickness","Insulin"]
    for col in corr_columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x="Age",y=col,data=dataset,hue="Outcome")
        plt.title(f"Comparing distribution of {col} of patients with diabetes and without diabetes")
        plt.show()
root = Tk()
root.geometry("900x900")
name=StringVar()
global entry1
global entry2
global entry3
global entry4
global entry5
Label(root,text="D I A B E T E S     P R E D I C T I O N",fg="black",font=10).place(x=350,y=85)
Label(root,text="A C C U R A C Y    C H E C K",fg="black",font=10).place(x=350,y=280)
Label(root,text="D A T A     V I S U A L I Z A T I O N",fg="black",font=10).place(x=350,y=440)
Label(root,text="Glucose").place(x=200,y=150)
entry1=Entry(root,width=10)
entry1.place(x=250,y=150)
Label(root,text="BloodPressure").place(x=320,y=150)
entry2=Entry(root,width=10)
entry2.place(x=400,y=150)
Label(root,text="Skinthickness").place(x=470,y=150)
entry3=Entry(root,width=10)
entry3.place(x=550,y=150)
Label(root,text="Insulin").place(x=620,y=150)
entry4=Entry(root,width=10)
entry4.place(x=660,y=150)
Label(root,text="Age").place(x=730,y=150)
entry5=Entry(root,width=10)
entry5.place(x=760,y=150)

Button(root, text='Predict', font=10, width=30, bg="green", fg="white", command=predict).place(x=350,y=200)
Button(root, text='Accuracy Score of training data', font=10, width=25, bg="red", fg="white", command=accuracy_train).place(x=220,y=350)
Button(root, text='Accuracy Score of testing data', font=10, width=25, bg="red", fg="white", command=accuracy_test).place(x=550,y=350)
Button(root, text='Histograms', font=10, width=25, bg="blue", fg="white", command=histogram).place(x=350,y=500)
Button(root, text='Pie Chart', font=10, width=25, bg="blue", fg="white", command=pie_chart).place(x=350,y=550)
Button(root, text='Confusion Matrix', font=10, width=25, bg="blue", fg="white", command=conf_matrix).place(x=350,y=600)
Button(root, text='Scatter Plots', font=10, width=25, bg="blue", fg="white", command=s_plot).place(x=350,y=650)
Button(root,text="Exit", font=10,width=20,bg="yellow",fg="black",command=root.destroy).place(x=680,y=650)
root.mainloop()
