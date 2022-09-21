import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import svm 
from sklearn import metrics
from sklearn import tree

import pydotplus
from IPython.display import Image
import sys
# agregando ruta de encoder.py
sys.path.insert(0, '/home/antonio/Documents/Proyecto/ML/ML-based-Network-Intrusion-Detection-Systems/Preprocessing/')
from encoder import preprocessing_datos

"""
Created on Sun Sep 11 21:28:17 2022

@author: eric
"""



class Classifier: 
    def __init__(self,dataFrame): #Input dataframe
        self.dF = dataFrame
        self.x = self.dF.drop(['xAttack'], axis=1).values   
        self.y = self.dF['xAttack'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, random_state=0)
    def kNN(self,n_neighbors):
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(self.X_train,self.y_train)
        print('Accuracy of K-NN classifier on training set: {:.2f}'
              .format(knn.score(self.X_train, self.y_train)))
        print('Accuracy of K-NN classifier on test set: {:.2f}'
              .format(knn.score(self.X_test, self.y_test)))
        #Report Confusion Matrix
        #self.pred = knn.predict(self.X_test)
       # print("Confusion matrix")
        #print(confusion_matrix(self.y_test, self.pred))
        #Report Classification
      # print(classification_report(self.y_test, pred))
    def SVM(self):
        self.clf = svm.SVC(kernel='poly', degree=4) #Linear kernel 
        self.clf.fit(self.X_train,self.y_train)
        self.ypred = self.clf.predict(self.X_test)
        print("Accuracy:",metrics.accuracy_score(self.y_test, self.ypred)) 
    def arbolDecision(self):
        print("---------------------------Arbol de decision-----------------------")
        clf=tree.DecisionTreeClassifier()
        self.X_train=self.X_train.astype('int')
        self.y_train=self.y_train.astype('int')
        clf_train=clf.fit(self.X_train,self.y_train);
        prediccion=clf.predict(self.X_test);
        print("Accuracy:",metrics.accuracy_score(self.y_test,prediccion))
        #print(tree.export_graphviz(clf_train, None))
        #dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(self.dF.columns.values), class_names=['0', '1'],rounded=True, filled=True)
        #graph=pydotplus.graph_from_dot_data(dot_data)
        #Image(graph.create_png())
        
        

nfile = '/home/antonio/Documents/Proyecto/ML/ML-based-Network-Intrusion-Detection-Systems/Dataset/csv_result-TRAindos3000.csv'
namesX = ['id','duration','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate,srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','protocol_type','service','flag','xAttack']
df = pd.read_csv(nfile, names = namesX)
df = df.select_dtypes(include=[object])
le = preprocessing.LabelEncoder()
df2 = df.apply(le.fit_transform)
print(df2.dtypes)
classifier_obj = Classifier(df2)
classifier_obj.arbolDecision();
#classifier_obj.kNN(3)
#classifier_obj.SVM()


#Error en algunas etiquetas de la BD con extension arff#
#p=preprocessing_datos()
#ruta_nombreBD="/home/antonio/Documents/Proyecto/ML/ML-based-Network-Intrusion-Detection-Systems/NSLKDD-Dataset-master/DOS -d/KDDTest21DOSFS.arff"
#bd,t=p.leerBD(ruta_nombreBD);
#pre=p.label_Encoder_BD(bd);
#p.imprimirTipos_LabelEncoding()
#pre=p.one_hot_encoder_BD(bd);
#p.imprimirTipos_OneHot()
#c=Classifier(bd)
#c.arbolDecision()
#c.arbolDecision()





    
    
