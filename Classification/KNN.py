import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


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
       # pred = knn.predict(self.X_test)
       # print("Confusion matrix")
       # print(confusion_matrix(y_test, pred))
        #Report Classification
      # print(classification_report(self.y_test, pred))
   
    

nfile = 'dos.csv'
namesX = ['id','duration','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate,srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','protocol_type','service','flag','xAttack']
df = pd.read_csv(nfile, names = namesX)
df = df.select_dtypes(include=[object])
le = preprocessing.LabelEncoder()
df2 = df.apply(le.fit_transform)
classifier_obj = Classifier(df2)
classifier_obj.kNN(3)







