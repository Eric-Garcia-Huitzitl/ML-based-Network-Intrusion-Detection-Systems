#
#Author: José Antonio Sánchez Tiro
#
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.io import arff
import os.path

class preprocessing_datos:
    def __init__(self) -> None:
        self.df=pd.DataFrame;
        self.df_le=pd.DataFrame;
        self.df_ohe=pd.DataFrame;
        self.laen=0;
        self.onhot=0;
        self.original=0;
    def leerBD(self,ruta): #recibe ruta con nombre de la bd/
        if(os.path.exists(ruta)):
            data=arff.loadarff(ruta); #lee bd
            self.df = pd.DataFrame(data[0]) #la guarda para usarla en toda la clase
            self.df_le= pd.DataFrame(data[0]) #la guarda para usarla en toda la clase
            self.df_ohe= pd.DataFrame(data[0]) #la guarda para usarla en toda la clase
            return self.df,True
        else:
            print("-----------------------------------Verifique su ruta o archivo---------------------");
        return False
    def label_Encoder_BD(self,BD):#(bd,->return bd-preprocesado
        self.df=BD;
        le = preprocessing.LabelEncoder()
        #print(self.df.head())
        x=self.df.columns;
        i=0;
        for y in self.df.dtypes:
            if(y=="object" and x[i]!="xAttack"):
                self.df_le[x[i]]=le.fit_transform(self.df[x[i]]) #conviertiendo todos los atributos nominales con label encoding
            i=i+1;
        self.laen=1;
        print(self.df_le.dtypes)
        self.df_le['xAttack'] = self.df_le['xAttack'].values.astype(np.int64)
        for colum in self.df_le.columns:
            self.df_le[colum] = self.df_le[colum].values.astype(np.int64)
        return self.df_le;
    def one_hot_encoder_BD(self,BD):
        self.df=BD;
        #sparse (false): Devolverá una matriz DISPERSA si se establece True, en este caso devolvera un array
        #error: genera un error si una categoria desconocida estta presente durante la transformación
        #drop:{first,if_binary} o una forma similar a una matriz (n_características), predeterminado = Ninguno
        #     Ninguno: conserva todas las funciones (predeterminado).
        #     'first': suelta la primera categoría en cada característica. Si solo hay una categoría presente, la función se eliminará por completo.
        #     'if_binary': suelta la primera categoría en cada característica con dos categorías. Las características con 1 o más de 2 categorías se dejan intactas.
        #      array : drop[i] es la categoría en la función X[:, i] que debe eliminarse
        ohencoder=preprocessing.OneHotEncoder(sparse=False,
                           handle_unknown='error',
                           drop='first')
        x=self.df.columns;
        i=0;
        self.df.select_dtypes(include=[object])
        self.df_ohe = pd.DataFrame(self.df.apply(ohencoder.fit_transform))
        #for y in self.df.dtypes:
        #    if(y=="object" and x[i]!="xAttack"):
        #        self.df_ohe=pd.DataFrame(ohencoder.fit_transform(self.df[[x[i]]])) #conviertiendo todos los atributos nominales con one hot
        #    i=i+1;
        self.onhot=2;
        print(self.df_ohe)
        self.df_ohe['xAttack'] = self.df_ohe['xAttack'].values.astype('Int64')

        return self.df_ohe


    def imprimirTipos_LabelEncoding(self):
        if(self.laen==0):
            print("BD vacia!");
        else:
            print("--------------------------------");
            print("Atributo             |            tipo");
            print(self.df_le.dtypes);
            print("--------------------------------");
    
    def imprimirTipos_OneHot(self):
        if(self.onhot==0):
            print("BD vacia!");
        else:
            print("--------------------------------");
            print("Atributo             |            tipo");
            print(self.df_ohe.dtypes);
            print("--------------------------------");
    
    def imprimirTipos_Original(self):
        if(self.laen==0 and self.onhot==0):
            print("BD vacia!");
        else:
            print("--------------------------------");
            print("Atributo             |            tipo");
            print(self.df.dtypes);
            print("--------------------------------");
    
    def imprimirBD(self):
        if(self.laen==0 and self.onhot==0):
            print("BD vacia!");
        else:
            print("--------------------------------");
            print("-------------Mostrando base de datos-------------------");
            print(self.df.columns.get_value)
            print("--------------------------------");
            print("");
            print(self.df);
            print("--------------------------------");
    def imprimirBD_LE(self):
        if(self.laen==0):
            print("BD vacia!");
        else:
            print("--------------------------------");
            print("-------------Mostrando base de datos-------------------");
            print(self.df_le.columns.get_value)
            print("--------------------------------");
            print("");
            print(self.df);
            print("--------------------------------");
    def imprimirBD_OHE(self):
        if(self.onhot==0):
            print("BD vacia!");
        else:
            print("--------------------------------");
            print("-------------Mostrando base de datos-------------------");
            print(self.df_ohe.columns.get_value)
            print("--------------------------------");
            print("");
            print(self.df);
            print("--------------------------------");

#ruta_nombreBD="/home/antonio/Documents/Proyecto/ML-based-Network-Intrusion-Detection-Systems/NSLKDD-Dataset-master/DOS -d/KDDTest21DOSFS.arff"
#p=preprocessing_datos();
#p.leerBD('');#lee la bd de tipo arff, 
#if(p.leerBD(ruta_nombreBD)):#lee la bd de tipo arff, 
    #p.label_Encoder_BD()
    #p.imprimirTipos_LabelEncoding()
    #p.imprimirBD()
 #   p.one_hot_encoder_BD()
 #   p.imprimirTipos_OneHot()
 #   p.imprimirBD_OHE()
