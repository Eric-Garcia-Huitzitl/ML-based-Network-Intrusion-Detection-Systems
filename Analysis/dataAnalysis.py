import pandas as pd 
import matplotlib.pyplot as plt


dataSet = pd.DataFrame

class Analysis:
    def __init__(self,PATH,nameX):
       self.dataSet = pd.read_csv(PATH, names = nameX ) #Read CSV file
    def data_Inf(self): #Print the number of instances and attributes of the dataset
        print('Number of instances and attributes')
        print(self.dataSet.shape)    
    def data_Describe(self):
        pd.set_option('display.width',100)
        pd.set_option('display.precision', 3)
        print(self.dataSet.describe())    
    def data_Types(self):
        print('Data types of features') #Print the data types of the dataset
        print(self.dataSet.dtypes)    
    def show_CorrelationMatrix(self):
        print('Correlation Matrix')
        print(self.dataSet.corr())
    def show_hist(self):
        self.dataSet.hist()
        plt.show()
    
        
        

nameX = ['id','duration','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate,srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','protocol_type','service','flag','xAttack']
A = Analysis('dos.csv',nameX)
A.data_Types()
