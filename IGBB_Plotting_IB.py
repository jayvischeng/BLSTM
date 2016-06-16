import os
import time
import math
start = time.time()
import numpy as np
import random as RANDOM
#from svmutil import *
#import seaborn as sns
import matplotlib.pyplot as plt
from numpy import *
from sklearn import tree
from InformationGain import *
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA,KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,datasets,preprocessing,linear_model
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier

def LoadData(filename):
    """
    global input_data_path,out_put_path

    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"AS_Path_Error_half_minutes.txt"))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"AS_Leak_half_minutes.txt"))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"AS_Filtering_Error_half_minutes.txt"))
    y_svmformat, x_svm_format = svm_read_problem(os.path.join(input_data_path,filename))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"Nimda_AS_513_half_minutes.txt"))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"Code_Red_I_AS_513_half_minutes.txt"))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"AS_Filtering_Error_AS_286_half_minutes.txt"))

    y_svmformat=np.array(y_svmformat)
    y_svmformat[y_svmformat==-1]=positive_sign#Positive is -1

    Data=[]
    for tab in range(len(x_svm_format)):
        Data.append([])
        temp=[]
        for k,v in x_svm_format[tab].items():
            temp.append(int(v))
        Data[tab].extend(temp)
        Data[tab].append(int(y_svmformat[tab]))
    Data=np.array(Data)
    np.random.shuffle(Data)
    np.random.shuffle(Data)
    return Data
    """
    global count_positive,count_negative
    with open(os.path.join(input_data_path,filename)) as fin:
        if filename == 'sonar.dat':
            negative_flag = 'M'
        elif filename == 'bands.dat':
            negative_flag = 'noband'
        elif filename =='Ionosphere.dat':
            negative_flag = 'g'
        elif filename =='spectfheart.dat':
            negative_flag = 'g'
        elif filename =='spambase.dat':
            negative_flag = '0'
        elif filename =='page-blocks0.dat':
            negative_flag = 'negative'
        elif filename =='blocks0.dat':
            negative_flag = 'g'
        elif filename =='heart.dat':
            negative_flag = '2'
        elif filename =='segment0.dat':
            negative_flag = 'g'
        elif filename == 'BGP_DATA.txt':
            negative_flag = '1.0'
        elif filename == 'Code_Red_I.txt':
            negative_flag = '1.0'
        elif filename == 'Nimda.txt':
            negative_flag = '1.0'
        elif filename == 'Slammer.txt':
            negative_flag = '1.0'
    #with open("AS_Filtering_Error_AS_286_half_minutes.txt") as fin:
        Data=[]

        for each in fin:
            if '@' in each:
                continue
            val=each.split(",")
            if len(val)>0 or val[-1].strip()=="negative" or val[-1].strip()=="positive":
                #print(each)
                if val[-1].strip()== negative_flag:
                    val[-1]=negative_sign
                    count_negative += 1
                else:
                    val[-1]=positive_sign
                    count_positive += 1
                try:
                    val=map(lambda a:float(a),val)
                except:
                    val=map(lambda a:str(a),val)

                val[-1]=int(val[-1])
                Data.append(val)
        Data=np.array(Data)
        return Data
if __name__=='__main__':
    global positive_sign,negative_sign,count_positive,count_negative,out_put_path
    #os.chdir("/home/grads/mcheng223/IGBB")
    positive_sign=-1
    negative_sign=1
    count_positive=0
    count_negative=0
    input_data_path = os.path.join(os.getcwd(),"Data4")
    out_put_path = os.path.join(os.getcwd(),"Output4")
    if not os.path.isdir(out_put_path):
        os.makedirs(out_put_path)
    filenamelist=os.listdir(input_data_path)

    #Method_Dict={"DT":1,"LR":4}
    #Method_Dict={"IGBB":1,"DT":2,"SVM":3,"LR":4,"KNN":5}
    index_ = 31
    for tab in range(len(filenamelist)):
        #plt.subplot(2,2,tab+1)
        Data_=LoadData(filenamelist[tab])
        D_data = Data_[:,index_]
        D_label = list(Data_[:,-1])
        D_Positive=Data_[Data_[:,-1]==positive_sign][:,index_]
        D_Negative=Data_[Data_[:,-1]==negative_sign][:,index_]
        X = [ i for i in range(len(D_data))]
        #print(list(Data_[:,-1]).index(positive_sign))
        #print(len(D_Positive))
        X_Positive =  [ i+D_label.index(positive_sign) for i in range(len(D_Positive))]
        plt.title(filenamelist[tab].replace(".txt",""))
        plt.plot(X,D_data,'b')
        plt.plot(X_Positive,D_Positive,'r')

    plt.suptitle("Feature_"+str(index_+1))
    plt.show()
        #Main(Method_Dict,eachfile)

    #print(time.time()-start)
#Method_Dict={"IGBB":2}
#Method_Dict={"IGBB":1,"DT":2,"SVM":3,"LR":4,"KNN":5}
#Method_List=[k for k,v in Method_Dict.items()]
#Plot_auc_list=[]
#Plot_g_mean_list=[]

"""
plt.xlim(Top_K_List[0],Top_K_List[-1]+2)
plt.xlabel("Number of Features",fontsize=18)
plt.ylim(0.4,1)
plt.ylabel("Performance",fontsize=18)
plt.tick_params(labelsize=18)
#plt.plot(Top_K_List,plot_g_mean_list,"bs-",label='IGBB-G_Mean')
plt.plot(Top_K_List,plot_auc_list,"bs-",label='AUC')
plt.grid()
plt.tight_layout()
legend = plt.legend(loc='best', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FFCC')
plt.show()
#print(result)
#print(true_label)
#print(ac_positive)
#print(ac_negative)
"""