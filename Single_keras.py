'''Trains a LSTM on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.

Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys
import os
import random as RANDOM
from sklearn import svm,datasets,preprocessing,linear_model
from sklearn.metrics import roc_auc_score
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
import matplotlib.pyplot as plt
def LoadData(input_data_path,filename):
    """
    global input_data_path,out_put_pathmodified_negative
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
    global positive_sign,negative_sign,count_positive,count_negative,out_put_path,modified_negative,modified_positivei
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
        else:
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
                    val[-1]=modified_negative
                    count_negative += 1
                else:
                    val[-1]= modified_positive
                    count_positive += 1
                try:
                    val=map(lambda a:float(a),val)
                except:
                    val=map(lambda a:str(a),val)

                val[-1]=int(val[-1])
                Data.append(val)
        Data=np.array(Data)
        return Data


def reConstruction(window_size,data,label):
    newdata = []
    newlabel = []
    L = len(data)
    D = len(data[0])
    #W = 20
    interval = 1
    index = 0
    newdata_count = 0
    initial_value = -999
    while index+window_size <= L:
        newdata.append(initial_value)
        newlabel.append(initial_value)
        Sequence = []
        for i in range(window_size):
            Sequence.append(data[index+i])
            newlabel[newdata_count] = label[index+i]
        index += interval
        newdata[newdata_count]=Sequence
        newdata_count += 1
    return np.array(newdata),np.array(newlabel)

def returnAllIndex(Data):
    temp = []
    for i in range(len(Data)):
        temp.append(i)
    return temp

def returnPositiveIndex(Data,positive_sign):
    temp = []
    for i in range(len(Data)):
        if Data[i][-1] == positive_sign:
            temp.append(i)
    return temp

def returnNegativeIndex(Data,negative_sign):
    temp = []
    for i in range(len(Data)):
        if Data[i][-1] == negative_sign:
            temp.append(i)
    return temp


def Plotting(Data1,Index_list,dimension):
    global positive_sign,negative_sign,count_positive,count_negative,out_put_path,modified_negative,modified_positivei
    X = [i+1 for i in range(len(Data1))]
    X1_positive = np.array(X)[Data1[:,-1]==modified_positive]
    Y1_positive = Data1[Data1[:,-1]==modified_positive,dimension]
    Y2 = Data1[Index_list,dimension]
    plt.plot(X,Data1[:,dimension],'b.')
    plt.plot(X1_positive,Y1_positive,'r.')
    plt.plot(Index_list,Y2,'g.')
    plt.show()
def Main(base_url,eachfile,bagging_number,cross_valtab,window_size,lstm_size,bagging_label):

    global positive_sign,negative_sign,count_positive,count_negative,out_put_path,modified_negative,modified_positivei
    Data_=LoadData(input_data_path,eachfile)

    Positive_Data=Data_[Data_[:,-1]==modified_positive]
    Negative_Data=Data_[Data_[:,-1]==negative_sign]

    #Positive_Data_Index_list=[i for i in range(len(Positive_Data))]
    #Negative_Data_Index_list=[i for i in range(len(Negative_Data))]
    print("IR is :"+str(float(len(Negative_Data))/len(Positive_Data)))
    cross_folder=3
    AllIndex = returnAllIndex(Data_)
    PositiveIndex = returnPositiveIndex(Data_,modified_positive)
    NegativeIndex = returnNegativeIndex(Data_,negative_sign)
    #cross_folder_acc_list=[]
    #cross_folder_auc_list =[]
    #cross_folder_g_mean_list=[]
    for tab_cross in range(cross_folder):
        if not tab_cross == (cross_valtab-1): continue
        #VotingList=[[] for i in range(bagging_number)]
        Positive_Data_Index_Training=[]
        Positive_Data_Index_Testing=[]
        Negative_Data_Index_Training=[]
        Negative_Data_Index_Testing=[]

        for tab_positive in range(len(PositiveIndex)):
            if int((cross_folder-tab_cross-1)*len(Positive_Data)/cross_folder)<=tab_positive<int((cross_folder-tab_cross)*len(Positive_Data)/cross_folder):
                Positive_Data_Index_Testing.append(PositiveIndex[tab_positive])
            else:
                Positive_Data_Index_Training.append(PositiveIndex[tab_positive])
        for tab_negative in range(len(NegativeIndex)):
            if int((cross_folder-tab_cross-1)*len(Negative_Data)/cross_folder)<=tab_negative<int((cross_folder-tab_cross)*len(Negative_Data)/cross_folder):
                Negative_Data_Index_Testing.append(NegativeIndex[tab_negative])
            else:
                Negative_Data_Index_Training.append(NegativeIndex[tab_negative])

        #Positive_Training_Data=np.array(Positive_Data)[Positive_Data_Index_Training]
        #Positive_Testing_Data=np.array(Positive_Data)[Positive_Data_Index_Testing]
        #Negative_Training_Data=np.array(Negative_Data)[Negative_Data_Index_Training]
        #Negative_Testing_Data=np.array(Negative_Data)[Negative_Data_Index_Testing]

        #Training_Data=np.concatenate((Positive_Training_Data,Negative_Training_Data))
        Testing_Data_Index=np.append(Negative_Data_Index_Testing,Positive_Data_Index_Testing,axis=0)
        Testing_Data_Index.sort()
        Testing_Data = Data_[Testing_Data_Index,:]
        X_Testing=Testing_Data[:,:-1]
        Y_Testing=Testing_Data[:,-1]
        #print("Y_Testing Length is "+str(len(X_Testing)))
        #outfutfilename = os.path.join("/Users/chengmin/Dropbox/IGBB_Imbalanced/SingleEvent_Keras","True_Label_folder_"+str(tab_cross+1)+"_for_"+eachfile)
        outputpath = os.path.join(base_url,"SingleEvent_Keras_B_"+str(bagging_label)+"_W_"+str(window_size))
        if not os.path.isdir(outputpath):
            os.makedirs(outputpath)
        outfutfilename1 = os.path.join(outputpath,"True_Label_folder_"+str(tab_cross+1)+"_for_"+eachfile)

        #with open(outfutfilename,"w")as fout:
            #np.savetxt(fout,Y_Testing)
            #fout.write("Bagging_Number: "+str(t)+"  Window Size: "+str(window_size)+"  LSTM Size: "+str(lstm_size)+'\t\tAccuracy: '+str(acc)+'\t\tAUC: '+str(auc)+'\t\tG_mean: '+str(g_mean))
            #fout.write("Window Size: "+str(window_size)+" LSTM Size: "+str(lstm_size)+'\t\tAccuracy: '+str(acc))
            #fout.writelines(Y_Testing)
            #fout.write('\n')
        #Training_Data=np.concatenate((Positive_Data[:int(len(Positive_Data)*0.5),:],Negative_Data[:int(len(Positive_Data)*0.5),:]))
        #Testing_Data=np.concatenate((Positive_Data[int(len(Positive_Data)*0.5):,:],Negative_Data[int(len(Positive_Data)*0.5):,:]))


        print("bagging_number is "+str(bagging_number)+"----------")
        for t in range(1):
            #Positive_Data_Samples=Positive_Training_Data
            Positive_Data_Samples_Index=Positive_Data_Index_Training

            #Negative_Data_Samples=RANDOM.sample(Negative_Training_Data,len(Positive_Data_Samples))
            Negative_Data_Samples_Index=RANDOM.sample(NegativeIndex,len(Positive_Data_Samples_Index))

            #TrainingSamples=np.concatenate((Negative_Data_Samples,Positive_Data_Samples))
            TrainingSamples_Index=np.concatenate((Negative_Data_Samples_Index,Positive_Data_Samples_Index))
            TrainingSamples_Index.sort()

            TrainingSamples = Data_[TrainingSamples_Index,:]

            Plotting(Data_,Negative_Data_Samples_Index,31)

            """
            X_Training=TrainingSamples[:,:-1]
            Y_Training=TrainingSamples[:,-1]


            X_Testing=Testing_Data[:,:-1]
            Y_Testing=Testing_Data[:,-1]

            ac_positive=0
            ac_negative=0

            scaler = preprocessing.StandardScaler()
            batch_size = 200
            #print("X_Tesing00000 Length is "+str(len(X_Testing)))
            (X_Training,Y_Training) = reConstruction(window_size,scaler.fit_transform(X_Training),Y_Training)
            (X_Testing,Y_Testing) = reConstruction(window_size,scaler.fit_transform(X_Testing),Y_Testing)

            #with open(os.path.join("/Users/chengmin/Dropbox/IGBB_Imbalanced/SingleEvent_Keras","True_Label_folder_"+str(tab_cross+1)+"_for_"+eachfile),"w")as fout:
            with open(outfutfilename1,"w")as fout:

                np.savetxt(fout,Y_Testing)

            lstm_object = LSTM(lstm_size,input_length=window_size,input_dim=33)
            print('Build model...'+'Window Size is '+str(window_size)+' LSTM Size is '+str(lstm_size))
            model = Sequential()
            model.add(lstm_object)#X.shape is (samples, timesteps, dimension)
            model.add(Dense(output_dim=1))
            model.add(Activation("sigmoid"))
            model.compile( optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            model.fit(X_Training, Y_Training, batch_size=batch_size,nb_epoch=10)
            #score, acc = model.evaluate(X_Testing, Y_Testing, batch_size=batch_size)
            #print('Test score for '+eachfile+' :', score)
            #print('Test accuracy for '+eachfile+ ' :', acc)
            #return acc
            #print("X_Testing Length is "+str(len(X_Testing)))
            TempList=model.predict(X_Testing,batch_size=batch_size)
            #print("TempList Length is "+str(len(TempList)))
            #for tab in range(len(TempList)):
                #if int(TempList[tab])==modified_positive and TempList[tab]==int(Y_Testing[tab]):
                    #ac_positive += 1
                #if int(TempList[tab])==negative_sign and TempList[tab]==int(Y_Testing[tab]):
                    #ac_negative += 1
            #ACC = float(ac_positive+ac_negative)/len(TempList)
            #auc = roc_auc_score(Y_Testing,TempList)
            #g_mean=np.sqrt(float(ac_positive*ac_negative)/(len(Y_Testing[Y_Testing==modified_positive])*len(Y_Testing[Y_Testing==negative_sign])))

            #with open(os.path.join("/Users/chengmin/Dropbox/IGBB_Imbalanced/SingleEvent_Keras","Bagging_Predict_folder_"+str(tab_cross+1)+"_for_"+eachfile+".txt"),"a")as fout:
            outfutfilename2 = os.path.join(outputpath,"Bagging_Predict_folder_"+str(tab_cross+1)+"_for_"+eachfile)

            with open(outfutfilename2,"a")as fout:

                #fout.write("Bagging_Number: "+str(t)+"  Window Size: "+str(window_size)+"  LSTM Size: "+str(lstm_size)+'\t\tAccuracy: '+str(acc)+'\t\tAUC: '+str(auc)+'\t\tG_mean: '+str(g_mean))
                #fout.write("Window Size: "+str(window_size)+" LSTM Size: "+str(lstm_size)+'\t\tAccuracy: '+str(acc))

                fout.write("----------bagging_: "+str(bagging_number)+"th"+" window size: "+str(window_size)+" lstm_size: "+str(lstm_size)+"----------\n")
                np.savetxt(fout,TempList)
                #fout.write('\n')




            #cross_folder_acc_list.append(ACC*100)
            #cross_folder_auc_list.append(auc*100)
            #cross_folder_g_mean_list.append(g_mean*100)


            #clf.fit(X_Training, Y_Training)
            #TempList = clf.predict(X_Testing)

            #VotingList[t].extend(TempList)
            """


    """
        TempOutput=[[] for i in range(len(VotingList[0]))]
        Output=[]
        for tab_i in range(len(VotingList[0])):
            for tab_j in range(len(VotingList)):
                TempOutput[tab_i].append(VotingList[tab_j][tab_i])
        for tab_i in range(len(TempOutput)):
            if TempOutput[tab_i].count(modified_positive)>TempOutput[tab_i].count(negative_sign):
                Output.append(modified_positive)
            else:
                Output.append(negative_sign)

        print(len(VotingList))
        print(len(VotingList[0]))

        print(len(Output))
        print(len(Y_Testing))
        print(len(Y_Testing[Y_Testing==modified_positive]))
        print(len(Y_Testing[Y_Testing==negative_sign]))
        for tab in range(len(Output)):
            if int(Output[tab])==modified_positive and Output[tab]==int(Y_Testing[tab]):
                ac_positive += 1
            if int(Output[tab])==negative_sign and Output[tab]==int(Y_Testing[tab]):
                ac_negative += 1
        ACC = float(ac_positive+ac_negative)/len(Output)
        auc = roc_auc_score(Y_Testing,Output)
        g_mean=np.sqrt(float(ac_positive*ac_negative)/(len(Y_Testing[Y_Testing==modified_positive])*len(Y_Testing[Y_Testing==negative_sign])))
        print(Y_Testing)
        print(Y_Testing==modified_positive)
        print(Y_Testing==negative_sign)
        cross_folder_acc_list.append(ACC*100)
        cross_folder_auc_list.append(auc*100)
        cross_folder_g_mean_list.append(g_mean*100)


    return np.average(cross_folder_acc_list),np.average(cross_folder_auc_list),np.average(cross_folder_g_mean_list)
    """




if __name__=="__main__":
    global positive_sign,negative_sign,count_positive,count_negative,out_put_path,modified_positive,modified_negative
    positive_sign=-1
    negative_sign=1
    modified_positive = 0
    modified_negative = 1
    count_positive=0
    count_negative=0
    with open("Setting.txt")as fin:
        vallist = fin.readlines()
        baseurl, base_url = vallist[0].split(":")

    base_url = base_url.strip()
    input_data_path = os.path.join(base_url,"SingleEvent")

    filenamelist=filter(lambda a:os.path.isfile(os.path.join(input_data_path,a)),os.listdir(input_data_path))

    #total_args = list(sys.argv)
    #total_args.pop(0)Negative_Data_Samples_Index
    #processing_file = str(total_args[0])+".txt"
    #print("Start->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->on "+processing_file)
    #window_size = int(total_args[1])
    #bagging_label = int(total_args[2])
    #lstm_size = int(total_args[3])
    #cross_valtab = int(total_args[4])
    #bagging_number = int(total_args[5])
    #Main(base_url,processing_file,bagging_number,cross_valtab,window_size,lstm_size,bagging_label)
    Main(base_url,"Code_Red_I.txt",1,1,10,10,50)


