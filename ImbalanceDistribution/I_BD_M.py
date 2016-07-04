#_author_by_MC@20160424
import os
import time
start = time.time()
import numpy as np
from numpy import *
from sklearn import tree
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
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,preprocessing,linear_model
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
def get_auc(arr_score, arr_label, pos_label):
    score_label_list = []
    for index in xrange(len(arr_score)):
        score_label_list.append((float(arr_score[index]), int(arr_label[index])))
    score_label_list_sorted = sorted(score_label_list, key = lambda line:line[0], reverse = True)

    fp, tp = 0, 0
    lastfp, lasttp = 0, 0
    A = 0
    lastscore = None

    for score_label in score_label_list_sorted:
        score, label = score_label[:2]
        if score != lastscore:
            A += trapezoid_area(fp, lastfp, tp, lasttp)
            lastscore = score
            lastfp, lasttp = fp, tp
        if label == pos_label:
            tp += 1
        else:
            fp += 1

    A += trapezoid_area(fp, lastfp, tp, lasttp)
    A /= (fp * tp)
    return A
def trapezoid_area(x1, x2, y1, y2):
    delta = abs(x2 - x1)
    return delta * 0.5 * (y1 + y2)
def LoadData(input_data_path,filename):
    with open(os.path.join(input_data_path,filename)) as fin:
        if filename == 'sonar.dat':
            negative_flag = 'M'
        else:
            negative_flag = '1.0'
        Data=[]

        for each in fin:
            if '@' in each:continue
            val=each.split(",")
            if len(val)>0 or val[-1].strip()=="negative" or val[-1].strip()=="positive":
                #print(each)
                if val[-1].strip()== negative_flag:
                    val[-1] = negative_sign
                else:
                    val[-1] = positive_sign
                try:
                    val=map(lambda a:float(a),val)
                except:
                    val=map(lambda a:str(a),val)

                val[-1]=int(val[-1])
                Data.append(val)
        Data=np.array(Data)
        return Data

def Compute_average_list(mylist):
    temp = 0
    for i in range(len(mylist)):
        temp += float(mylist[i])
    return float(temp)/len(mylist)

def reConstruction(window_size,data,label):
    newdata = []
    newlabel = []
    L = len(data)
    D = len(data[0])
    interval = 1

    index = 0
    newdata_count = 0
    initial_value = -999
    while index+window_size < L:
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

def Manipulation(X_Taining,Y_Training,time_scale_size):
    window_size = len(X_Taining[0])
    TEMP_XTraining = []
    N = window_size/time_scale_size
    for tab1 in range(len(X_Taining)):
        TEMP_XTraining.append([])
        for tab2 in range(N):
            TEMP_Value = np.zeros((1,len(X_Taining[0][0])))
            for tab3 in range(time_scale_size):
                TEMP_Value += X_Taining[tab1][tab2*time_scale_size+tab3]
            TEMP_Value = TEMP_Value/time_scale_size
            TEMP_XTraining[tab1].extend(list(TEMP_Value[0]))
    return TEMP_XTraining,Y_Training
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

def Convergge(X_Training,Y_Training,time_scale_size):
    window_size = len(X_Training[0])
    TEMP_XData = []
    N = window_size/time_scale_size
    for tab1 in range(len(X_Training)):
        TEMP_XData.append([])
        for tab2 in range(N):
            TEMP_Value = np.zeros((1,len(X_Training[0][0])))
            for tab3 in range(time_scale_size):
                TEMP_Value += X_Training[tab1][tab2*time_scale_size+tab3]
            TEMP_Value = TEMP_Value/time_scale_size
            TEMP_XData[tab1].append(list(TEMP_Value[0]))
    return  np.array(TEMP_XData),Y_Training

def Main(Method_Dict,filename,bagging_label,window_size_label,window_size=0,time_scale_size=0):
    global positive_sign,negative_sign,input_data_path_training,input_data_path_testing,output_data_path
    scaler = preprocessing.Normalizer()

    if eachfile == "IB_Code_Red_I.txt":
        Training_Data = LoadData(input_data_path_training,"IB_N_S_Training.txt")
    elif eachfile == "IB_Nimda.txt":
        Training_Data = LoadData(input_data_path_training,"IB_C_S_Training.txt")
    elif eachfile == "IB_Slammer.txt":
        Training_Data = LoadData(input_data_path_training,"IB_C_N_Training.txt")


    Testing_Data=LoadData(input_data_path,filename)

    Pos_Data=Training_Data[Training_Data[:,-1]==positive_sign]
    Neg_Data=Training_Data[Training_Data[:,-1]==negative_sign]
    PositiveIndex = returnPositiveIndex(Training_Data,positive_sign)
    NegativeIndex = returnNegativeIndex(Training_Data,negative_sign)

    Auc_list = {}
    ACC_R_list = {}
    ACC_A_list = {}
    G_mean_list = {}
    ACC_list = {}
    F1_list = {}

    Temp_Bagging_Auc_list = {}
    Temp_Bagging_ACC_R_list = {}
    Temp_Bagging_ACC_A_list = {}

    Temp_Bagging_G_mean_list = {}

    Temp_Bagging_ACC_list = {}
    Temp_Bagging_F1_list = {}

    Deviation_ACC_R_list={}
    Deviation_ACC_A_list={}
    Deviation_Auc_list={}
    Deviation_G_mean_list={}
    Deviation_ACC_list = {}
    Deviation_F1_list = {}

    Temp_SubFeature_Auc_list = {}
    Temp_SubFeature_G_mean_list = {}
    Temp_SubFeature_ACC_R_list = {}
    Temp_SubFeature_ACC_A_list = {}
    Temp_SubFeature_ACC_list = {}
    Temp_SubFeature_F1_list = {}

    for eachMethod,methodLabel in Method_Dict.items():
        print(eachMethod+" is running...")
        Auc_list[eachMethod] = []
        ACC_R_list[eachMethod] = []
        ACC_A_list[eachMethod] = []
        G_mean_list[eachMethod] = []
        ACC_list[eachMethod] = []
        F1_list[eachMethod] = []

        for bagging_number in range(bagging_label, 252, 5000):
            print("The Bagging Number is " + str(bagging_number) + "...")
            Temp_Bagging_ACC_R_list[eachMethod + "_BN_" + str(bagging_number)] = []
            Temp_Bagging_ACC_A_list[eachMethod + "_BN_" + str(bagging_number)] = []
            Temp_Bagging_Auc_list[eachMethod + "_BN_" + str(bagging_number)] = []
            Temp_Bagging_G_mean_list[eachMethod + "_BN_" + str(bagging_number)] = []
            Temp_Bagging_ACC_list[eachMethod + "_BN_" + str(bagging_number)] = []
            Temp_Bagging_F1_list[eachMethod + "_BN_" + str(bagging_number)] = []

            Iterations = 1
            cross_folder_auc_list = []
            cross_folder_acc_r_list = []
            cross_folder_acc_a_list = []
            cross_folder_g_mean_list = []
            cross_folder_acc_list = []
            cross_folder_f1_list = []
            cross_folder = 1
            for iteration_count in range(Iterations):
                print(str(iteration_count+1)+"th iterations is running...")
                for tab_cross in range(cross_folder):


                    Training_Data_Pos = Training_Data[Training_Data[:,-1]==positive_sign]
                    Training_Data_Neg = Training_Data[Training_Data[:,-1]==negative_sign]

                    print(str(tab_cross + 1) + "th Cross Validation is running and the training size is " + str( \
                        len(Training_Data)) + ", testing size is " + str(len(Testing_Data)) + "......")
                    # positive_=Training_Data[Training_Data[:,-1]==positive_sign]
                    # negative_=Training_Data[Training_Data[:,-1]==negative_sign]
                    # print("IR000000000000000000000 is :"+str(float(len(negative_))/len(positive_)))
                    if window_size_label == "true":
                        if methodLabel != 0 :
                            (X_Testing_1, Y_Testing_1) = reConstruction(window_size, Testing_Data[:, :-1],Testing_Data[:, -1])
                            X_Testing, Y_Testing = Manipulation(X_Testing_1, Y_Testing_1, time_scale_size)
                            X_Testing = scaler.fit_transform(X_Testing)
                        else:
                            (X_Testing_1, Y_Testing_1) = reConstruction(window_size, scaler.fit_transform(Testing_Data[:, :-1]),Testing_Data[:, -1])
                            X_Testing, Y_Testing = Convergge(X_Testing_1, Y_Testing_1, time_scale_size)

                    else :
                        X_Testing = Testing_Data[:, :-1]
                        Y_Testing = Testing_Data[:, -1]
                        X_Testing = scaler.fit_transform(X_Testing)
                    # print(str(tab_cross+1)+"th cross validation is running and the training size is "+str(len(X_Training))+", testing size is "+str(len(X_Testing))+"......")

                    #print("IR is :" + str(float(len(negative_)) / len(positive_)))
                    lstm_size = 30



                    #(X_Testing, Y_Testing) = reConstruction(window_size, scaler.fit_transform(X_Testing), Y_Testing)

                    VotingList = []
                    for t in range(bagging_number):
                        VotingList.append([])
                        print(str(t+1)+" th base leaner is running......and positive testing is "+str(list(Y_Testing).count(positive_sign))+" and negative testing is "+str(list(Y_Testing).count(negative_sign)))
                        #Positive_Data_Samples=RANDOM.sample(Positive_Training_Data,int(len(Positive_Training_Data)))
                        Positive_Data_Samples_Index = PositiveIndex
                        Negative_Data_Samples_Index = RANDOM.sample(NegativeIndex, len(Positive_Data_Samples_Index))

                        TrainingSamples_Index = np.concatenate((Negative_Data_Samples_Index, Positive_Data_Samples_Index))
                        TrainingSamples_Index.sort()
                        TrainingSamples = Training_Data[TrainingSamples_Index,:]

                        Y_Training_0 = TrainingSamples[:, -1]
                        X_Training_0 = TrainingSamples[:, :-1]
                        if window_size_label == "true":
                            if methodLabel == 0:
                                np.random.seed(1337)  # for reproducibility
                                batch_size = 200
                                (X_Training_1,Y_Training_1) = reConstruction(window_size,scaler.fit_transform(Training_Data[:,:-1]),Training_Data[:,-1])
                                X_Training,Y_Training = Convergge(X_Training_1,Y_Training_1,time_scale_size)

                                print(X_Training.shape)
                                lstm_object = LSTM(lstm_size,input_length=len(X_Training[0]),input_dim=33)
                                print('Build model...'+'Window Size is '+str(window_size)+' LSTM Size is '+str(lstm_size) + " Time Scale is "+ str(time_scale_size))
                                model = Sequential()

                                model.add(lstm_object)#X.shape is (samples, timesteps, dimension)
                                model.add(Dense(output_dim=1))
                                model.add(Activation("sigmoid"))
                                model.compile( optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


                                model.fit(X_Training, Y_Training, batch_size=batch_size,nb_epoch=20)

                                result = model.predict(X_Testing,batch_size=batch_size)
                                VotingList[t].extend(result)
                                del model
                                break
                            else:
                                (X_Training_1, Y_Training_1) = reConstruction(window_size, X_Training_0,Y_Training_0)
                                X_Training, Y_Training = Manipulation(X_Training_1, Y_Training_1, time_scale_size)
                                X_Training = scaler.fit_transform(X_Training)


                                print('Window Size is ' + str(window_size) + " Time Scale is " + str(time_scale_size))
                                if methodLabel == 1:
                                    # clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
                                    clf = AdaBoostClassifier()
                                elif methodLabel == 2:
                                    clf = tree.DecisionTreeClassifier()
                                elif methodLabel == 3:
                                    clf = svm.SVC(kernel="rbf", gamma=0.001, C=1000)
                                elif methodLabel == 4:
                                    clf = linear_model.LogisticRegression()
                                elif methodLabel == 5:
                                    clf = KNeighborsClassifier(3)


                                clf.fit(X_Training, Y_Training)
                                result = clf.predict(X_Testing)
                                VotingList[t].extend(result)
                        else:
                            Y_Training = TrainingSamples[:, -1]
                            X_Training = TrainingSamples[:, :-1]

                            X_Training = scaler.fit_transform(X_Training)

                            if methodLabel==1:
                                #clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
                                clf = AdaBoostClassifier()
                            elif methodLabel==2:
                                clf=tree.DecisionTreeClassifier()
                            elif methodLabel==3:
                                clf = svm.SVC(kernel="rbf", gamma=0.001,C=1000)
                            elif methodLabel==4:
                                clf = linear_model.LogisticRegression()
                            elif methodLabel==5:
                                clf = KNeighborsClassifier(3)

                            clf.fit(X_Training,Y_Training)
                            result=clf.predict(X_Testing)
                            VotingList[t].extend(result)

                    TempOutput = [[] for i in range(len(VotingList[0]))]
                    Output = []
                    if len(VotingList[0])==len(Y_Testing):
                        for tab_i in range(len(VotingList[0])):
                            for tab_j in range(len(VotingList)):
                                TempOutput[tab_i].append(int(round(VotingList[tab_j][tab_i])))
                        for tab_i in range(len(TempOutput)):
                            if TempOutput[tab_i].count(positive_sign) > TempOutput[tab_i].count(negative_sign):
                                Output.append(positive_sign)
                            else:
                                Output.append(negative_sign)

                    else:print("Error!")

                    ac_positive=0
                    ac_negative=0
                    for tab in range(len(Output)):
                        if Output[tab]==positive_sign and Output[tab]==int(Y_Testing[tab]):
                            ac_positive += 1
                        if Output[tab]==negative_sign and Output[tab]==int(Y_Testing[tab]):
                            ac_negative += 1
                    try:
                        ACC_R = float(ac_negative)/Output.count(negative_sign)
                    except:
                        ACC_R = float(ac_negative)*100/(Output.count(negative_sign)+1)
                    try:
                        ACC_A = float(ac_positive)/Output.count(positive_sign)
                    except:
                        ACC_A = float(ac_positive)*100/(Output.count(positive_sign)+1)

                    auc = roc_auc_score(Y_Testing,Output)
                    g_mean=np.sqrt(float(ac_positive*ac_negative)/(len(np.array(Y_Testing)[np.array(Y_Testing)==positive_sign])*len(np.array(Y_Testing)[np.array(Y_Testing)==negative_sign])))
                    precision = ACC_A
                    recall = float(ac_positive)/list(Y_Testing).count(positive_sign)
                    ACC = round(float(ac_positive+ac_negative)/len(Output),5)
                    try:
                        f1_score = round((2*precision*recall)/(precision+recall),5)
                    except:
                        f1_score = round((2 * precision * recall) / (precision + recall+1), 5)


                    cross_folder_acc_r_list.append(ACC_R*100)
                    cross_folder_acc_a_list.append(ACC_A*100)
                    cross_folder_auc_list.append(auc*100)
                    cross_folder_g_mean_list.append(g_mean*100)
                    cross_folder_acc_list.append(ACC*100)
                    cross_folder_f1_list.append(f1_score*100)

                for tab1 in range(int(len(cross_folder_auc_list)/cross_folder)):
                    temp_acc_r=0.0
                    temp_acc_a=0.0
                    temp_auc=0.0
                    temp_g_mean=0.0
                    temp_acc=0.0
                    temp_f1_score=0.0
                    for tab2 in range(cross_folder):
                        temp_acc_r += cross_folder_acc_r_list[tab1*cross_folder+tab2]
                        temp_acc_a += cross_folder_acc_a_list[tab1*cross_folder+tab2]
                        temp_auc += cross_folder_auc_list[tab1*cross_folder+tab2]
                        temp_g_mean += cross_folder_g_mean_list[tab1*cross_folder+tab2]
                        temp_acc += cross_folder_acc_list[tab1*cross_folder+tab2]
                        temp_f1_score += cross_folder_f1_list[tab1*cross_folder+tab2]

                    temp_acc_r=temp_acc_r/float(cross_folder)
                    temp_acc_a=temp_acc_a/float(cross_folder)
                    temp_auc=temp_auc/float(cross_folder)
                    temp_g_mean=temp_g_mean/float(cross_folder)
                    temp_acc=temp_acc/float(cross_folder)
                    temp_f1_score=temp_f1_score/float(cross_folder)


                Temp_Bagging_ACC_R_list[eachMethod + "_BN_" + str(bagging_number)].append(temp_acc_r)
                Temp_Bagging_ACC_A_list[eachMethod + "_BN_" + str(bagging_number)].append(temp_acc_a)
                Temp_Bagging_Auc_list[eachMethod + "_BN_" + str(bagging_number)].append(temp_auc)
                Temp_Bagging_G_mean_list[eachMethod + "_BN_" + str(bagging_number)].append(temp_g_mean)
                Temp_Bagging_ACC_list[eachMethod + "_BN_" + str(bagging_number)].append(temp_acc)
                Temp_Bagging_F1_list[eachMethod + "_BN_" + str(bagging_number)].append(temp_f1_score)

            ACC_R_list[eachMethod].append(Compute_average_list(Temp_Bagging_ACC_R_list[eachMethod + "_BN_" + str(bagging_number)]))
            ACC_A_list[eachMethod].append(Compute_average_list(Temp_Bagging_ACC_A_list[eachMethod + "_BN_" + str(bagging_number)]))
            ACC_list[eachMethod].append(Compute_average_list(Temp_Bagging_Auc_list[eachMethod + "_BN_" + str(bagging_number)]))
            Auc_list[eachMethod].append(Compute_average_list(Temp_Bagging_G_mean_list[eachMethod + "_BN_" + str(bagging_number)]))
            G_mean_list[eachMethod].append(Compute_average_list(Temp_Bagging_ACC_list[eachMethod + "_BN_" + str(bagging_number)]))
            F1_list[eachMethod].append(Compute_average_list(Temp_Bagging_F1_list[eachMethod + "_BN_" + str(bagging_number)]))



    Write_Out(output_data_path,filename,time_scale_size,ACC_R_list,"ACC_Regular")
    Write_Out(output_data_path,filename,time_scale_size,ACC_A_list,"ACC_Anomaly")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_ACC_R_list,"SubFeature_ACC_Regular",Deviation_ACC_R_list)
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_ACC_A_list,"SubFeature_ACC_Anomaly",Deviation_ACC_A_list)
    Write_Out(output_data_path,filename,time_scale_size,Auc_list,"Auc")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_Auc_list,"SubFeature_Auc",Deviation_Auc_list)
    Write_Out(output_data_path,filename,time_scale_size,G_mean_list,"G_mean")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_G_mean_list,"SubFeature_G_mean",Deviation_G_mean_list)
    Write_Out(output_data_path,filename,time_scale_size,ACC_list,"ACC")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_ACC_list,"SubFeature_ACC",Deviation_ACC_list)
    Write_Out(output_data_path,filename,time_scale_size,F1_list,"F1_score")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_F1_list,"SubFeature_F1_score",Deviation_F1_list)
def Write_Out(filefolderpath,filename,time_scale_size,Result_List,Tag,Result_List_back=[]):
    with open(os.path.join(filefolderpath,filename.split('.')[0]+"_Info_"+Tag+"_List.txt"),"a")as fout:
        fout.write("-----------(time scale: "+str(time_scale_size)+ ")-----------------\n")
        for eachk,eachv in Result_List.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            if len(Result_List_back) > 0:
                fout.write(str(Result_List_back[eachk]))
            fout.write('\n')

def get_all_subfactors(number):
    temp_list = []
    temp_list.append(1)
    temp_list.append(2)
    for i in range(3,number):
        if number%i == 0 :
            temp_list.append(i)
    temp_list.append(number)
    return temp_list

if __name__=='__main__':
    global positive_sign,negative_sign,input_data_path_training,input_data_path_testing,output_data_path
    #os.chdir("/home/grads/mcheng223/IGBB")
    positive_sign=0
    negative_sign=1
    input_data_path =os.getcwd()
    input_data_path_training =os.path.join(input_data_path,"Training")
    input_data_path_testing =input_data_path
    window_size_label_list = ['true','false']
    window_size_list = [60]

    #window_size_list = [60]

    bagging_label = 50
    filenamelist=os.listdir(input_data_path)

    #Method_Dict={"LSTM":0}
    #Method_Dict={"LSTM":0,"AdaBoost":1,"DT":2,"SVM":3,"LR":4,"KNN":5}

    for eachfile in filenamelist:
        if  not eachfile=='IB_Slammer.txt':continue
        if '.py' in eachfile or '.DS_' in eachfile: continue
        if '.txt' in eachfile:
            pass
        else:
            continue
        print(eachfile+ " is processing--------------------------------------------------------------------------------------------- ")
        for window_size_label in window_size_label_list:
            if window_size_label == 'true':
                Method_Dict = {"LSTM": 0}
                #Method_Dict = {"LSTM": 0, "AdaBoost": 1, "DT": 2, "SVM": 3, "LR": 4, "KNN": 5}
                for window_size in window_size_list:
                    time_scale_size_list = get_all_subfactors(window_size)
                    for time_scale_size in time_scale_size_list:
                        output_data_path = os.path.join(os.getcwd(),'Window_Size_'+str(window_size)+'_Multi_For_'+eachfile.split('.')[0]+'_LSTM_'+str(bagging_label))
                        if not os.path.isdir(output_data_path):
                            os.makedirs(output_data_path)
                        Main(Method_Dict,eachfile,bagging_label,window_size_label,window_size,time_scale_size)

            else:
                continue
                Method_Dict = {"AdaBoost": 1, "DT": 2, "SVM": 3, "LR": 4, "KNN": 5}
                output_data_path = os.path.join(os.getcwd(),'Traditional')
                if not os.path.isdir(output_data_path):
                    os.makedirs(output_data_path)
                Main(Method_Dict,eachfile,bagging_label,window_size_label)

    print(time.time()-start)

