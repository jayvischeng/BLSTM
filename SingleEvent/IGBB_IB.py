#_author_by_MC@20160424
import os
import time
import math
start = time.time()
import numpy as np
import random as RANDOM
#from svmutil import *
#import seaborn as sns
#import matplotlib.pyplot as plt
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

def LoadData(input_data_path,filename):
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
                    val[-1]=negative_sign
                else:
                    val[-1]=positive_sign
                try:
                    val=map(lambda a:float(a),val)
                except:
                    val=map(lambda a:str(a),val)

                val[-1]=int(val[-1])
                Data.append(val)
        Data=np.array(Data)
        return Data


# Building weak stump function
def buildWeakStump(d,l,D,Sub_Features):
    d2=d[:,Sub_Features]
    dataMatrix = mat(d2)
    labelmatrix = mat(l).T
    m,n = shape(dataMatrix)
    numstep = 10.0
    bestStump = {}
    bestClass = mat(zeros((5,1)))
    minErr = inf
    for i in range(n):
        datamin = dataMatrix[:,i].min()
        datamax = dataMatrix[:,i].max()
        stepSize = (datamax - datamin) / numstep
        for j in range(-1,int(numstep)+1):
            for inequal in ['lt','gt']:
                threshold = datamin + float(j) * stepSize
                predict = stumpClassify(dataMatrix,i,threshold,inequal)
                err = mat(ones((m,1)))
                err[predict == labelmatrix] = 0.0
                weighted_err = D.T * err
                if weighted_err < minErr:
                    minErr = weighted_err
                    bestClass = predict.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshold
                    bestStump['ineq'] = inequal
    return bestStump, minErr, bestClass

# Use the weak stump to classify training data
def stumpClassify(datamat,dim,threshold,inequal):
    res = ones((shape(datamat)[0],1))
    if inequal == 'lt':
        res[datamat[:,dim] <= threshold] = positive_sign
    else:
        res[datamat[:,dim] > threshold] = negative_sign

    return res


def Return_Top_K_Features(data,label,W,K):

    Features=[i for i in range(len(data[0]))]
    data_copy=data.copy()
    y_=label
    Top_List=[]

    for tab in range(len(Features)):

        if len(data_copy[:,tab])==len(W):
            for i in range(len(data_copy[:,tab])):
                data_copy[:,tab][i]=W[i]*data_copy[:,tab][i]
        else:
            print("Error! Data_[Column] Not Equal to Weight")


        Top_List.append(informationGain(data_copy[:,tab],y_))

    result=(sorted(enumerate(Top_List),key=lambda a:a[1],reverse=True))
    Label=[e[0] for e in result]
    return Label[:K]


# Training
def train(data,label,Top_K,numIt = 1000,flag = 0):
    SubSpace_WeakClassifiers={"weakClassifiers":[],"subSpace":[]}
    #weakClassifiers = []
    m = shape(data)[0]
    D = mat(ones((m,1))/m)

    Sub_Features=sorted(Return_Top_K_Features(data,label,D,Top_K))

    EnsembleClassEstimate = mat(zeros((m,1)))
    Sub_Features_List=[]
    for i in range(numIt):
        #print("The "+str(i)+" th iterations...")

        bestStump, error, classEstimate = buildWeakStump(data,label,D,Sub_Features)
        #print("Error is -------------------"+str(error))
        alpha = float(0.5*log((1.0-error) / (error+1e-15)))
        bestStump['alpha'] = alpha
        #weakClassifiers.append(bestStump)
        SubSpace_WeakClassifiers["weakClassifiers"].append(bestStump)
        weightD = multiply((-1*alpha*mat(label)).T,classEstimate)
        D = multiply(D,exp(weightD))
        D = D/D.sum()
        EnsembleClassEstimate += classEstimate*alpha
        EnsembleErrors = multiply(sign(EnsembleClassEstimate)!=mat(label).T,\
                                  ones((m,1)))  #Converte to float
        errorRate = EnsembleErrors.sum()/m
        #print "total error:  ",errorRate
        if errorRate == 0.0:
            break
        Sub_Features=sorted(Return_Top_K_Features(data,label,D,Top_K))
        SubSpace_WeakClassifiers["subSpace"].append([Sub_Features])

        #if not flag==0:
            #for each_feature in Sub_Features:
                #Sub_Features_List.append(str(each_feature))
            #Sub_Features_List.append('\n')
    #if not flag==0:
        #with open("Sub_Feature_List.txt","w")as fout:
            #for each in Sub_Features_List:
                #fout.write(each)
    print("Complete...")
    #print(SubSpace_WeakClassifiers)
    return SubSpace_WeakClassifiers


# Applying adaboost classifier for a single data sample
def adaboostClassify(dataTest,classifier):
    dataMatrix = mat(dataTest)
    m = shape(dataMatrix)[0]
    EnsembleClassEstimate = mat(zeros((m,1)))
    for i in range(len(classifier["weakClassifiers"])):
        Temp = dataTest[classifier["subSpace"][i]]
        classEstimate = stumpClassify(mat(Temp),classifier["weakClassifiers"][i]['dim'],classifier["weakClassifiers"][i]['threshold'],classifier["weakClassifiers"][i]['ineq'])
        EnsembleClassEstimate += classifier["weakClassifiers"][i]['alpha']*classEstimate
        #print EnsembleClassEstimate
    return sign(EnsembleClassEstimate)

# Testing
def test(dataSet,classifier):
    label = []
    #print '\n\n\nResults: '
    for i in range(shape(dataSet)[0]):
        label.append(adaboostClassify(dataSet[i,:],classifier))
        #print('%s' %(label[0]))
    #print(label)
    return label

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




def Compute_average_list(mylist):
    temp = 0
    for i in range(len(mylist)):
        temp += float(mylist[i])
    return float(temp)/len(mylist)
def Main(Method_Dict,filename):
    #Name_Str_List = ["Code_Red_I_NimdaSlammer.txt","Code_Red_I_SlammerNimda.txt","Nimda_SlammerCode_Red_I.txt"]

    global input_data_path,out_put_path

    print(filename+" is processing......")
    Data_=LoadData(input_data_path,filename)

    Positive_Data=Data_[Data_[:,-1]==positive_sign]
    Negative_Data=Data_[Data_[:,-1]==negative_sign]
    print("IR is :"+str(float(len(Negative_Data))/len(Positive_Data)))
    cross_folder=3
    Positive_Data_Index_list=[i for i in range(len(Positive_Data))]
    Negative_Data_Index_list=[i for i in range(len(Negative_Data))]

    Method_List=[k for k,v in Method_Dict.items()]
    Plot_auc_list=[]
    Plot_g_mean_list=[]
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
        Top_K_List = []
        Total_Dimensions = len(Positive_Data[0])-1

        #for iteration_count in range(10):
        for bagging_number in range(50,152,500):
            print("The Bagging Number is "+str(bagging_number)+"...")
            Temp_Bagging_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)] = []
            Temp_Bagging_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)] = []

            Temp_Bagging_Auc_list[eachMethod+"_BN_"+str(bagging_number)] = []
            Temp_Bagging_G_mean_list[eachMethod+"_BN_"+str(bagging_number)] = []
            Temp_Bagging_ACC_list[eachMethod+"_BN_"+str(bagging_number)] = []
            Temp_Bagging_F1_list[eachMethod+"_BN_"+str(bagging_number)] = []
            Iterations = 10
            for Top_K in range(Total_Dimensions,Total_Dimensions+1,2):
                Temp_SubFeature_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] = []
                Temp_SubFeature_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] = []

                Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] = []
                Temp_SubFeature_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] = []
                Temp_SubFeature_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] = []
                Temp_SubFeature_F1_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] = []


                Deviation_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] =[]
                Deviation_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] =[]

                Deviation_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] =[]
                Deviation_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] =[]
                Deviation_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] =[]
                Deviation_F1_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] =[]

                print("The Top_K is :"+str(Top_K))
                Top_K_List.append(Top_K)

                for iteration_count in range(Iterations):
                    print(str(iteration_count+1)+"th iterations is running...")
                    cross_folder_auc_list=[]
                    cross_folder_acc_r_list=[]
                    cross_folder_acc_a_list=[]

                    cross_folder_g_mean_list=[]
                    cross_folder_acc_list=[]
                    cross_folder_f1_list=[]
                    for tab_cross in range(cross_folder):
                        Positive_Data_Index_Training=[]
                        Positive_Data_Index_Testing=[]
                        Negative_Data_Index_Training=[]
                        Negative_Data_Index_Testing=[]

                        for tab_positive in Positive_Data_Index_list:
                            if int((cross_folder-tab_cross-1)*len(Positive_Data)/cross_folder)<=tab_positive<int((cross_folder-tab_cross)*len(Positive_Data)/cross_folder):
                                Positive_Data_Index_Testing.append(tab_positive)
                            else:
                                Positive_Data_Index_Training.append(tab_positive)
                        for tab_negative in Negative_Data_Index_list:
                            if int((cross_folder-tab_cross-1)*len(Negative_Data)/cross_folder)<=tab_negative<int((cross_folder-tab_cross)*len(Negative_Data)/cross_folder):
                                Negative_Data_Index_Testing.append(tab_negative)
                            else:
                                Negative_Data_Index_Training.append(tab_negative)

                        Positive_Training_Data=np.array(Positive_Data)[Positive_Data_Index_Training]
                        Positive_Testing_Data=np.array(Positive_Data)[Positive_Data_Index_Testing]
                        Negative_Training_Data=np.array(Negative_Data)[Negative_Data_Index_Training]
                        Negative_Testing_Data=np.array(Negative_Data)[Negative_Data_Index_Testing]


                        #Features=[i for i in range(len(Positive_Training_Data[0])-1)]
                        #Sub_Features=Features[:Top_K]
                        Testing_Data=np.append(Negative_Testing_Data,Positive_Testing_Data,axis=0)

                        Y_Testing=Testing_Data[:,-1]

                        ac_positive=0
                        ac_negative=0
                        if bagging_number==1:
                            Training_Data=np.concatenate((Negative_Training_Data,Positive_Training_Data))
                            #X_Training = Training_Data[:,Sub_Features]
                            Y_Training = Training_Data[:,-1]

                            D=[1/float(len(Y_Training)) for i in range(len(Y_Training))]
                            Sub_Features=sorted(Return_Top_K_Features(Training_Data[:,:-1],Y_Training,D,Top_K))

                            X_Training = Training_Data[:,Sub_Features]
                            X_Testing=Testing_Data[:,Sub_Features]

                            if methodLabel==1:
                                #clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
                                clf = AdaBoostClassifier()
                                #classifier = train(X_Training,Y_Training,Top_K)
                                #TempList=test(X_test,classifier)
                            elif methodLabel==2:
                                clf=tree.DecisionTreeClassifier()
                            elif methodLabel==3:
                                scaler = preprocessing.StandardScaler()
                                #X_Training = scaler.fit_transform(X_Training)
                                #X_Testing = scaler.fit_transform(X_Testing)
                                clf = svm.SVC(kernel="rbf", gamma=0.001,C=1000)
                            elif methodLabel==4:
                                clf = linear_model.LogisticRegression()
                            elif methodLabel==5:
                                clf = KNeighborsClassifier(3)
                            #clf = AdaBoostClassifier()
                            #classifier = train(X_Training,Y_Training,Top_K,100)
                            #result=test(X_Testing,classifier)

                            clf.fit(X_Training,Y_Training)
                            result=clf.predict(X_Testing)

                            Output=[]
                            if len(result)==len(Y_Testing):
                                for tab in range(len(Y_Testing)):
                                    Output.append(int(result[tab]))
                            else:
                                print("Error!")

                        else:
                            VotingList=[[] for i in range(bagging_number)]
                            for t in range(bagging_number):
                                #Positive_Data_Samples=RANDOM.sample(Positive_Training_Data,int(len(Positive_Training_Data)))
                                Positive_Data_Samples=Positive_Training_Data

                                Negative_Data_Samples=RANDOM.sample(Negative_Training_Data,len(Positive_Data_Samples))

                                TrainingSamples=np.concatenate((Negative_Data_Samples,Positive_Data_Samples))
                                #X_Training=TrainingSamples[:,Sub_Features]
                                Y_Training=TrainingSamples[:,-1]

                                #D=[1/float(len(Y_Training)) for i in range(len(Y_Training))]
                                #Sub_Features=sorted(Return_Top_K_Features(TrainingSamples[:,:-1],Y_Training,D,Top_K))
                                #print(Sub_Features)
                                X_Training=TrainingSamples[:,:-1]
                                X_Testing=Testing_Data[:,:-1]

                                if methodLabel==1:
                                    #clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
                                    clf = AdaBoostClassifier()
                                    #classifier = train(X_Training,Y_Training,Top_K)
                                    #TempList=test(X_test,classifier)
                                elif methodLabel==2:
                                    clf=tree.DecisionTreeClassifier()
                                elif methodLabel==3:
                                    scaler = preprocessing.StandardScaler()
                                    X_Training = scaler.fit_transform(X_Training)
                                    X_Testing = scaler.fit_transform(X_Testing)
                                    clf = svm.SVC(kernel="rbf", gamma=0.001,C=1000)
                                elif methodLabel==4:
                                    clf = linear_model.LogisticRegression()
                                elif methodLabel==5:
                                    clf = KNeighborsClassifier(15)

                                clf.fit(X_Training, Y_Training)
                                TempList = clf.predict(X_Testing)

                                VotingList[t].extend(TempList)

                            TempOutput=[[] for i in range(len(VotingList[0]))]
                            Output=[]
                            for tab_i in range(len(VotingList[0])):
                                for tab_j in range(len(VotingList)):
                                    TempOutput[tab_i].append(VotingList[tab_j][tab_i])
                            for tab_i in range(len(TempOutput)):
                                if TempOutput[tab_i].count(positive_sign)>TempOutput[tab_i].count(negative_sign):
                                    Output.append(positive_sign)
                                else:
                                    Output.append(negative_sign)

                        for tab in range(len(Output)):
                            if Output[tab]==positive_sign and Output[tab]==int(Y_Testing[tab]):
                                ac_positive += 1
                            if Output[tab]==negative_sign and Output[tab]==int(Y_Testing[tab]):
                                ac_negative += 1
                        ACC_R = float(ac_negative)/Output.count(negative_sign)
                        ACC_A = float(ac_positive)/Output.count(positive_sign)

                        auc = roc_auc_score(Y_Testing,Output)
                        g_mean=np.sqrt(float(ac_positive*ac_negative)/(len(np.array(Y_Testing)[np.array(Y_Testing)==positive_sign])*len(np.array(Y_Testing)[np.array(Y_Testing)==negative_sign])))


                        precision = ACC_A
                        recall = float(ac_positive)/list(Y_Testing).count(positive_sign)
                        ACC = round(float(ac_positive+ac_negative)/len(Output),5)
                        f1_score = round((2*precision*recall)/(precision+recall),5)


                        cross_folder_acc_r_list.append(ACC_R*100)
                        cross_folder_acc_a_list.append(ACC_A*100)

                        cross_folder_auc_list.append(auc*100)
                        cross_folder_g_mean_list.append(g_mean*100)
                        cross_folder_acc_list.append(ACC)
                        cross_folder_f1_list.append(f1_score)

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

                        Temp_SubFeature_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(temp_acc_r)
                        Temp_SubFeature_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(temp_acc_a)

                        Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(temp_auc)
                        Temp_SubFeature_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(temp_g_mean)
                        Temp_SubFeature_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(temp_acc)
                        Temp_SubFeature_F1_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(temp_f1_score)


                deviation_acc_r=0.0
                deviation_acc_a=0.0

                deviation_auc=0.0
                deviation_g_mean=0.0
                deviation_acc=0.0
                deviation_f1_score=0.0

                mean_acc_r=Compute_average_list(Temp_SubFeature_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])
                mean_acc_a=Compute_average_list(Temp_SubFeature_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])

                mean_auc=Compute_average_list(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])
                mean_g_mean=Compute_average_list(Temp_SubFeature_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])
                mean_acc=Compute_average_list(Temp_SubFeature_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])
                mean_f1_score=Compute_average_list(Temp_SubFeature_F1_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])

                for tab in range(len(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])):
                    temp_acc_r = Temp_SubFeature_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)][tab]
                    temp_acc_a = Temp_SubFeature_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)][tab]

                    deviation_acc_r=deviation_acc_r+((temp_acc_r-mean_acc_r)*(temp_acc_r-mean_acc_r))
                    deviation_acc_a=deviation_acc_a+((temp_acc_a-mean_acc_a)*(temp_acc_a-mean_acc_a))

                    temp_auc = Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)][tab]
                    deviation_auc=deviation_auc+((temp_auc-mean_auc)*(temp_auc-mean_auc))

                    temp_g_mean = Temp_SubFeature_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)][tab]
                    deviation_g_mean=deviation_g_mean+((temp_g_mean-mean_g_mean)*(temp_g_mean-mean_g_mean))

                    temp_acc = Temp_SubFeature_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)][tab]
                    deviation_acc=deviation_acc+((temp_acc-mean_acc)*(temp_acc-mean_acc))
                    temp_f1_score = Temp_SubFeature_F1_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)][tab]
                    deviation_f1_score=deviation_f1_score+((temp_f1_score-mean_f1_score)*(temp_f1_score-mean_f1_score))

                deviation_acc_r/=Iterations
                deviation_acc_a/=Iterations

                deviation_auc/=Iterations
                deviation_g_mean/=Iterations

                deviation_acc/=Iterations
                deviation_f1_score/=Iterations

                Deviation_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(deviation_acc_r)
                Deviation_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(deviation_acc_a)

                Deviation_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(deviation_auc)
                Deviation_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(deviation_g_mean)

                Deviation_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(deviation_acc)
                Deviation_F1_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(deviation_f1_score)


                Temp_Bagging_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)].append(Compute_average_list(Temp_SubFeature_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))
                Temp_Bagging_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)].append(Compute_average_list(Temp_SubFeature_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))

                Temp_Bagging_Auc_list[eachMethod+"_BN_"+str(bagging_number)].append(Compute_average_list(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))
                Temp_Bagging_G_mean_list[eachMethod+"_BN_"+str(bagging_number)].append(Compute_average_list(Temp_SubFeature_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))

                Temp_Bagging_ACC_list[eachMethod+"_BN_"+str(bagging_number)].append(Compute_average_list(Temp_SubFeature_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))
                Temp_Bagging_F1_list[eachMethod+"_BN_"+str(bagging_number)].append(Compute_average_list(Temp_SubFeature_F1_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))


            Auc_list[eachMethod].append(Compute_average_list(Temp_Bagging_Auc_list[eachMethod+"_BN_"+str(bagging_number)]))
            G_mean_list[eachMethod].append(Compute_average_list(Temp_Bagging_G_mean_list[eachMethod+"_BN_"+str(bagging_number)]))
            ACC_R_list[eachMethod].append(Compute_average_list(Temp_Bagging_ACC_R_list[eachMethod+"_BN_"+str(bagging_number)]))
            ACC_A_list[eachMethod].append(Compute_average_list(Temp_Bagging_ACC_A_list[eachMethod+"_BN_"+str(bagging_number)]))

            ACC_list[eachMethod].append(Compute_average_list(Temp_Bagging_ACC_list[eachMethod+"_BN_"+str(bagging_number)]))
            F1_list[eachMethod].append(Compute_average_list(Temp_Bagging_F1_list[eachMethod+"_BN_"+str(bagging_number)]))


        #print(Auc_list)
        #print(ACC_list)
        #print(G_mean_list)
        #print(Temp_Bagging_Auc_list)
        #print(Temp_SubFeature_Auc_list)

        #print("auclist.....for......"+str()+"---------MaxAUC:"+str(max(plot_auc_list))+"---------MeanAUC:"+str(sum(plot_auc_list)/float(len(plot_auc_list)))+"-----Deviation:"+str(deviation_auc))
        #print("gmeanlist.....for......"+str()+"---------MaxGmean:"+str(max(plot_g_mean_list))+"---------MeanGmean:"+str(sum(plot_g_mean_list)/float(len(plot_g_mean_list))))

        Write_Out(out_put_path,filename,ACC_R_list,"ACC_Regular")
        Write_Out(out_put_path,filename,ACC_A_list,"ACC_Anomaly")

        Write_Out(out_put_path,filename,Temp_Bagging_ACC_R_list,"Bagging_ACC_Regular")
        Write_Out(out_put_path,filename,Temp_Bagging_ACC_A_list,"Bagging_ACC_Anomaly")

        Write_Out(out_put_path,filename,Temp_SubFeature_ACC_R_list,"SubFeature_ACC_Regular",Deviation_ACC_R_list)
        Write_Out(out_put_path,filename,Temp_SubFeature_ACC_A_list,"SubFeature_ACC_Anomaly",Deviation_ACC_A_list)

        Write_Out(out_put_path,filename,Auc_list,"Auc")
        Write_Out(out_put_path,filename,Temp_Bagging_Auc_list,"Bagging_Auc")
        Write_Out(out_put_path,filename,Temp_SubFeature_Auc_list,"SubFeature_Auc",Deviation_Auc_list)

        Write_Out(out_put_path,filename,G_mean_list,"G_mean")
        Write_Out(out_put_path,filename,Temp_Bagging_G_mean_list,"Bagging_G_mean")
        Write_Out(out_put_path,filename,Temp_SubFeature_G_mean_list,"SubFeature_G_mean",Deviation_G_mean_list)

        Write_Out(out_put_path,filename,ACC_list,"ACC")
        Write_Out(out_put_path,filename,Temp_Bagging_ACC_list,"Bagging_ACC")
        Write_Out(out_put_path,filename,Temp_SubFeature_ACC_list,"SubFeature_ACC",Deviation_ACC_list)

        Write_Out(out_put_path,filename,F1_list,"F1_score")
        Write_Out(out_put_path,filename,Temp_Bagging_F1_list,"Bagging_F1_score")
        Write_Out(out_put_path,filename,Temp_SubFeature_F1_list,"SubFeature_F1_score",Deviation_F1_list)




def Write_Out(filefolderpath,filename,Result_List,Tag,Result_List_back=[]):
    with open(os.path.join(filefolderpath,filename+"Info_"+Tag+"_List.txt"),"w")as fout:
        for eachk,eachv in Result_List.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            if len(Result_List_back) > 0:
                fout.write(str(Result_List_back[eachk]))
            fout.write('\n')



if __name__=='__main__':
    global positive_sign,negative_sign,out_put_path
    #os.chdir("/home/grads/mcheng223/IGBB")
    positive_sign=-1
    negative_sign=1
    input_data_path =os.getcwd()
    bagging_label = 250

    out_put_path = os.path.join(os.getcwd(),"Output_SingleEvent_B_"+str(bagging_label))
    if not os.path.isdir(out_put_path):
        os.makedirs(out_put_path)
    filenamelist=os.listdir(input_data_path)

    #Method_Dict={"DT":1}
    Method_Dict={"IGBB":1,"DT":2,"SVM":3,"LR":4,"KNN":5}
    for eachfile in filenamelist:
        if eachfile=='BGP_DATA.txt':
            continue
        if '.py' in eachfile or '.DS_' in eachfile: continue
        if '.txt' in eachfile:
            pass
        else:
            continue
        Main(Method_Dict,eachfile,bagging_label)

    print(time.time()-start)