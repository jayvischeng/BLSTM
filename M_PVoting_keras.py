import os
import numpy as np
from sklearn.metrics import roc_auc_score


def Run(B,W,filetype,filename,lstm_size):
    processing_folder = os.path.join(os.getcwd(),filetype+"_B_"+str(B)+"_W_"+str(W))
    filelist = os.listdir(processing_folder)
    for eachfile in filelist:
        Predict = []
        Predict_Voting = []
        mytype = "True"
        if  mytype in eachfile and filename in eachfile:
            #print(eachfile+" is processing...")
            with open(os.path.join(processing_folder,eachfile))as fin:
                True_Label = np.loadtxt(fin)
            True_Label2 = np.array(True_Label)
        else:
            continue

    for eachfile in filelist:
        mytype = "Predict"
        flag = 0
        is_begin = 0
        if  mytype in eachfile and filename in eachfile:
            #print(eachfile+" is processing...")
            with open(os.path.join(processing_folder,eachfile))as fin:

                for eachline in fin:
                    if "lstm_size: " in eachline:
                        #print(eachline)
                        if  "lstm_size: "+str(lstm_size)+"-" in eachline:
                            is_line_value = 0
                            flag = 1
                            is_begin = 1
                        else:
                            is_line_value = 0
                            flag = 0
                    else:
                        is_line_value = 1

                    if is_begin == 0:
                        continue
                    if is_line_value == 1:
                        Predict_Voting.append(eachline.strip())
                    else:
                        if len(Predict_Voting) > 1 and flag >= 1 :
                            Predict_Voting = filter(lambda a:len(a)>1,Predict_Voting)
                            Predict_Voting = map(lambda a:float(a),Predict_Voting)
                            Predict.append(Predict_Voting)
                            Predict_Voting = []
                        elif len(Predict_Voting) > 1 and flag == 0:
                            Predict_Voting = filter(lambda a:len(a)>1,Predict_Voting)
                            Predict_Voting = map(lambda a:float(a),Predict_Voting)
                            Predict.append(Predict_Voting)
                            Predict_Voting = []
                            #if len(Predict) == 50:
                            break
                            #else:
                                #continue
                        else:
                            pass


            ac_positive = 0
            ac_negative = 0
            predict_list = []
            Final_Precit = []

            for tab1 in range(len(Predict[0])):
                predict_list.append([])
                for tab2 in range(len(Predict)):
                    if Predict[tab2][tab1] < 0.5:
                        #predict_list[tab1].append(Predict[tab2][tab1]*-1)
                        predict_list[tab1].append(0)

                    elif Predict[tab2][tab1] >= 0.5:
                        #predict_list[tab1].append(Predict[tab2][tab1]*1)
                        predict_list[tab1].append(1)
                    else:
                        print(50*"*"+str(Predict[tab2][tab1]))
                        #predict_list[tab1].append(Predict[tab2][tab1]*1)
                        #predict_list[tab1].append(int(Predict[tab2][tab1]*1))
            #print((predict_list[0]))

            for tab1 in range(len(predict_list)):
                #if sum(predict_list[tab1]) > 0:
                if predict_list[tab1].count(1)>predict_list.count(0):
                    Final_Precit.append(1)
                else:
                    Final_Precit.append(0)

            for tab in range(len(True_Label)):
                if True_Label[tab]==Final_Precit[tab] and Final_Precit[tab]==0:
                    ac_positive += 1
                elif True_Label[tab]==Final_Precit[tab] and Final_Precit[tab]==1:
                    ac_negative += 1


            ACC_R = round(float(ac_negative)/Final_Precit.count(1),5)*100
            ACC_A = round(float(ac_positive)/Final_Precit.count(0),5)*100

            ACC = round(float(ac_positive+ac_negative)/len(Final_Precit),5)*100
            auc = round(roc_auc_score(True_Label,Final_Precit),5)*100
            g_mean=round(np.sqrt(float(ac_positive*ac_negative)/(len(True_Label2[True_Label2==0])*len(True_Label2[True_Label2==1]))),5)*100

            precision = ACC_A
            recall = round(float(ac_positive)/list(True_Label).count(0),5)*100

            f1_score = (2*precision*recall)/(precision+recall)
        else:
            continue
        return ACC_R,ACC_A,ACC,auc,g_mean,f1_score

def Return_Evaluation(B,W,filetype,filename,lstm_size_list):
    ACC_R_list = []
    ACC_A_list = []
    ACC_list = []
    auc_list = []
    g_mean_list = []
    f1_score_list = []
    for lstm_size in lstm_size_list:
        print("lstm_size is "+str(lstm_size))
        ACC_R,ACC_A,ACC,auc,g_mean,f1_score = Run(B,W,filetype,filename,lstm_size)
        ACC_R_list.append(ACC_R)
        ACC_A_list.append(ACC_A)
        ACC_list.append(ACC)
        auc_list.append(auc)
        g_mean_list.append(g_mean)
        f1_score_list.append(f1_score)

    auc_max = max(auc_list)
    index_max = auc_list.index(auc_max)
    ACC_R_max = ACC_R_list[index_max]
    ACC_A_max = ACC_A_list[index_max]
    ACC_max = ACC_list[index_max]
    g_mean_max = g_mean_list[index_max]
    f1_score_max = f1_score_list[index_max]

    #print("The ACC_Regular result is: "+str(round(ACC_R_max,5))+"%")
    #print("The ACC_Anomaly result is: "+str(round(ACC_A_max,5))+"%")
    #print("The ACC result is: "+str(round(ACC_max,5))+"%")
    #print("The auc result is: "+str(round(auc_max,5))+"%")
    #print("The g_mean result is: "+str(round(g_mean_max,5))+"%")
    #print("The f1_score result is: "+str(round(f1_score_max,5))+"%")
    #print("The MAX Index is: "+str(auc_list.index(auc_max)))

    return ACC_R_max,ACC_A_max,ACC_max,auc_max,g_mean_max,f1_score_max
if __name__=='__main__':


    filetype = "MultiEvent_Keras"
    bagging_label_list = [50,100,150,200,250]
    window_size_list = [10,30,50]
    filename = "Nimda"
    ACC_R_L = []
    ACC_A_L = []
    ACC_L = []
    auc_L = []
    g_mean_L = []
    f1_score_L = []
    lstm_size_list = [i for i in range(10,41,10)]

    for tab1 in range(len(bagging_label_list)):
        B = bagging_label_list[tab1]
        print("--------------------------------------------------------------------------------------Bagging Size is :"+str(B))
        ACC_R_L.append([])
        ACC_A_L.append([])
        ACC_L.append([])
        auc_L.append([])
        g_mean_L.append([])
        f1_score_L.append([])
        for tab2 in range(len(window_size_list)):
            W = window_size_list[tab2]
            print("---------------------------------------------------------Window Size is :"+str(W))

            ACC_R_max,ACC_A_max,ACC_max,auc_max,g_mean_max,f1_score_max = Return_Evaluation(B,W,filetype,filename,lstm_size_list)

            ACC_R_L[tab1].append(ACC_R_max)
            ACC_A_L[tab1].append(ACC_A_max)
            ACC_L[tab1].append(ACC_max)
            auc_L[tab1].append(auc_max)
            g_mean_L[tab1].append(g_mean_max)
            f1_score_L[tab1].append(f1_score_max)

        auc_L_Max = max(auc_L[tab1])
        index_Max = auc_L[tab1].index(auc_L_Max)
        ACC_R_L_Max= ACC_R_L[tab1][index_Max]
        ACC_A_L_Max = ACC_A_L[tab1][index_Max]
        ACC_L_Max= ACC_L[tab1][index_Max]
        g_mean_L_Max= g_mean_L[tab1][index_Max]
        f1_score_L_Max= f1_score_L[tab1][index_Max]

        ACC_R_L[tab1]=ACC_R_L_Max
        ACC_A_L[tab1]=ACC_A_L_Max
        ACC_L[tab1]=ACC_L_Max
        auc_L[tab1]=auc_L_Max
        g_mean_L[tab1]=g_mean_L_Max
        f1_score_L[tab1]=f1_score_L_Max
    print("///////////////////////////////////////////////////////////////////////////////////////////////////")
    print(ACC_R_L)
    print(ACC_A_L)
    print(ACC_L)
    print(auc_L)
    print(g_mean_L)
    print(f1_score_L)

    #return ACC_R_L,ACC_A_L,ACC_L,auc_L,g_mean_L,f1_score_L

