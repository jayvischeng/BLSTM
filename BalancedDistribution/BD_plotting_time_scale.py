import os
import random
import numpy as np
import collections
import matplotlib.pyplot as plt
def get_all_subfactors(number):
    temp_list = []
    temp_list.append(1)
    temp_list.append(2)
    for i in range(3,number):
        if number%i == 0 :
            temp_list.append(i)
    temp_list.append(number)
    return temp_list
def Main(filename,window_size_list,lstm_size_list,Evaluation_List):
    global Evaluation_Dict_Max_Time_Scale,Normalization_Method
    Evaluation_Dict = collections.defaultdict(dict)
    Evaluation_DictOutPut = collections.defaultdict(dict)
    Evaluation_Dict_Max_Time_Scale = collections.defaultdict(dict)
    Evaluation_DictOutPut222 = collections.defaultdict(dict)
    Evaluation_Dict_Max_Time_Scale222 = collections.defaultdict(dict)

    for each_eval in Evaluation_List:
        Evaluation_DictOutPut222[each_eval] = collections.defaultdict(list)
    for each_eval in Evaluation_List:
        Evaluation_Dict_Max_Time_Scale222[each_eval] = collections.defaultdict(list)

    for each_eval in Evaluation_List:
        Evaluation_DictOutPut[each_eval] = collections.defaultdict(list)
    for each_eval in Evaluation_List:
        Evaluation_Dict_Max_Time_Scale[each_eval] = collections.defaultdict(list)

    Method_Dict = collections.defaultdict(list)
    Method_list = ["LSTM","AdaBoost","KNN","DT","SVM","LR"]

    for lstm_size in lstm_size_list:


        for window_size in window_size_list:
            for each_eval in Evaluation_List:
                Evaluation_Dict[each_eval] = collections.defaultdict(list)

            time_scale_size_list = get_all_subfactors(window_size)
            processingfolder = "Window_Size_"+str(window_size)+'_LS_'+str(lstm_size)+Normalization_Method
            filelist = os.listdir(processingfolder)

            for each_eval in Evaluation_List:
                for eachfile in filelist:
                    if "Bagging" in eachfile or "SubFeature" in eachfile:continue
                    if filename in eachfile and each_eval in eachfile:

                        for each_method in Method_list:
                            Method_Dict[each_method] = []
                        pass

                    else:continue
                    print(os.path.join(processingfolder,eachfile))
                    try:
                        with open(os.path.join(processingfolder,eachfile)) as fin:
                            vallines = fin.readlines()
                            for tab in range(len(vallines)):
                                if '(' in vallines[tab]:continue
                                temp_method = vallines[tab].split(':')[0].strip()
                                temp_value = float((vallines[tab].split(':')[-1].replace(",","")).strip())
                                #print(vallines[tab])
                                #print(temp_value)
                                if temp_value < 1:
                                    Evaluation_Dict[each_eval][temp_method].append(temp_value*100)
                                else:
                                    Evaluation_Dict[each_eval][temp_method].append(temp_value)

                    except:
                        pass

            for each_eval in Evaluation_List:
                for eachMethod in Method_list:
                    temp_list = Evaluation_Dict[each_eval][eachMethod]
                    try:
                        Evaluation_DictOutPut[each_eval][eachMethod].extend(temp_list)
                        Evaluation_Dict_Max_Time_Scale[each_eval][eachMethod].extend(time_scale_size_list)
                    except:
                        Evaluation_DictOutPut[each_eval][eachMethod].extend([])
                        Evaluation_Dict_Max_Time_Scale[each_eval][eachMethod].extend([])

    print("AAA")
    print(Evaluation_DictOutPut)


    for each_eval in Evaluation_List:
        for eachMethod in Method_list:
            value_list = Evaluation_DictOutPut[each_eval][eachMethod]
            value_list_time_scale = Evaluation_Dict_Max_Time_Scale[each_eval][eachMethod]
            for tab1 in range(len(time_scale_size_list)):
                out_put_value_list = []
                out_put_value_list_time_scale = []
                for tab2 in range(len(lstm_size_list)):
                    out_put_value_list.append(value_list[tab1+len(time_scale_size_list)*tab2])
                    out_put_value_list_time_scale.append(value_list_time_scale[tab1+len(time_scale_size_list)*tab2])

                Evaluation_DictOutPut222[each_eval][eachMethod].append(max(out_put_value_list))
                Evaluation_Dict_Max_Time_Scale222[each_eval][eachMethod].append(out_put_value_list_time_scale[out_put_value_list.index(max(out_put_value_list))])
    print("BBB")
    print(Evaluation_DictOutPut222)
    print("CCC")
    print(Evaluation_Dict_Max_Time_Scale222)
    return Evaluation_DictOutPut222,Evaluation_Dict_Max_Time_Scale222,time_scale_size_list



if __name__=='__main__':
    global Normalization_Method
    Normalization_Method = "_L2norm-1"
    filename = "B_C_N_S"
    if filename == "B_Code_Red_I" or filename == "B_Slammer":
        window_size_list = [10,20,30,40,50]
        y_base = 50
    else:
        if filename == "B_C_N_S":
            y_upper = 95
        else:
            y_upper = 100
        window_size_list = [50]
        y_base = 75

    lstm_size_list = [30]
    Evaluation_List = ["ACC_R","ACC_A","ACC_L","Auc","G_mean","F1_score"]

    #color_list = ['r','g','b','c','m','y']
    color_dict = {"KNN":'c',"AdaBoost":'r',"DT":'y',"LR":'m',"SVM":'g',"LSTM":'b'}
    Evaluation_Dict,Evaluation_Dict_Max_Time_Scale,time_scale_size_list = Main(filename,window_size_list,lstm_size_list,Evaluation_List)
    if not os.path.isdir(os.path.join(os.getcwd(),"Images_W_"+str(window_size_list[0])+"_T_"+Normalization_Method)):
        os.makedirs(os.path.join(os.getcwd(),"Images_W_"+str(window_size_list[0])+"_T_"+Normalization_Method))


    for each_evalk,each_evalv in Evaluation_Dict.items():
        title = each_evalk
        X = time_scale_size_list
        #Y_list = [[] for i in range(len(each_evalv))]
        plt.figure()
        signal_list = [0,1]
        count = 0
        for eachMethod,eachList in each_evalv.items():
            #plt.subplot(1,)
            Y = eachList
            Y_max_time_scale = Evaluation_Dict_Max_Time_Scale[each_evalk][eachMethod]
            print("For "+eachMethod+": the max "+each_evalk+ " is "+str(round(np.max(Y),1)))
            #print(X)
            print(Y_max_time_scale)
            plt.plot(X,Y,color_dict[eachMethod]+"-s",label=eachMethod)
            #for i, txt in enumerate(Y_max_time_scale):
                #plt.annotate('('+str(txt)+')', xy=(X[i],Y[i]),xycoords='data',xytext=(X[i],Y[i]),size = 8)


            plt.ylim(y_base,100)
            plt.grid()
            #plt.tight_layout()
            legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc='center right', ncol=3, mode="expand", borderaxespad=0.)
                                #shadow=True, fontsize='x-small')
            #legend.get_frame().set_facecolor('#00FFCC')
            count += 1
        #print("start"),
        if each_evalk=="ACC_R":
            plt.ylabel("regular precision")
        elif each_evalk=="ACC_A":
            plt.ylabel("anomaly precision")
        elif each_evalk=="ACC_L":
            plt.ylabel("accuracy")
        else:
            plt.ylabel(each_evalk)
        plt.xlabel('Time Scale(W='+str(window_size_list[0])+')')

        #plt.show()

        plt.savefig(os.path.join(os.path.join(os.getcwd(),"Images_W_"+str(window_size_list[0])+"_T_"+Normalization_Method),filename+'_'+title+".png"))
#import matplotlib.rcsetup as rcsetup
#print(rcsetup.all_backends)
#import matplotlib
#print(matplotlib.matplotlib_fname())