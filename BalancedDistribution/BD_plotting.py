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
def Main(filename,window_size_list,Evaluation_List):
    global Evaluation_Dict_Max_Time_Scale
    Evaluation_Dict = collections.defaultdict(dict)
    Evaluation_DictOutPut = collections.defaultdict(dict)
    Evaluation_Dict_Max_Time_Scale = collections.defaultdict(dict)

    Method_Dict = collections.defaultdict(list)
    Method_list = ["LSTM","AdaBoost","KNN","DT","SVM","LR"]

    for each_eval in Evaluation_List:
        Evaluation_DictOutPut[each_eval] = collections.defaultdict(list)
    for each_eval in Evaluation_List:
        Evaluation_Dict_Max_Time_Scale[each_eval] = collections.defaultdict(list)


    for window_size in window_size_list:

        for each_eval in Evaluation_List:
            Evaluation_Dict[each_eval] = collections.defaultdict(list)


        time_scale_size_list = get_all_subfactors(window_size)
        processingfolder = "Window_Size_"+str(window_size)
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



        for each_eval in Evaluation_List:
            print(each_eval)
            for eachMethod in Method_list:
                print(eachMethod)
                temp_list = Evaluation_Dict[each_eval][eachMethod]
                Evaluation_DictOutPut[each_eval][eachMethod].append(max(temp_list))
                print(temp_list)
                print(max(temp_list))
                print(temp_list.index(max(temp_list)))
                Evaluation_Dict_Max_Time_Scale[each_eval][eachMethod].append(time_scale_size_list[temp_list.index(max(temp_list))])


    return Evaluation_DictOutPut,Evaluation_Dict_Max_Time_Scale



if __name__=='__main__':

    filename = "B_C_N_S"
    window_size_list = [10,20,30,40,50,60]
    Evaluation_List = ["ACC_R","ACC_A","ACC_L","Auc","G_mean","F1_score"]

    #color_list = ['r','g','b','c','m','y']
    color_dict = {"KNN":'c',"AdaBoost":'r',"DT":'y',"LR":'m',"SVM":'g',"LSTM":'b'}
    Evaluation_Dict,Evaluation_Dict_Max_Time_Scale = Main(filename,window_size_list,Evaluation_List)
    if not os.path.isdir(os.path.join(os.getcwd(),"Images")):
        os.makedirs(os.path.join(os.getcwd(),"Images"))


    for each_evalk,each_evalv in Evaluation_Dict.items():
        title = each_evalk
        X = window_size_list
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
            plt.plot(X,Y,color_dict[eachMethod]+"s-",label=eachMethod)
            for i, txt in enumerate(Y_max_time_scale):
                plt.annotate('('+str(txt)+')', xy=(X[i],Y[i]),xycoords='data',xytext=(X[i],Y[i]),size = 10)

            plt.xlim(10,70)
            plt.ylim(75,95)
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
        plt.xlabel('Window Size')
        #plt.show()

        plt.savefig(os.path.join(os.path.join(os.getcwd(),"Images"),filename+'_'+title+".png"))
#import matplotlib.rcsetup as rcsetup
#print(rcsetup.all_backends)
#import matplotlib
#print(matplotlib.matplotlib_fname())