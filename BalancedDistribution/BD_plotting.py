import os
import numpy as np
import collections
import matplotlib.pyplot as plt

def Main(filename,window_size_list,Evaluation_List):

    Evaluation_Dict = collections.defaultdict(dict)
    Evaluation_Dict2 = collections.defaultdict(dict)

    Method_Dict = collections.defaultdict(list)
    Method_list = ["KNN","AdaBoost","LSTM","DT","SVM","LR"]


    for each_eval in Evaluation_List:
        Evaluation_Dict[each_eval] = collections.defaultdict(list)
    for each_eval in Evaluation_List:
        Evaluation_Dict2[each_eval] = collections.defaultdict(list)
    #print(Evaluation_Dict)

    for window_size in window_size_list:

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
                        print(vallines[tab])
                        print(temp_value)
                        if temp_value < 1:
                            Evaluation_Dict[each_eval][temp_method].append(temp_value*100)
                        else:
                            Evaluation_Dict[each_eval][temp_method].append(temp_value)
        print(Evaluation_Dict)
        for each_eval in Evaluation_List:
            for eachMethod in Method_list:
                Evaluation_Dict2[each_eval][eachMethod].append(max(Evaluation_Dict[each_eval][eachMethod]))
        print(Evaluation_Dict2)
    return Evaluation_Dict2



if __name__=='__main__':

    filename = "B_C_N_S"
    window_size_list = [10,30]
    Evaluation_List = ["ACC_R","ACC_A","ACC_L","Auc","G_mean","F1_score"]

    #color_list = ['r','g','b','c','m','y']
    color_dict = {"KNN":'c',"AdaBoost":'r',"DT":'y',"LR":'m',"SVM":'g',"LSTM":'b'}
    Evaluation_Dict = Main(filename,window_size_list,Evaluation_List)
    if not os.path.isdir(os.path.join(os.getcwd(),"Images")):
        os.makedirs(os.path.join(os.getcwd(),"Images"))


    for each_evalk,each_evalv in Evaluation_Dict.items():
        title = each_evalk
        X = window_size_list
        #Y_list = [[] for i in range(len(each_evalv))]
        count = 0
        plt.figure()
        for eachMethod,eachList in each_evalv.items():
            #plt.subplot(1,)
            Y = eachList
            print("For "+eachMethod+": the max "+each_evalk+ " is "+str(round(np.max(Y),1)))
            print(X)
            print(Y)
            plt.plot(X,Y,color_dict[eachMethod]+"s-",label=eachMethod)
            plt.xlim(10,70)
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