import os
import numpy as np
import collections
import matplotlib.pyplot as plt

def Main(filename,bagging_label_list,Evaluation_List):

    Evaluation_Dict = collections.defaultdict(dict)

    Method_Dict = collections.defaultdict(list)
    Method_list = ["KNN","IGBB","DT","SVM","LR"]


    for each_eval in Evaluation_List:
        Evaluation_Dict[each_eval] = collections.defaultdict(list)
    print(Evaluation_Dict)

    for bagging_label in bagging_label_list:

        processingfolder = "Output_MultiEvent_B_"+str(bagging_label)
        filelist = os.listdir(processingfolder)

        for each_eval in Evaluation_List:
            for eachfile in filelist:
                if "Bagging" in eachfile or "SubFeature" in eachfile:continue
                if filename in eachfile and each_eval in eachfile:

                    for each_method in Method_list:
                        Method_Dict[each_method] = []
                    pass
                else:continue

                with open(os.path.join(processingfolder,eachfile)) as fin:
                    vallines = fin.readlines()
                    for tab in range(len(vallines)):
                        temp = float((vallines[tab].split(':')[-1].replace(",","")).strip())
                        if temp < 1:
                            Evaluation_Dict[each_eval][Method_list[tab]].append(temp*100)
                        else:
                            Evaluation_Dict[each_eval][Method_list[tab]].append(temp)
    return Evaluation_Dict



if __name__=='__main__':

    filename = "Slammer"
    bagging_label_list = [50,100,150,200,250]
    Evaluation_List = ["ACC_R","ACC_A","ACC_L","Auc","G_mean","F1_score"]

    #color_list = ['r','g','b','c','m','y']
    color_dict = {"KNN":'c',"IGBB":'r',"DT":'y',"LR":'m',"SVM":'g',"BLSTM":'b'}
    Evaluation_Dict = Main(filename,bagging_label_list,Evaluation_List)

    BLSTM = collections.defaultdict(list)

    if filename == "Code_Red_I":
        BLSTM["ACC_R"] = [100.0, 99.919, 99.64699999999999, 99.655, 99.816]
        BLSTM["ACC_A"] = [28.693999999999996, 30.686000000000003, 37.153999999999996, 40.777, 38.157000000000004]
        BLSTM["ACC_L"] = [81.533, 83.292, 87.627, 89.336, 88.048]
        BLSTM["Auc"] = [90.025, 90.592, 91.554, 92.477, 92.625]
        BLSTM["G_mean"] = [89.471, 90.185, 91.438, 92.404, 92.46799999999999]
        BLSTM["F1_score"] = [44.59259949958817, 46.86897587271761, 53.59978875045941, 57.270150703937375, 54.927561564958104]
    elif filename == "Nimda":
        BLSTM["ACC_R"] = [99.195, 98.681, 87.298, 86.631, 97.069]
        BLSTM["ACC_A"] = [68.458, 68.857, 76.972, 77.974, 71.14200000000001]
        BLSTM["ACC_L"] = [79.744, 80.002, 82.331, 82.578, 81.604]
        BLSTM["Auc"] = [81.96, 82.138, 82.621, 82.705, 83.38]
        BLSTM["G_mean"] = [80.10000000000001, 80.416, 82.589, 82.699, 82.211]
        BLSTM["F1_score"] = [81.05120367147455, 81.17824309741425, 80.73597592966719, 80.73606647904748, 82.18563077836491]
    elif filename == "Slammer":
        BLSTM["ACC_R"] = [99.854, 99.899, 99.884, 99.87, 99.74199999999999]
        BLSTM["ACC_A"] = [82.825, 87.834, 87.661, 89.464, 90.316]
        BLSTM["ACC_L"] = [97.416, 98.265, 98.22800000000001, 98.49, 98.515]
        BLSTM["Auc"] = [98.08200000000001, 98.7, 98.63300000000001, 98.737, 98.346]
        BLSTM["G_mean"] = [98.078, 98.698, 98.63199999999999, 98.737, 98.346]
        BLSTM["F1_score"] = [90.17560883030866, 93.20295036476844, 93.05969540968162, 94.01867931213731, 94.05869741722873]

    for each_eval in Evaluation_List:
        Evaluation_Dict[each_eval]["BLSTM"] = BLSTM[each_eval]
    print(BLSTM)
    print(Evaluation_Dict)

    for each_evalk,each_evalv in Evaluation_Dict.items():
        title = each_evalk
        X = bagging_label_list
        #Y_list = [[] for i in range(len(each_evalv))]
        count = 0
        plt.figure()
        for eachMethod,eachList in each_evalv.items():
            #plt.subplot(1,)
            Y = eachList
            print("For "+eachMethod+": the max "+each_evalk+ " is "+str(round(np.max(Y),1)))
            plt.plot(X,Y,color_dict[eachMethod]+"s-",label=eachMethod)
            plt.xlim(40,260)
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
        plt.xlabel('Bagging Size')
        #plt.show()
        plt.savefig(os.path.join(os.path.join(os.getcwd(),"Images"),filename+'_'+title+".png"))
#import matplotlib.rcsetup as rcsetup
#print(rcsetup.all_backends)
#import matplotlib
#print(matplotlib.matplotlib_fname())