import os
import collections
import matplotlib.pyplot as plt

def Main(filename,bagging_label_list,Evaluation_List):

    Evaluation_Dict = collections.defaultdict(dict)

    Method_Dict = collections.defaultdict(list)
    Method_list = ["KNN","IGBB","DT","SVM","LR"]
    for each_eval in Evaluation_List:
        Evaluation_Dict[each_eval] = Method_Dict
    #print(Evaluation_Dict)
    for bagging_label in bagging_label_list:
        processingfolder = "Output_MultiEvent_B_"+str(bagging_label)
        filelist = os.listdir(processingfolder)
        for each_eval in Evaluation_List:
            Evaluation_Dict[each_eval] = Method_Dict.copy()
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

    filename = "Nimda"
    bagging_label_list = [50,100,150,200,250]
    Evaluation_List = ["ACC_R","ACC_A","ACC_L","Auc","G_mean","F1_score"]
    color_list = ['r','g','b','c','m','y']

    Evaluation_Dict = Main(filename,bagging_label_list,Evaluation_List)

    BLSTM = collections.defaultdict(list)

    BLSTM["ACC_R"] = [84.032]
    BLSTM["ACC_A"] = [76.189]
    BLSTM["ACC_L"] = [80.445]
    BLSTM["Auc"] = [80.404]
    BLSTM["G_mean"] = [80.403]
    BLSTM["F1_score"] = [78.08843524834583]

    for each_eval in Evaluation_List:
        Evaluation_Dict[each_eval]["BLSTM"] = BLSTM[each_eval]
    print(BLSTM)
    print(Evaluation_Dict)
    for each_evalk,each_evalv in Evaluation_Dict.items():
        title = each_evalk
        X = bagging_label_list
        #Y_list = [[] for i in range(len(each_evalv))]
        count = 0
        for eachMethod,eachList in each_evalv.items():
            Y = eachList
            plt.plot(X,Y,color_list[count]+"s-",label=eachMethod)
            plt.grid()
            plt.tight_layout()
            legend = plt.legend(loc='best', shadow=True, fontsize='x-large')
            legend.get_frame().set_facecolor('#00FFCC')
            count += 1
        plt.title(title)
        plt.show()

