import os
import numpy as np
input_file_list = os.listdir(os.getcwd())
global positive_sign,modified_positive,negative_sign, input_data_path_training, input_data_path_testing, out_put_path
# os.chdir("/home/grads/mcheng223/IGBB")
positive_sign = -1
negative_sign = 1
modified_positive = 0
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
for eachfile in input_file_list:
    if not ('.txt' in eachfile and 'IB_' in eachfile):continue
    Data_ = LoadData(os.getcwd(),eachfile)
    print(len(Data_))
    PositiveIndex = returnPositiveIndex(Data_,positive_sign)
    NegativeIndex = returnNegativeIndex(Data_,negative_sign)
    New_NegativeIndex = []

    for tab in range(len(PositiveIndex)):
        New_NegativeIndex.append(PositiveIndex[0]-tab-1)
    New_Data_Index = np.append(New_NegativeIndex,PositiveIndex,axis=0)
    New_Data_Index.sort()
    print(len(New_Data_Index))
    New_Data = Data_[New_Data_Index,:]
    with open(eachfile.replace("IB_","B_"),"w")as fout:
        for tab1 in range(len(New_Data)):
            for tab2 in range(len(New_Data[0])):
                fout.write(str(round(New_Data[tab1][tab2],2)))
                if tab2 < len(New_Data[0])-1:
                    fout.write(',')
                else:
                    fout.write('\n')
