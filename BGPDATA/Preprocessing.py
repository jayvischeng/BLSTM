import os
import time
start = time.time()
#from svmutil import *
from numpy import *
from InformationGain import *


def LoadData(filename):
    global input_data_path,out_put_path
    y_svmformat, x_svm_format = svm_read_problem(os.path.join(input_data_path,filename))

    y_svmformat=np.array(y_svmformat)
    y_svmformat[y_svmformat==-1]=positive_sign#Positive is -1

    Data=[]
    for tab in range(len(x_svm_format)):
        Data.append([])
        temp=[]
        for k,v in x_svm_format[tab].items():
            temp.append(float(v))
        Data[tab].extend(temp)
        Data[tab].append(int(y_svmformat[tab]))
    Data=np.array(Data)


    with open(filename+".txt","w")as fout:
        for tab1 in range(len(Data)):
            for tab2 in range(len(Data[0])-1):
                fout.write(str(Data[tab1][tab2]))
                fout.write(',')
            fout.write(str(Data[tab1][tab2+1]))
            fout.write('\n')

    return Data
def LoadData2(input_data_path,filename):
    count_positive = 0
    count_negative = 0
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
                    count_negative += 1
                else:
                    count_positive += 1
                    val[-1]=positive_sign
                try:
                    val=map(lambda a:float(a),val)
                except:
                    val=map(lambda a:str(a),val)

                val[-1]=int(val[-1])
                Data.append(val)
        Data=np.array(Data)
        print("IR is "+str(float(count_negative/float(count_positive))))
        return Data


if __name__ == '__main__':
    global input_data_path,out_put_path,positive_sign,negative_sign
    positive_sign=-1
    negative_sign=1
    input_data_path = os.getcwd()
    #LoadData("BGP_DATA")
    LoadData2(os.getcwd(),"Code_Red_I.txt")
    #LoadData("Nimda")
    #LoadData("Slammer")