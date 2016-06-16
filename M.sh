#!/bin/bash
clear
echo "Good morning, world."
read -p "Press any key to START..."

#cat  M_.txt|while read line; do
#NAME = awk 'BEGIN {split("'"$line"'",arr);print arr[1]}'
#VALUE =  awk 'BEGIN {split("'"$line"'",arr);print arr[2]}'
#done
#while read NAME VALUE; do echo $NAME $VALUE; done < M_.txt
cat Setting.txt|while read line
    do
    NAME=$(echo $line | cut -d \: -f 1)
    VALUE=$(echo $line | cut -d \: -f 2)
    if [[ $NAME = *baseurl* ]];then
        BASE_URL=$VALUE
    fi
    if [[ $NAME = *processingfile* ]];then
        PROCESSING_FILE=$VALUE
    fi
    if [[ $NAME = *window_size* ]];then
        WINDOW_SIZE=$VALUE
    fi
    if [[ $NAME = *bagging* ]];then
        BAGGING_LABEL=$VALUE
        array_lstm_size=("10" "15" "20" "25" "30" "35" "40" "45" "50" "55" "60" "70" "80" "90" "100")
        for s in ${array_lstm_size[@]}
            do
            for n in  $( seq 1 $VALUE )
                do
                #echo $BASE_URL"/Main_Keras_MultiEvent.py"  $s $n &
                python $BASE_URL"/Multi_keras.py"  $PROCESSING_FILE $WINDOW_SIZE $BAGGING_LABEL $s $n &
                wait
                done
            done


    fi
    done
wait
read -p "Press any key to END..."

#echo "Do you wish to install this program?"
#select yn in "Yes" "No"; do
    #case $yn in
        #Yes ) make install; break;;
        #No ) exit;;
    #esac
#done
