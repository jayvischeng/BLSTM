#!/bin/bash
clear
echo "Good morning, world."
read -p "Press any key to START..."



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
        array_lstm_size=("10" "15" "20" "25" "30" "35" "40" "45" "50" "55" "60" "65" "70" "75" "80" "85" "90" "95" "100")
        array_cv_tab=("1" "2" "3")
        for s in ${array_lstm_size[@]}
            do
            for c in  ${array_cv_tab[@]}
                do
                for n in  $( seq 1 $VALUE )
                    do
                    #echo $BASE_URL"/Main_Keras_SingleEvent.py" $s $c $n &
                    python $BASE_URL"/Main_Keras_SingleEvent.py" $PROCESSING_FILE $WINDOW_SIZE $BAGGING_LABEL $s $c $n &
                    wait
                    done
                done
            done


    fi
    done
wait
read -p "Press any key to END..."




