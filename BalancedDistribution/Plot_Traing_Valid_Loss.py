import matplotlib.pyplot as plt
filename = "Loss_Ploting_40_8_c1.txt"
Loss_List_1 = []
Val_Loss_List_1 = []
with open(filename)as fin:
    for each in fin.readlines():
        if 'loss' in each:
            loss_index = each.strip().index('loss:')
            val_loss_index = each.strip().index('val_loss:')
            Loss_List_1.append(float(each.strip()[loss_index+6:loss_index+12]))
            Val_Loss_List_1.append(float(each.strip()[val_loss_index+10:val_loss_index+16]))
X1 = []
if len(Loss_List_1)==len(Val_Loss_List_1):
    for tab in range(len(Loss_List_1)):
        X1.append(tab+1)

filename = "Loss_Ploting_40_8_c2.txt"
Loss_List_2 = []
Val_Loss_List_2 = []
with open(filename)as fin:
    for each in fin.readlines():
        if 'loss' in each:
            loss_index = each.strip().index('loss:')
            val_loss_index = each.strip().index('val_loss:')
            Loss_List_2.append(float(each.strip()[loss_index+6:loss_index+12]))
            Val_Loss_List_2.append(float(each.strip()[val_loss_index+10:val_loss_index+16]))



filename = "Loss_Ploting_40_8_c3.txt"
Loss_List_3 = []
Val_Loss_List_3 = []
with open(filename)as fin:
    for each in fin.readlines():
        if 'loss' in each:
            loss_index = each.strip().index('loss:')
            val_loss_index = each.strip().index('val_loss:')
            Loss_List_3.append(float(each.strip()[loss_index+6:loss_index+12]))
            Val_Loss_List_3.append(float(each.strip()[val_loss_index+10:val_loss_index+16]))

Loss_List = []
Val_Loss_List = []
for tab in range(len(X1)):
    temp = float(Loss_List_1[tab] + Loss_List_2[tab] + Loss_List_3[tab])/3
    Loss_List.append(temp)

for tab in range(len(X1)):
    temp = float(Val_Loss_List_1[tab] + Val_Loss_List_2[tab] + Val_Loss_List_3[tab])/3
    Val_Loss_List.append(temp)



plt.plot(X1,Loss_List,'b--',markersize=8,linewidth=3,label='Train Loss')
plt.plot(X1,Val_Loss_List,'gx-.',markersize=8,linewidth=3,label='Validation Loss')
plt.legend()
plt.xlabel('Epochs Num ')
plt.ylabel('Loss')
plt.grid()
plt.show()