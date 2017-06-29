import numpy as np 
import xlwt
    
#write to excel 
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")

sheet1.write(0, 0, "Epoch")
sheet1.write(0, 1, "loss")
sheet1.write(0, 2, "accuracy")

#parsing 
filename = 'LSTM_model_4_result_text.rtf'
with open(filename) as f: 
    content = f.readlines()
content = [x.strip() for x in content] 

#write epoch to the first column 
epoch = np.arange(1,101).tolist()
print(epoch)
epoch_list = [x for x in np.arange(1,101)]
print(epoch_list)
i = 0 
for n in epoch:
    i += 1
    sheet1.write(i, 0, n)

j = 0
for i in np.arange(8,208,2):
    j += 1
    loss = content[i].split(':')[1].split()[0]
    accuracy = content[i].split(':')[2].strip()[:-1]
    sheet1.write(j, 1, loss)
    sheet1.write(j, 2, accuracy)

book.save("lstm_model_4.xls")
