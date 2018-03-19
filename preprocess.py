# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from PIL import Image

nb_classes = 55
master_categories = pd.read_csv("./master.tsv", header=None, delimiter="\t")
train_file = pd.read_csv("./train_master.tsv", header=None, delimiter="\t")

#file_names = train_file[0][1:1000]
#Y_train = np.asarray(train_file[1][1:1000])
file_names = train_file[0][1:]
Y_train = np.asarray(train_file[1][1:])

# フォルダ作成
for i in range(nb_classes) :
	os.mkdir('./sorted_train/'+str(i))

for i, file_name in enumerate(file_names):
	image = Image.open( './train/'+file_name)
	image.save('./sorted_train/'+str(Y_train[i])+'/'+file_name)
	if(i%100 == 0):
		print(i)






