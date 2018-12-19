# Minist_pytorch_cpp
You must download LibTorch from pytorch1.0 version  
# Using
bash do.sh  
Remeber here U must revise LibTorch_dir in this file.  
# Change ?
From the Pytorch official website，I find there are some error!  
first is Normalization function in minist.cpp , where sub_() funciton can change the original data ,so U must replace sub_() into sub().  
