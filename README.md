# STS-NN
The implementation of our paper "Deep Spatial-Temporal Structure Learning for Rumor Detection on Twitter" 

# Requirements
python 3.6.6  
numpy==1.17.2  
scipy==1.3.1  
pytorch==1.1.0  
scikit-learn==0.21.3  
# How to use
## Dataset
The main directory contains the directories of two Twitter datasets: twitter15 and twitter16. In each directory, there are:  
- twitter15(6)/label.txt: this file provide the labels of source tweets in a format like:'label: source tweet ID';  
- twitter15(6)/data.strnn.vol5000.txt : this file provide the temporal and spitial information of message propagations in a format like: 'source tweets ID\t index-of-the-current-tweet\t index-of-parent-tweet-list(split by ' ')\t index-of-perior-time-tweet(split by ' ')\t text-length\t list-of-index-word-belong-to-current-tweet'
- twitter15(6)/data.strnn.vol5000.f*.et(tc)*.txt: this file provide the temporal and spitial information of message propagation in the early siustuation as the formation like data.strnn.vol5000.txt file.
- nfold/RNNtrainSet_twitter15(6)*_tree.txt, RNNtestSet_twitter15(6)*_tree.txt : this files provide the traing and testing ID.  
These datasets are preprocessed according to our requirement and original datasets can be available at [here](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0)

## Training & Testing
sh run.sh 0 twitter15\[twitter16\]

## Early Detection
sh run.sh 0 twitter15(6) '' '' '' '' 0(60,120,240,480,720,1440,2160)  
sh run.sh 0 twitter15(6) '' '' '' '' '' 0(10,20,40,60,80,200,300)  
# Citation
If you find the code is useful for your research, please cite this paper:  
<pre><code>@inproceedings{huang2020deep,
author = {Huang, Qi and Zhou, Chuan and Wu, Jia and Liu, Luchen and Wang, Bin},
title = {Deep Spatial-Temporal Structure Learning for Rumor Detection on Twitter},
doi = {}
}</code></pre>

