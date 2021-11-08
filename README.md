# Non-linear Feature Selection (NFS)


__this is the repository containing a series of codes for an efficient and precise non-linear feature selection.
First, we train an NN model on the dataset using all features (33 features), then we compute the gradient of each 
feature with respect to the outcome (here disease diagnosis). The magnitude of the gradient shows the importance 
of features. Features will be sorted based on their significance scores. Then, we start removing features with the 
lowest scores one at a time and repeat the experiment using the remained features. 

![](https://github.com/sadafkabir/NFS/blob/master/helper/Figure_1.png)

Results are shown in the two following figures. Figure 1 demonstrates the significance score of features. As we
can observe from this figure, feature #9 has the least score and be removed first. Then, feature #10 is the 
second least important feature and will be removed after feature #9. 
We continued removing features in that order, and compute the classification accuracy. Figure 2 illustrates the 
disease classification accuracy after removing each features. Results show that, the non-linear feature selection 
effectively improves the accuracy while reducing the number of features. 
Best results is obtained with only 9 features out of original 33 features. We successfully identified the most
important features for the dermatology dataset and removed 25 unnecessary features.__

![](https://github.com/sadafkabir/NFS/blob/master/helper/Figure_2.png)
