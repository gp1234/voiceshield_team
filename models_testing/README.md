# Interpretation

I made this class after I try my jupyter notebook. In my jupyter notebook I watch that after extracting the **MFCCs** will be enough to tests the following algorithms :

1. Random forest
2. SVM
3. MLP
4. Logistic regression

But then I realize I was getting into **data leakeage** as I am using train_test_split which splits the fake and real voices into the training data so it already knows how real voices work. Before that I tried to underfit too the models and I did see some better performance with SVM and logistic regression. Still since we have a **data leakeage** we should try a different solution which will be puting just the real data into the training split and the synthethized in the testing.
