# spam-ham-filtering
A spam and ham filtering program using Naive Bayes, Logistic Regression and Neural Networks

## ReadMe:

### Naive Bayes and Logistic Regression

Kindly execute the file using the following command:

```
python homework2.py <training-set-dir> <test-set-dir> <e> <l> <i> <k>
```

 where -   
 training-set-dir: The directory path containing folders titled 'ham' and 'spam' which contain the training data.  
 test-set-dir: The directory path containing folders titled 'ham' and 'spam' which contain the testing data.  
 e: eta value or the learning rate for Logistic Regression  
 l: lambda value for Logistic Regression  
 i: Number of iterations for Logistic Regression  
 k: Feature Selection desired size.  

### Sample Execution:
```
python homework2.py train/ test/ 0.05 1.3 100 1500
```

### Sample Output:

 Naive Bayes Accuracy with Stop Words :  94.769874477  
 Logistic Regression Accuracy with Stop Words :  93.9330543933  
 Naive Bayes Accuracy without Stop Words :  94.3514644351  
 Logistic Regression Accuracy with Stop Words :  95.8158995816  
 Naive Bayes Accuracy with Feature Selection :  92.050209205  
 Logistic Regression Accuracy with Feature Selection :  88.7029288703  

## Perceptron and Neural Networks

For Neural Networks - scikit-learn is used. No external script is needed to convert the given data to needed input format. The code implicitly does the same. The dataset path given which contains the spam and ham files is taken and these files are converted to the required input format.

Kindly execute the file using the following command:

```
python homework3.py <training-set-dir> <test-set-dir> <p_e> <p_i> <nn_e> <nn_i> <nn_hu> <nn_m>
```

 where-  
 training-set-dir: The directory path containing folders titled 'ham' and 'spam' which contain the training data.  
 test-set-dir: The directory path containing folders titled 'ham' and 'spam' which contain the testing data.  
 p_e: eta value or the learning rate for Perceptron.  
 p_i: Number of iterations for Perceptron.  
 nn_e: eta value or the learning rate for Neural Network.  
 nn_i: Number of iterations for Neural Network.  
 nn_hu: Number of hidden units for Neural Network.  
 nn_m: Momentum for Neural Network.  

Sample Execution:
```
python homework3.py data_set1/train/ data_set1/test/ 0.1 200 0.1 100 5 0.1

Perceptron Accuracy :  91.6317991632
Perceptron Accuracy without Stop Words :  90.7949790795
Neural Network Accuracy :  92.050209205
```
