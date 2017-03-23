# spam-ham-filtering
A spam and ham filtering program using Naive Bayes and Logistic Regression

## ReadMe for Homework 2.

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

## Sample Execution:
```
python homework2.py train/ test/ 0.05 1.3 100 1500
```

## Sample Output:

 Naive Bayes Accuracy with Stop Words :  94.769874477  
 Logistic Regression Accuracy with Stop Words :  93.9330543933  
 Naive Bayes Accuracy without Stop Words :  94.3514644351  
 Logistic Regression Accuracy with Stop Words :  95.8158995816  
 Naive Bayes Accuracy with Feature Selection :  92.050209205  
 Logistic Regression Accuracy with Feature Selection :  88.7029288703  
