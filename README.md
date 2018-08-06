---
### Issues:  
- Commit EHR project  
- More lectures/labs versus final project   
- Content review  
- Clustering lab needs work  
- Define final project  
---

### Introduction to Data Science

This course provides an introduction to applied data science including data preparation, data analysis and statistical inference, predictive modeling, factor analysis, and data visualization.

The overall goal of data science is to extract information from a data set and transform it into an understandable structure for further use.

The course will place an emphasis placed on understanding fundamentals using scripting languages and interactive methods to learn course concepts. Problems and data sets are selected from a broad range of disciplines of interest to students, faculty, and industry partners.

2 hours of weekly lectures and 2-hour labs are provided each week. Lectures are augmented with hands-on tutorials using Jupyter Notebooks. Laboratory assignments will be completed using Python and related data science packages: numpy, pandas, scipy, statsmodels, scikit-learn, and matplotlib.

Pre-requisite: MA-262 Probability and Statistics; and CS-2852 Data Structures or equivalent level of programming maturity.  

Outcomes:   
•	The ability to identify, load, and prepare a data set for a given problem.  
•	The ability to analyze a data set including the ability to understand which data attributes (dimensions) affect the outcome.  
•	The ability to perform basic data analysis and statistical inference.  
•	The ability to perform supervised learning of prediction models.  
•	The ability to perform unsupervised learning.  
•	The ability to perform data visualization and report generation.  
•	The ability to assess the quality of predictions and inferences.  
•	The ability to apply methods to real world data sets.  

Tools: Python and related packages for data analysis and visualization, Jupyter Notebooks.  

References:  

[An Introduction to Statistical Learning: with Applications in R]. Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. 2015 Edition, Springer.](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf)

[Python Data Science Handbook, Jake VanderPlas, O'Reilly.](https://jakevdp.github.io/PythonDataScienceHandbook/)

[Data Science from Scratch], Joel Grus, O'Reilly](http://shop.oreilly.com/product/0636920033400.do)

[Hands-On Machine Learning with Scikit-Learn and TensorFlow
Concepts, Tools, and Techniques to Build Intelligent Systems, Aurélien Géron. O'Reilly Media.](http://shop.oreilly.com/product/0636920052289.do)

Mining of Massive Datasets. Anand Rajaraman and Jeffrey David Ullman. http://infolab.stanford.edu/~ullman/mmds.html

Trevor Hastie, Robert Tibshirani and Jerome Friedman, The Elements of Statistical Learning. Springer, 2009.

---

### Week 1: Intro to Data Science, data science programming in Python  

#### Lecture:    
[Introduction to Data Science](slides/01_intro_data_science.pdf)

Walkthrough example: [Data Science for EHR]()  *Need to check into github*

[Python for Data Science](slides/02_python_data_science.pdf)

[Introduction to Git and GitHub](slides/00_git_github.pdf)

#### Hands-on Notebooks:  

[Dates and Time](<notebooks/2&#32;-&#32;Dates&#32;and&#32;Time.ipynb>)
[Python Objects Map Lambda List Comprehensions](<notebooks/3&#32;-&#32;Python&#32;Objects&#32;Map&#32;Lambda&#32;List&#32;Comprehensions.ipynb>)  
[Python Numpy](notebooks/4&#32;-&#32;Python&#32;Numpy.ipynb) *Submission required*

*Note: Initiate walkthrough of hands-on notebooks with students, let them complete submissions on their own.*

#### Lab Notebooks:
[Using Jupyter Notebooks](labs/lab_0_python/lab_0_python.ipynb)  
[NumPy Stack](labs/lab_0_python/lab_0_python.ipynb) *Submission required*   

---

### Week 2: Exploratory Data Analysis, Probability and Statistical Inference   

#### Lecture:   

[Exploratory Data Analysis, Pandas Dataframe](slides/03_eda_data.pdf)  

[Probability, Stats, and Visualization](slides/04_eda_stats_viz.pdf)  

*Note: Consider expanding visualization to include latest D3.js*

#### Hands-on Notebooks:  
[Python Numpy Aggregates](notebooks/5&#32;-&#32;Python&#32;Numpy&#32;Aggregates.ipynb)  
[Pandas Data Manipulation](notebooks/6&#32;-&#32;Pandas&#32;Data&#32;Manipulation.ipynb)   
[Python Reading and Writing CSV files](notebooks/7&#32;-&#32;Python&#32;Reading&#32;and&#32;Writing&#32;CSV&#32;files.ipynb)  
[Data Visualization](notebooks/8&#32;-&#32;Data&#32;Visualization.ipynb)  

#### Lab Notebooks:
[NumPy Stack](labs/lab_1_numpy_stack/lab_1_numpy_stack.ipynb) *Submission required*  
[Stanford Low Back Pain Data Analysis](labs/lab_2_eda/lab_2_eda_backpain.ipynb) *Submission required*

---

#### Week 3: Linear Regression, Multivariate Regression  

#### Lecture:

[Linear Regression](slides/08_linear_regression.pdf)

[Linear Regression Notebook](notebooks/08_linear_regression.ipynb) *Use for second lecture*

#### Lab Notebooks:  

[Introduction to Machine Learning with Scikit Learn](labs/Lab3_LinearRegression/Introduction&#32;to&#32;Machine&#32;Learning&#32;with&#32;SciKit Learn.ipynb) *Submission required*   

[Linear Regression](labs/Lab3_LinearRegression/Supervised&#32;Learning&#32;-&#32;Linear Regression.ipynb) *Submission required*  
**OR**  
[Supervised Learning Linear Regression](labs/Lab3_LinearRegression/Supervised&#32;Learning&#32;-&#32;&#32;Linear&#32;Regression.ipynb) *Submission required*

---

### Week 4: Introduction to Machine Learning, KNN, Naive Bayes

#### Lecture:

[Introduction to Machine Learning with KNN](slides/06_machine_learning_knn.pdf)  

[Naive Bayes Classification](slides/06_naive_bayes.pdf)

#### Lab Notebooks:   
[Bayesian Analysis with pgmpy](labs/lab4_bayes/Learning&#32;Bayesian&#32;Networks&#32;from&#32;Data&#32;-&#32;Back&#32;Pain&#32;Dataset.ipynb) *Submission required*   

---

#### Week 5: Logistic Regression Classification, Model Selection and Regularization  

#### Lecture:

[Model Evaluation and Metrics](slides/07_model_evaluation_and_metrics.pdf)  
[Logistic Regression Classification](slides/09_logistic_regression_classification.pdf)

#### Lab Notebooks:   
[Supervised Learning - Logistic Regression](labs/Lab5_Logistic_Regression/Supervised Learning - Logistic Regression.ipynb) *Submission required*   

---

#### Week 6: Midterm  
- Midterm   
- Review

*Consider making prior lab 2-week lab*

---

#### Week 7: Decision Trees  
K-means, hierarchical agglomerative, probabilistic

#### Lecture:
[Decision Trees](slides/08_decision_trees.pdf)   

#### Lab Notebooks:   
[Decision Trees](labs/Lab7&#32;DecisionTrees/Decision&#32;Trees.ipynb) *Submission required*   

#### Week 8: Unsupervised learning, clustering, dimensionality reduction

#### Lecture:

[Clustering - K-Means](slides/12_clustering.pdf)  

[Clustering - Hierarchical, Probabilistic](slides/12_clustering.pdf)  

#### Lab Notebooks:   
[K-Means Clustering](labs/Lab8_clustering/K-Means.ipynb) *Submission required*   *Need to finish this lab*

[PCA](classification-and-pca-lab.ipynb)   

[Introduce final project ]()  *Define*

---

#### Week 9:  Ensemble methods, Random Forests, Bagging, Boosting
[Validation, Bagging, Boosting, Random Forests](09_validation_boostrap_boosting.pdf)

#### Lab Notebooks:

[Random Forests and Boosting](RF_and_Boosting.ipynb)  *Need to finish this lab or drop*
[Work on final project ]()   *Define*   

---

#### Week 10: Advanced methods, Time Series

Potential topics:   
[Neural Networks](http://jayurbain.com/msoe/cs498-machinelearning/slides/neuralnetwork.pdf)

[Recommendation Systems](http://jayurbain.com/msoe/cs498-machinelearning/slides/Recommendations.pdf)

[Recommendation Systems with Latent Factor Models][http://jayurbain.com/msoe/cs498-machinelearning/Machine%20Learning%20Lab%206%20-%20Collaborative%20Filtering.txt](http://jayurbain.com/msoe/cs498-machinelearning/Machine%20Learning%20Lab%206%20-%20Collaborative%20Filtering.txt)

[Intro to Big Data and Spark](https://github.com/jayurbain/SparkIntro)

[Intro to Big Data and Spark](https://github.com/jayurbain/SparkIntro)

[Deep Learning](http://jayurbain.com/msoe/cs498-machinelearning/slides/Deep%20Learning.pdf)

[Intro to Deep Learning with TensorFlow](https://github.com/jayurbain/TensorFlowIntro)

[Simple Deep Learning with Keras and MNIST](https://github.com/jayurbain/DeepLearningIntro)

[Deep Sequence Learning](https://github.com/jayurbain/DeepSequenceLearningIntro)

[Deep NLP Intro](https://github.com/jayurbain/DeepNLPIntro)


Optional content notes:  
- NLP   
- NN, DL  
- SVM  
- XGBoost  
- Linear Discriminant Methods  
