----

### Introduction to Data Science

This course provides an introduction to applied data science including data preparation, data analysis, factor analysis, statistical inference, predictive modeling, and data visualization.

The goal of data science is to extract information from a data set and transform it into an understandable structure for further use.

An emphasis will be placed on understanding the fundamentals using scripting languages and interactive methods to learn course concepts. Problems and data sets are selected from a broad range of disciplines of interest to students, faculty, and industry partners.

Lectures are augmented with hands-on tutorials using Jupyter Notebooks. Laboratory assignments will be completed using Python and related data science packages: NumPy, Pandas, SciPy, StatsModels, SciKit-Learn, and MatPlotLib.

2-2-3 (class hours/week, laboratory hours/week, credits)

Prerequisites: MA-262 Probability and Statistics; programming maturity, and the ability to program in Python.  

ABET: Math/Science, Engineering Topics.

Outcomes:   
- Understand the basic process of data science.
- The ability to identify, load, and prepare a data set for a given problem.  
- The ability to analyze a data set including the ability to understand which data attributes (dimensions) affect the outcome.  
- The ability to perform basic data analysis and statistical inference.  
- The ability to perform supervised learning of prediction models.  
- The ability to perform unsupervised learning.  
- The ability to perform data visualization and report generation.  
- The ability to assess the quality of predictions and inferences.  
- The ability to apply methods to real world data sets.  

Tools: Python and related packages for data analysis, machine learning, and visualization. Jupyter Notebooks.  

References:  

[An Introduction to Statistical Learning: with Applications in R. Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. 2015 Edition, Springer.](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf)

[Python Data Science Handbook, Jake VanderPlas, O'Reilly.](https://jakevdp.github.io/PythonDataScienceHandbook/)

[Data Science from Scratch, Joel Grus, O'Reilly](http://shop.oreilly.com/product/0636920033400.do)

[Hands-On Machine Learning with Scikit-Learn and TensorFlow
Concepts, Tools, and Techniques to Build Intelligent Systems, Aurélien Géron. O'Reilly Media.](http://shop.oreilly.com/product/0636920052289.do)

[Mining of Massive Datasets. Anand Rajaraman and Jeffrey David Ullman.](http://infolab.stanford.edu/~ullman/mmds.html)

Trevor Hastie, Robert Tibshirani and Jerome Friedman, The Elements of Statistical Learning. Springer, 2009.

---

### Week 1: Intro to Data Science, data science programming in Python  

#### Lecture:    
1. [Introduction to Data Science](slides/01_intro_data_science.pdf)

2. Data Science end-to-end:   
- [Back Pain](https://github.com/jayurbain/BackPain/blob/master/Back%20Pain%20Data%20Analysis.ipynb)    
- [Kaggle Titanic](https://www.kaggle.com/niklasdonges/end-to-end-project-with-python)    
- [Building Energy Data](https://github.com/WillKoehrsen/machine-learning-project-walkthrough)  
- [Donor's Choice](https://www.kaggle.com/codename007/a-very-extensive-end-to-end-project-donorschoose)  
- [Vehicle Detection](https://github.com/jayurbain/CarND-Vehicle-Detection)  

3. [Python for Data Science](slides/02_python_data_science.pdf)  
- Reading: Python Data Science Handbook (PDSH) Ch. 1    
- Reading optional: Data Science from Scratch (DSfS) Ch. 1   

4. [Introduction to Git and GitHub](slides/00_git_github.pdf)
-  Reference: [git - the simple guide](http://rogerdudler.github.io/git-guide/)   

#### Lab Notebooks:
- [Using Jupyter Notebooks](labs/lab_0_python/lab_0_jupyter.ipynb)  
- [Python Programming for Data Science](labs/lab_0_python/lab_0_python.ipynb) *Submission required*
- [Python Programming Style](labs/lab_0_python/python_programming_style.ipynb) *Optional*   
- [Dates and Time](<notebooks/2&#32;-&#32;Dates&#32;and&#32;TIme.ipynb>)  
- [Python Objects, Map, Lambda, and List Comprehensions](<notebooks/3&#32;-&#32;Python&#32;Objects&#32;Map&#32;Lambda&#32;List&#32;Comprehensions.ipynb>) *Submission required*     

*Note: Initiate walkthrough of hands-on notebooks with students, let them complete submissions on their own.*  

Outcomes addressed in week 1:   
- Understand the basic process of data science

---

### Week 2: NumPy Stack, Exploratory Data Analysis

#### Lecture:   

1. Lecture with Hands-on Notebooks:   
- [Python Numpy](notebooks/4&#32;-&#32;Python&#32;Numpy.ipynb) *Submission required*   
- [Python Numpy Aggregates](notebooks/5&#32;-&#32;Python&#32;Numpy&#32;Aggregates.ipynb)
- Reading: PDSH Ch. 2  
- Reading: DSfS Ch. 2  

2. [Exploratory Data Analysis, Pandas Dataframe](slides/03_eda_data.pdf)  
- Reading: PDSH Ch. 3  
- Reading: DSfS Ch. 10  

#### Hands-on Notebooks:
- [Pandas Data Manipulation](notebooks/6&#32;-&#32;Pandas&#32;Data&#32;Manipulation.ipynb)   
- [Python Reading and Writing CSV files](notebooks/7&#32;-&#32;Python&#32;Reading&#32;and&#32;Writing&#32;CSV&#32;files.ipynb)  
- [Data Visualization](notebooks/8&#32;-&#32;Data&#32;Visualization.ipynb)  
- [Diabetes Notebook](http://localhost:8888/notebooks/DataScienceIntro/DataScienceIntro/notebooks/glucose_analysis.ipynb)  

#### Lab Notebooks:
- [NumPy Stack](labs/lab_1_numpy_stack/lab_1_numpy_stack.ipynb) *Submission required*  
- [Stanford Low Back Pain Data Analysis](labs/lab_2_eda/lab_2_eda_backpain.ipynb) *Submission required*

Outcomes addressed in week 2:  
- Understand the basic process of data science    
- The ability to identify, load, and prepare a data set for a given problem.  

---

### Week 3: Probability and Statistical Inference, Visualization   

#### Lecture:   

1. [Probability, Stats, and Visualization](slides/04_eda_stats_viz.pdf)  
- Reading: PDSH Ch. 4  
- Reading: DSfS Ch. 5, 6  

2. [Visualization Tools](slides/04_viz_methods_tools.pdf)  *Add state-of-the-art visualization; Tableau, d3.js, etc.*

#### Lab Notebooks:
- [Data Visualization](notebooks/8&#32;-&#32;Data&#32;Visualization.ipynb)   
- [EDA Visualization](labs/Lab3_eda_viz/edaviz.ipynb#) *Submission required*

Outcomes addressed in week 3:   
- Understand the basic process of data science  
- The ability to identify, load, and prepare a data set for a given problem.  
- The ability to perform data visualization and report generation.  
- The ability to perform basic data analysis and statistical inference.  

---

#### Week 4: Linear Regression, Multivariate Regression  

#### Lecture:

1. [Linear Regression 1](slides/08_linear_regression.pdf)
- Reading: PDSH Ch. 5 p. 331-375, 390-399   
- Reading: An Introduction to Statistical Learning: with Applications in R (ISLR) Ch. 1, 2  

2. [Linear Regression Notebook](notebooks/08_linear_regression.ipynb) *Use for second lecture*  
- [Linear Regression 2](slides/08_linear_regression.pdf)   
- [Gradient Descent notebook](notebooks/GradientDescent.ipynb)  
- Reading: ISLR Ch. 3  
- Reading: PDSH Ch. 5 p. 359-375  

#### Lab Notebooks:  

- [Introduction to Machine Learning with Scikit Learn](labs/Lab3_LinearRegression/Introduction&#32;to&#32;Machine&#32;Learning&#32;with&#32;SciKit&#32;Learn.ipynb)    
- [Supervised Learning Linear Regression](labs/Lab3_LinearRegression/Supervised&#32;Learning&#32;-&#32;&#32;Linear&#32;Regression.ipynb) *Submission required*

Outcomes addressed in week 4:   
- The ability to analyze a data set including the ability to understand which data attributes (dimensions) affect the outcome.  
- The ability to perform basic data analysis and statistical inference.  
- The ability to perform supervised learning of prediction models.
- The ability to perform data visualization and report generation.   
- The ability to apply methods to real world data sets.

---

### Week 5: Introduction to Machine Learning, KNN, Model Evaluation and Metrics. Logistic Regression

#### Lecture:

1. [Introduction to Machine Learning with KNN](slides/06_machine_learning_knn.pdf)  
- Reading: ISLR Ch. 4.6.5  

2. [Logistic Regression Classification](slides/09_logistic_regression_classification.pdf)  
- Reading: ISLR Ch. 4  
- Midterm review

#### Lab Notebooks:   
- [Supervised Learning - Logistic Regression](labs/Lab5_Logistic_Regression/Supervised&#32;Learning&#32;-&#32;Logistic&#32;Regression.ipynb)   *Submission required*   

Outcomes addressed in week 5:   
- The ability to perform data visualization and report generation.  
- The ability to assess the quality of predictions and inferences.  
- The ability to apply methods to real world data sets.  
- The ability to perform supervised learning of prediction models.  

---

#### Week 6: Midterm

#### Lecture:
1. [Model Evaluation and Metrics](slides/07_model_evaluation_and_metrics.pdf)   
- [Scikit-learn ROC Curve notebook](notebooks/plot_roc.ipynb)  
- Reading: PDSH Ch. 5 p. 331-375, 390-399   
- Reading: ISLR Ch. 5

2. Midterm   
- [Midterm review study guide](handouts/Midterm-Review.pdf)

#### Lab Notebooks:   
- [Supervised Learning - Logistic Regression continued](labs/Lab5_Logistic_Regression/Supervised&#32;Learning&#32;-&#32;Logistic&#32;Regression.ipynb)   *2-week lab, Submission required*   

Outcomes addressed in week 6:
- The ability to identify, load, and prepare a data set for a given problem.  
- The ability to analyze a data set including the ability to understand which data attributes (dimensions) affect the outcome.  
- The ability to perform supervised learning of prediction models.  
- The ability to perform data visualization and report generation.  
- The ability to assess the quality of predictions and inferences.  
- The ability to apply methods to real world data sets.

---

#### Week 7: Bayesian Methods, Decision Trees  

#### Lecture:
1. [Naive Bayes Classification](slides/06_naive_bayes.pdf)  
- Reading: PDSH Ch. 5 p. 382-389   

2. [Decision Trees](slides/08_decision_trees.pdf)   
- Reading: PDSH Ch. 5 p. 421-432  
- Reading: ISLR Ch. 8   

#### Lab Notebooks:   
- [Decision Trees](labs/Lab7_DecisionTrees/Decision&#32;Trees.ipynb) *Submission required*   
- Reading: PDSH Ch. 5 p. 421-432   

Outcomes addressed in week 8:  
- The ability to identify, load, and prepare a data set for a given problem.  
- The ability to analyze a data set including the ability to understand which data attributes (dimensions) affect the outcome.  
- The ability to perform supervised learning of prediction models.  
- The ability to perform data visualization and report generation.  
- The ability to assess the quality of predictions and inferences.  
- The ability to apply methods to real world data sets.

#### Week 8: Unsupervised learning, clustering, dimensionality reduction

#### Lecture:

1. [Clustering - K-Means](slides/12_clustering.pdf)  
- Reading: PDSH Ch. 5 p. 462-475   
- Reading: ISLR Ch. 10.5.2  

2. [Clustering - Hierarchical, Probabilistic](slides/12_clustering.pdf)  
- Reading: ISLR Ch. 10.1, 10.3, 10.5.1    

#### Lab Notebooks:   
[K-Means Clustering](labs/Lab8_Clustering/K-Means.ipynb) *Submission required*   

[Introduce Data Science Project]()  

Outcomes addressed in week 9:
- The ability to identify, load, and prepare a data set for a given problem.   
- The ability to perform unsupervised learning.  
- The ability to perform data visualization and report generation.   
- The ability to apply methods to real world data sets.

---

#### Week 9: Dimensionality reduction, collaborative filtering

#### Lecture:

1. [Dimensionality Reduction](slides/09_imensionality_reduction.pdf)  
- Reading: MMDS Ch. 9        

2. [Collaborative Filtering](slides/12_collaborative_filtering.pdf)
- Reading: PDSH Ch. 5 p. 433-444

#### Lab Notebooks:     
- [SVD](labs/Lab6_Classification_PCA/Singular&#32;Value&#32;Decomposition.ipynb)   
- [PCA](labs/Lab6_Classification_PCA/classification-and-pca-lab.ipynb)   

[Data Science Project]()  

Outcomes addressed in week 9:   
- The ability to perform unsupervised learning.  
- The ability to perform data visualization and report generation.   

---

#### Week 10:  Optional, advanced methods: Ensemble methods, Random Forests, Bagging, Boosting, Neural Networks
[Validation, Bagging, Boosting, Random Forests](slides/09_validation_boostrap_boosting.pdf)  

#### Lab Notebooks:  
1. [Random Forests and Boosting](labs/Lab9_DT_RF_Boosting/RF_and_Boosting.ipynb)   
- Reading: ISLR Ch. 8.2     

[Data Science Project]()   

Outcomes addressed in week 10:
- Understand the basic process of data science
- The ability to identify, load, and prepare a data set for a given problem.  
- The ability to analyze a data set including the ability to understand which data attributes (dimensions) affect the outcome.  
- The ability to perform basic data analysis and statistical inference.  
- The ability to perform supervised learning of prediction models.  
- The ability to perform unsupervised learning.  
- The ability to perform data visualization and report generation.  
- The ability to assess the quality of predictions and inferences.  
- The ability to apply methods to real world data sets.

---
<!--
#### Week 10: Advanced methods, Time Series

Potential topics:  

[Bayesian Analysis with pgmpy](labs/lab4_bayes/Learning&#32;Bayesian&#32;Networks&#32;from&#32;Data&#32;-&#32;Back&#32;Pain&#32;Dataset.ipynb) *Submission required*   


[Neural Networks](http://jayurbain.com/msoe/cs498-machinelearning/slides/neuralnetwork.pdf)

[Recommendation Systems](http://jayurbain.com/msoe/cs498-machinelearning/slides/Recommendations.pdf)

[Intro to Big Data and Spark](https://github.com/jayurbain/SparkIntro)

[Deep Learning](http://jayurbain.com/msoe/cs498-machinelearning/slides/Deep%20Learning.pdf)

[Intro to Deep Learning with TensorFlow](https://github.com/jayurbain/TensorFlowIntro)

[Simple Deep Learning with Keras and MNIST](https://github.com/jayurbain/DeepLearningIntro)

[Deep Sequence Learning](https://github.com/jayurbain/DeepSequenceLearningIntro)

[Deep NLP Intro](https://github.com/jayurbain/DeepNLPIntro)

[Recommendation Systems with Latent Factor Models][http://jayurbain.com/msoe/cs498-machinelearning/Machine%20Learning%20Lab%206%20-%20Collaborative%20Filtering.txt](http://jayurbain.com/msoe/cs498-machinelearning/Machine%20Learning%20Lab%206%20-%20Collaborative%20Filtering.txt)

Optional content notes:  
- NLP   
- NN, DL  
- SVM  
- XGBoost  
- Linear Discriminant Methods  
-->
