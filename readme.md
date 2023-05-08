### NURSERY
This is [model](https://learn.microsoft.com/en-us/windows/ai/windows-ml/what-is-a-machine-learning-model) 
that predicts whether a child will be admitted to a nursery school or not on the basis of a number 
of features. Suppose you want your child to be admitted to such a school
The model is important to a school since, the school can use it to objectively determine whom should admitted and who
should not. A thousand argument can go on between staffs of the school and parents if a child were to be arbitrarily 
rejected. 
[Classification models](https://www.researchgate.net/publication/319370844_Classification_Techniques_in_Machine_Learning_Applications_and_Issues)
of supervised learning in Machine learning can be used help bring about an objective way 
of admitting or refusing to admit students.

**Data Source**: [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/nursery)

**Data Features**:
- parents: usual, pretentious, great_pret
- has_nurs: proper, less_proper, improper, critical, very_crit
- form: complete, completed, incomplete, foster
- children: 1, 2, 3, more
- housing: convenient, less_conv, critical
- finance: convenient, inconv
- social: non-prob, slightly_prob, problematic
- health: recommended, priority, not_recom

**Data Target**:

- admission: not_recom, recommend, very_recom, priority, special priority 
(The target was later collapsed into only three in this work: "not recommended", "recommended without special priority", "recommended with special priority" )

#### Description of Target Variable

<img src=https://github.com/Olupaula/nursery/blob/master/nursery_images/bar_plot_of_admission_status.jpeg height='50%' width='60%'>

4320 prospective were not recommended, 4595 of the prospective students where recommended without any special priority while about 4044 prospective students were recommended with special priority.


**Classification Technique used**:
- [Random Forest](https://link.springer.com/article/10.1023/a:1010933404324) 
- [Extremely Randomized Trees](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf)  
- [Bagging](https://www.researchgate.net/publication/45130375_Bagging_Boosting_and_Ensemble_Methods) 
for [K-Nearest Neighbours](https://dl.acm.org/doi/10.1145/3459665)
- [Stochastic Gradient Tree Boosting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885826/)  


**Evaluation Metric**: 
- [Out of bag score/Out of bag error](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html) 
- [Accuracy](https://link.springer.com/chapter/10.1007/11941439_114)

**Summary Table for the Models**

|S/N  |Model Name                               |Out of Bag Score |Out of Bag Error |Test Accuracy Score|
|-----|-----------------------------------------|-----------------|-----------------|-------------------|
|  1  |  Random Forest                          |  0.9476         | 0.0524          | 0.9491            |
|  2  |  Extremely Randomized Trees             |  0.9465         | 0.0535          | 0.9541            |
|  3  |  Bagging for K-Nearest Neighbour        |  0.9508         | 0.0492          | 0.9572            |
|  4  |  Stochastic Gradient Tree Boosting      |  0.9465         | 0.0535          | 0.9479            |
|     |                                         |                 |                 |                   |

Random Forest   
1         Extremely Randomized Trees   
2    Bagging for K-Nearest Neighbour   
3  Stochastic Gradient Tree Boosting
**Best Model**: The best model was a Bagging KNN model.

[View Code on Kaggle](https://www.kaggle.com/code/oluade111/nursery-notebook)

[Use API]()


