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

**Classification Technique used**:
- [Random Forest](https://link.springer.com/article/10.1023/a:1010933404324) 
- [Extremely Randomized Trees](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf)  
- [Bagging](https://www.researchgate.net/publication/45130375_Bagging_Boosting_and_Ensemble_Methods) 
for [K-Nearest Neighbours](https://dl.acm.org/doi/10.1145/3459665)
- [Stochastic Gradient Tree Boosting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885826/)  


**Evaluation Metric**: 
- [Out of bag score/Out of bag error](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html) 
- [Accuracy](https://link.springer.com/chapter/10.1007/11941439_114)


**Best Model**: The best model was a Bagging KNN model.



