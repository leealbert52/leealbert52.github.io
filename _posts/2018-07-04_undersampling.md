---
title: "Undersampling in Imbalanced Machine Learning"
layout: post
date: 2018-07-04
headerImage: false
tag:
- oversampling
- undersampling
- Imbalanced Machine Learning
- Index of Balanced Accuracy
star: true
category: blog
author: Albert F. Lee, Ph.D.
description: "Undersampling in Imbalanced Machine Learning"
---

Most machine learning algorithms in scikit-learn assume that, in a 2-class classification problem, the dataset is
balanced. If a class, or the minority class is highly under-represented the majority class will dominate the
learning process in the training, making it challenging to build a decent classifier.

To balance the two classes one can choose either oversampling of the minority class or undersampling of the majority
class. We have discussed resampling and oversampling of minority class in my last post. In this post I’ll explain
the idea of undersampling in machine learning when the dataset is imbalanced.

Undersampling aims to reduce the samples from majority class (in the training dataset) so that the dataset is
balanced. The idea of reducing samples runs counter-intuitive against the notion that samples are valuable and the
larger the sample the better the accuracy from a statistician's standpoint. However, in present days datasets are
large, and undersampling approach becomes very appealing. Nevertheless when you have a relatively small total samples
(train and test samples) oversampling is still preferred.

There are many undersampling algorithms in the imbalanced learn package. There is T-link or Tomek Links, combines
with a random under-sampling approach as a data reduction method. There is one-sided selection under-sampling
algorithm, to name a few.

Undersampling is a popular method in dealing with imbalanced datasets.  However, they are not without deficiency,
as they do not use all the samples.  Two newer algorithms are proposed to overcome such deficiency.  The
EasyEnsemble algorithm selects several subsets of the majority class, trains the learner using each and combines
the outputs from these learners.  BalanceCascade algorithm trains the learners sequentially, where in each step,
the majority class samples that are correctly classified by the current trained learners, are removed from further
consideration. (See Liu, Wu and Zhou. 2008)

For a given dataset some algorithms are suitable than the others and have varying degree of model performance.

One would then be concerned about the performance of such classifiers. The metrics used for model selection when
the datasets are balanced are not adequate in imbalanced learning. Most of the widely used measures, such as
Receiver Operating Characteristic (ROC), Area Under the ROC Curve (AUC) and geometric mean of class accuracies,
do not consider how dominant is the accuracy on an individual class over another. An alternative measure : Index
of Balanced Accuracy (IBA) is proposed along with Dominance Index, which measures how prevalent is the dominant
class rate with respect to other. (See Garcia, et al)

The reduced samples (the balanced majority and minority training samples) are then used to train the classifier.
In my project I use IBA, geometric mean and other metrics to compare the performance of a RandomForest classifier
using both oversampling and undersampling approaches. It turns out that the undersampling approach is better than
oversampling approach using IBA, despite the fact that the dataset is relatively small.

http://contrib.scikit-learn.org/imbalanced-learn

Exploratory Undersampling for Class-Imbalance Learning. Liu, Wu and Zhou (2008)

https://ieeexplore.ieee.org/abstract/document/4717268

Index of Balanced Accuracy: A Performance Measure for Skewed Class Distributions. Garcia, et al

https://leealbert52.github.io
