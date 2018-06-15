---
title: "Resampling and Over-sampling in Imbalanced Machine Learning"
layout: post
date: 2018-06-10
headerImage: false
tag:
- markdown
- resampling
- over-sampling
- Imbalanced machine Learning
star: true
category: blog
author: Albert F. Lee, Ph.D.
description: resampling and oversampling in imbalanced machine Learning
---

In this post I’ll explain the difference between resampling and oversampling in machine learning when the dataset is imbalanced.

Most machine learning algorithms in scikit-learn assumes that the dataset is balanced in a 2-class classification problem.  If a class, or the minority class is highly under-represented the majority class will dominate the learning process in the training, making it challenging to arrive at a descent classifier.


To balance the two classes one would either generate more samples or produce new samples from the minority class.  Resampling or random oversampling essentially duplicates samples from the original minority class, whereas an oversampling approach produces new samples from minority class.

In resampling samples are randomly sampled from minority class with replacement until similar proportion of minority and majority classes are achieved.  The critic of this approach is that there is no new information. The impact on the prediction accuracy is not clear.

Synthetic Minority Oversampling Technique or SMOTE, and Adaptive Synthetic oversampling or ADASYN, are two popular over-sampling techniques that produce new samples.  These two methods use the same algorithm of interpolation to produce new samples. What distinguishes these two methods is the selection of samples, as there are samples that are harder to classify than others in the minority class.

SMOTE, and its three variants, may select inliers and outliers using k-nearest neighbors concept.  Followed by a linear interpolation to produce the new samples. ADASYN, on the other hand, focuses on samples next to the original samples, which are incorrectly classified using a k-nearest neighbor classifier.

In summary, to balance a dataset one can either generate more samples or produce new samples. Resampling approach duplicates the existing samples, whereas oversampling produces new samples utilizing k-nearest neighbors concept and linear interpolation.

The augmented sample is then used as the training dataset to train the classifier. The prediction accuracy in general, is better for oversampling than resampling. In addition to over-sampling there is an under-sampling approach to balance the dataset. I’ll explain the under-sampling technique in my next post and compare the performance of a RandomForest classifier using both over-sampling and under-sampling approaches.      

http://contrib.scikit-learn.org/imbalanced-learn

https://lealbert52.github.io
