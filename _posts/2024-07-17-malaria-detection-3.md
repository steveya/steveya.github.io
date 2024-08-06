---
layout: post
title: "Malaria Detection (Part 3: Results and Reflections)"
date: 2024-06-18
categories: [Machine Learning]
tags: [study-notes, machine-learning, data-science, cnn, deep-learning]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

The notebook for this post can be found [here](https://github.com/steveya/data-science/blob/main/notebook/malaria-detection/malaria-detection.ipynb).

## Table of Contents
1. [Model Performances](#model-performances)
2. [Understanding the Metrics](#understanding-the-metrics)
3. [Key Insights and Lessons Learned](#key-insights-and-lessons-learned)
4. [What's Next?](#whats-next)
5. [Closing Thoughts](#closing-thoughts)

Welcome to the final part of building a malaria detection model. If you've been following along, you've seen how I prepared the data and built various models to detect Malaria from cell images. How well did our models perform on a relative basis?

## Model Performances

Here's a summary of how the different models performed:

| Model | Architecture | Regularization | Data | Accuracy | Precision (P/U)* | Recall (P/U)* | F1-Score (P/U)* |
|-------|--------------|----------------|------|----------|------------------|---------------|-----------------|
| Base Model | 3 CNN, 1 ANN | Dropout | Original | 98% | 0.98 / 0.99 | 0.99 / 0.98 | 0.98 / 0.98 |
| Model 1 | 4 CNN, 2 ANN | Dropout | Original | 98% | 0.98 / 0.98 | 0.98 / 0.98 | 0.98 / 0.98 |
| Model 2 | 8 CNN, 2 ANN | Dropout, Batch Normalization | Original | 98% | 0.98 / 0.99 | 0.99 / 0.98 | 0.98 / 0.98 |
| Model 3 | 8 CNN, 2 ANN | Dropout, Batch Normalization | Augmented | 98% | 0.98 / 0.99 | 0.99 / 0.98 | 0.99 / 0.99 |
| Model 4 (VGG16) | VGG16 + 2 CNN, 2 ANN | Dropout, Batch Normalization | Original | 98% | 0.99 / 0.96 | 0.97 / 0.99 | 0.98 / 0.98 |

*P/U: Parasitized / Uninfected

I was honestly amazed by these results. Even the base model achieved 98% accuracy! However, accuracy isn't everything. Let's break down what these numbers really mean.

## Understanding the Metrics

1. **Accuracy**: This is the proportion of correct predictions (both true positives and true negatives) among the total number of cases examined. Mathematically, it's:

   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

   While 98% accuracy sounds excellent, we must look closer at false negatives in medical applications.

2. **Precision**: This tells us how many of our positive predictions were actually correct:

   $$\text{Precision} = \frac{TP}{TP + FP}$$

3. **Recall**: This is the proportion of actual positives that were correctly identified:

   $$\text{Recall} = \frac{TP}{TP + FN}$$

   In our case, a high recall for parasitized cells is crucial to avoid potential malaria cases.

4. **F1-Score**: This is the harmonic mean of precision and recall:

   $$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

   It gives us a single score that balances both precision and recall.

## Key Insights and Lessons Learned

1. **Deeper isn't always better**: My deeper models (Model 2 and 3) didn't significantly outperform the simpler ones. This taught me that complexity is only sometimes the answer. 

2. **The importance of regularization**: Techniques like Dropout and Batch Normalization helped prevent overfitting, allowing me to train deeper networks without losing generalization. Our base model performed well on the test sample because it is sufficiently regularized.

3. ** The challenge of edge cases **: Pun intended. Some misclassifications occurred when the parasites were near the cell boundaries. This highlighted the importance of diverse training data or the need for specialized architectures.

4. **Data augmentation can help with data diversity **: Once we determine the type of images likely to be misclassified, we can use data augmentation to create more of this kind of data. Model 3 used augmented data to generate images where the parasite was at the edge of the images and was able to outperform other models.

5. **Transfer learning isn't a silver bullet**: Although the VGG16-based model (Model 4) performed well, it still needs to outperform our baseline model. This taught me the importance of understanding my problem rather than relying on off-the-shelf solutions.



## What's Next?

This project has been an incredible learning experience, but there's always room for improvement. Here are some ideas I'm considering for future iterations:

1. **Exploring other architectures**: I'm curious about trying out other model architectures like ResNet or DenseNet.

2. **Focused data augmentation**: Instead of general augmentations, I could try techniques that specifically address the edge case problem.

3. **Ensemble methods**: Combining predictions from multiple models could improve overall performance.

4. **Explainable AI**: Implementing techniques like Grad-CAM to visualize what the model is "looking at" could provide valuable insights.

5. **Real-world testing**: While the results on this dataset are promising, testing on a diverse set of real-world images is crucial for practical application.

## Closing Thoughts

This journey into deep learning for malaria detection has been challenging and frustrating but ultimately incredibly rewarding. With some Python code and patience, we can create tools that potentially save lives.

This series has been informative and inspired you to tackle similar problems. Remember, every expert was once a beginner, and the key is to keep learning and experimenting.

Thanks for joining me on this adventure! I'd love to hear your thoughts, questions, or ideas for future projects. Let's continue the conversation in the comments below!