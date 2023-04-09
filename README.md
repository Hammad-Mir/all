# Acute Lymphoblastic Leukemia (ALL) Detection using Deep Learning models

Acute Lymphoblastic Leukemia (ALL) is a cancer of the lymphoid
line of blood cells characterized by the development of large numbers of immature lymphocytes. Extensive research has been done over the application of machine learning algorithms for detection of ALL but use of Deep Learning (DL) models is relatively scarce. In this paper, we evaluate and compare the available DL architectures with our proposed model for ALL classification and obtain better results as compared to existing ones.

Our results conclude that a simple architecture can be devised which classifies our data with more precision and accuracy. The classical architectures fail mainly due to high invariance in the dataset and thus being incapable of effective feature selection and classification. The main advantage of this model is that it is computationally efficient compared to the much complex architectures. The model also continues to grow effectively as the epochs are increased.

We also found out that the of all the optimizers used, SGD gave the best classification result in all the configurations at a learning rate of 0.0002. ReLU, Sigmoid function was used as the activation functions for the convolution network layers and Softmax function was used as the activation function for the last Fully Connected (FC) layer for classification.

The results can further be improved by combining various similar models in an ensemble technique. Also, auto-encoders can be applied for weight matrix initialization. Implementing the same model with greater number of epochs in a faster GPU system might also enhance the models functionality. Further, additional image processing techniques may also be explored for data preparation. Standardizing the image acquisition and pre-processing standards can, in effect, make the model universally applicable.
