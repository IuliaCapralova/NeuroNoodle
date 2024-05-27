
INCLUDE:
- the current features, their distributions, outliers, relevance, correlations, etc.
- the feature selection techniques that were considered, dimensionality reduction... with motivation!

### INTRO
Cancer is a leading cause of death worldwide, with breast cancer being one of the most commonly observed, especially in female patients \parencite{siegel2024cancer}. Correctly interpreting data for diagnostic purposes is an important task that medical experts tackle often. Early detection of breast cancer stands as a critical factor in improving treatment outcomes. Identifying cell anomalies in the early stages, when the cancer is localized, increases the chances of it responding to treatment and is positively correlated with the survival rate \parencite{caplan}. Having suitable diagnosis tools is of course a fundamental factor. In recent years, medical researchers have recognized the potential of machine learning methods as a promising tool for increasing diagnosis efficiency as such models are capable of analyzing vast amounts of data \parencite{MLtech}. Moreover, uncertainty measurements can be integrated into the model's predictions and serve as a guide for the medical professionals involved in making the final diagnosis. We decided to use and analyze the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI ML Repository \parencite{misc_breast_cancer_wisconsin_diagnostic_1995}. This dataset describes the characteristics of cell nuclei obtained by fine-needle aspiration of a breast mass. The features are numerical values translated from the digital image (such as the nuclei radius, area, and concavity). We will explore the data to detect any interesting patterns or correlations and potentially reduce it. We intend to define a neural network with the aim of classifying cancer cells as malignant (M) or benign (B), as well as outputting some uncertainty measures. We can compare the accuracy of our model to some pre-established models, such as logistic regression or K nearest neighbors. 

### DATA DESCRIPTION
To analyze the data we first need to understand how it was obtained. A sample of cells was taken from each patient by fine-needle aspiration of a breast mass (either malignant or benign). The digital image including multiple cell nuclei was analyzed to gather numerical information about certain features of the nuclei, such as their radius, texture, area, and so on. This was designed in a way that larger values generally correspond to a higher likelihood of malignancy.

FROM THE PAPER: "All of the features are numerically modeled such that larger values will typically indicate a higher likelihood of malignancy."

Since features were extracted from multiple cells per patient / digital image, the largest value of each feature was noted (also named the worst, as the largest suggests the highest malignancy case), the values were averaged across the cells, and the standard deviation was calculated. Therefore, each digital image was described by 30 features. These features can be organized in three groups:
- the means (e.g. radius mean, texture mean, area mean)
- the standard errors (e.g. radius se, texture se, area se)
- the worst/largest values (e.g. radius worst, texture worst, area worst)

Thus, the dataset is characterized by the original 10 features (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), but they are now represented through their mean, standard error, and worst values. Such an aggregation was most likely done to reduce the effect of random noise in individual cell measurements and represent each patient with a single datapoint that encodes information about multiple cells. This would also prove effective in the prediction process, where multiple cells of a patient would have to be analyzed to get an idea of the patient's condition. In addition, a sample of cells might only include a few malignant cells, even if the mass was marked 'malignant' as a whole. In that sense, the worst/largest value might be the most useful. (from the paper)

FROM THE PAPER: "The extreme values are the most intuitively useful for the problem at hand, since only a few malignant cells may occur in a given sample."

FEATURES: 'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
'fractal_dimension_se', 'radius_worst', 'texture_worst',
'perimeter_worst', 'area_worst', 'smoothness_worst',
'compactness_worst', 'concavity_worst', 'concave points_worst',
'symmetry_worst', 'fractal_dimension_worst'

### DATA PREPROCESSING
Choosing the right preprocessing techniques that are fitted to our data type can improve the data quality and in turn increase the accuracy and efficiency of the model. The latter is true especially when the dimensionality of the data is reduced.

##### data cleaning
QUESTION: do we remove certain datapoints, why...
The dataset is checked for missing values or any anomalies. Feature 32 is unnamed and includes NaN values, therefore it was excluded from the dataframe. The patient ID was excluded as it provides no relevant information for the task at hand. The total number of null values within the dataset is now 0. The labels are converted to numerical values, benign (B) represented as 0, malignant (M) as 1.

The final dataset includes 569 datapoints. The total number of those labeled as malignant (1) is 212 (37.3%), while those labeled as benign (0) include 357 (62.7%). The problem of imbalanced data is mitigated by ...
QUESTIONS:
- are the methods we choose insensitive to imbalanced data?
- how do we take the imbalance into account when splitting the data into a training and testing set?
- would we consider up-sampling and down-sampling and why/why not? --> for example SMOTE
- class weights?


##### data normalization
QUESTION: what are the ways in which we can normalize our data and WHY would we normalize it this way?
- we need to try multiple normalization techniques and assess their impact on the model performance!
- MinMax Scaler = scales feature values in a range (e.g. [0,1]), best for uniform dist, but also when unknown, very few outliers, when maintaining the dist is essential
- Standard Scaler = assumes variable is normally distributed, scales it down until stdev is 1 and distribution is centered around 0, basically represents features in terms of the number of stdevs that lie away from the mean
- Robust Scaler = performs well with outliers, eliminates median, scales based on IQR
- clipping? for extreme outliers (get rid of values after some limit) --> NO!
- log transformations to reduce skew?

Data normalization should be considered as a way to transform data into a format that can effectively be used by the training algorithm because features in the dataset have varying ranges. This leads to a more accurate representation of how much each feature adds to the model.

BLABLABLA: Between the two fundamental techniques for normalization, min-max scaling and and z-score normalization, min-max scaling seems to be better suited for the Wisconsin dataset and the present task. Maintaining the relationships between feature values and their distribution is important as it ensures proportional contribution of each feature in a neural network that relies upon gradient-based optimization. This can potentially lead to faster training and convergence.
ALSO: PCA kinda needs normalization? it apparently needs the values to be in the same range.

OUTLIERS: Considering the way features were encoded, outliers may actually prove to be beneficial rather than detrimental for training. The inclusion of a worst feature value is specifically designed to capture the extreme cases that are most useful for identifying malignancy. This feature group therefore inherently represents outliers and is crucial for our analysis. The same goes for the mean and standard error groups, which describe each sample's distribution. Here, outliers might also contain important information. For example, the standard error might have a wider distribution and more outliers for malignant samples, as the cells in those differ more among each other (because the sample combines benign and malignant cells). Since neural networks are quite robust to outliers if the model architecture is properly defined and data is normalized, the final choice was to include them in the training data.

##### feature selection
Four main reasons why feature selection is essential (Chen et al):
- simplify the model by reducing the number of parameters
- decrease the training time
- reduce overfitting by enhancing generalization
- avoid the curse of dimensionality

When pre-processing the data, each predictor's relevance has to be considered as certain features might prove to be completely redundant. Removing them from the training data lowers the complexity of the model, thereby making the training faster and improving the model's performance. Focusing on the features that best predict the dependent variable while ignoring the rest reduces overfitting and avoids the curse of dimensionality. This can be achieved by techniques such as correlation-based feature selection (CFS) and recursive feature elimination (RFE). Other dimensionality reduction methods such as PCA and LDA are considered.

QUESTION: how do we determine which features are valuable and which are not?
ALSO: feature extraction = feature selection + dimensionality reduction

It is good practice to first consider the correlation matrix to identify redundancy and decide on whether dimensionality reduction is needed. Understanding the relationship between variables can help us make an informed decision on a dimensionality reduction technique (e.g. PCA, LDA).

A correlation matrix is generated for each of the three groups of features (mean, standard deviation, worst). A low to moderate correlation is observed for most features, which suggests individual features might provide unique information for a prediction. However, a high correlation between certain features (e.g. area and perimeter mean) is also present. This suggests the current training data includes some unnecessary overlap which can lead to an inefficient learning process. Overfitting is a potential risk here since the model might try to accomodate the redundant data from the correlated features.
ALSO: do we need to explain how a correlation matrix is computed?

Higher correlations can be observed in two distinct areas on the heatmap. These include the clusters [radius, perimeter, area] and [compactness, concavity, concave points]. The former group is inherently correlated as such measurements directly depend on one another. The latter is also a result of how these features are defined (e.g. concave points is similar to concavity but measures only the number, rather than the magnitude or contour concavities as stated in the paper).

QUESTIONS:
- do skewed distributions need transformations?
- how does the fact that we have mean, worst, and stdev as features come into play here? as the distributions depend on the group basically...

##### pca:
- rotates the dataset so that the rotated features are statistically uncorrelated, then we can choose a subset of the new features according to how important they are for explaining the data (ML notes)
- finds the direction of maximum variance and the direction that contains the most information (the principal components)
- mean centering as a preprocessing technique (also in ML notes!) to ensure the first principal component is in direction of maximum variance and not affected by the scale of data
- normalizing before might be important because of the differences between the scales of the three groups (especially std) as well as individual measures (e.g. scale of area being greater than that of the smoothness...)
QUESTION: is this normalization included in the PCA procedure or do we have to normalize before? do we perform a kind of normalization / standardization technique or mean centering (simply subtracting the mean)? do we want to keep the relationships between feature values and their distribution as they are though or is this not important when performing PCA?

### UNCERTAINTY MEASURES

It is especially important to use model predictions carefully in healthcare, as both false negatives (fail to recognize cancer) and false positives (diagnosing a patient with cancer when there is none) can be detrimental. It is therefore very important to consider data and model uncertainty before making further important decisions (e.g. treatment-wise) based on the outputs. Uncertainty measures can be integrated into the model's predictions and serve as a guide for the medical professionals involved in making the final diagnosis.
- bias introduced when medical personnel decide what part of the sample is digitized --> select a number of diff areas for digitization

ALEATORIC UNCERTAINTY: There is uncertainty inherent to the data which also stems from the procedure through which the data is acquired. Generating a digital image of the sample introduces some initial noise, which is then amplified by cell boundary approximation. This is done using an active contour model known as a "snake". Another point to consider, as stated before, is that the dataset is built upon samples that are malignant and labeled as such but might include benign cells as well. Therefore aleatoric uncertainty is indeed quite relevant within the various measurements obtained.

EPISTEMIC UNCERTAINTY: The uncertainty associated with the model itself might be increased by factors relevant in the training process, such as class imbalance, lack of training data, missing data points, model misspecification. It can be reduced by adding information to the training process, but still needs to be taken into account when we are interested in predictive uncertainty as a whole (aleatoric + epistemic uncertainty). In the present task, certain bias is introduced in the dataset when medical personnel decide what part of the sample is digitized (based on suspicions about malignancy for example). 

Example techniques:
- Gaussian NLL (learns aleatoric uncertainty), DUQ, gradient uncertainty, etc.
- sampling-based (expensive but good estimates) such as MC Dropout, MC DropConnect
- so for epistemic: MC Droupout / DropConnect, ensembles, bayes by backprop, flipout

