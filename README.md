# Week 8 - Poisoning

### Files

##### Outputs:
* clean - contains metrics and model for run without poisoning
* poison_5 - contains metrics and model for run with 5% poisoning
* poison_10 - contains metrics and model for run with 10% poisoning
* poison_15 - contains metrics and model for run with 15% poisoning

##### Scripts:
* poisoning.py - for poisoning by interchanging labels
* train.py - for running knn model with mlflow

##### Other files:
* data.dvc - for tracking clean source data
* poisoned_data.dvc - for tracking poisoned data

### Mitigating Data Poisoning Attacks

Data poisoning attacks happen when incorrect or intentionally manipulated data is added to the training set, causing the model to learn the wrong patterns. To reduce the impact of such attacks, we can use several practical strategies:

##### 1. Data Validation & Quality Checks

Before training, we should validate incoming data by checking:

* feature ranges
* outliers
* unexpected label patterns
* sudden distribution shifts

If the poisoned rows look very different from normal data, these checks can catch them early.

##### 2. Track Data Lineage

Tools like DVC help track where data came from and what changed.
If we detect a problem later, we can easily roll back to a clean version of the dataset.

##### 3. Robust Models or Training Methods

Some models (like KNN or Decision Trees) are very sensitive to label noise.
Using:
* robust loss functions
* regularization
* noise-tolerant models
can make training less affected by poisoned samples.

##### 4. Monitoring Model Behavior

Monitoring prediction distributions in production can detect when the model suddenly becomes worse — which is often an early signal of poisoning.

##### 5. Human Oversight for Labels

If labels might be attacked, having periodic manual review or cross-checking labels can prevent corrupted labels from entering the training set.


### How Data Quantity Requirements Change When Quality Drops

When data quality becomes worse, we need more data to learn the same patterns.

**Why?**

If a portion of the data is poisoned or noisy:

- the model receives fewer clean examples  
- the decision boundary becomes harder to learn  
- the model’s confidence and accuracy drop  

To compensate, we must increase the total amount of data so that the model still has enough **clean** samples to learn correctly.

---

### **Example**

If **10%** of the dataset is poisoned and we want **1,000 clean examples**:

`clean_needed = 1000`
`noise = 10%`
`total_required = clean_needed / (1 - noise)`
`= 1000 / 0.9`
`= 1111 examples`

If **50%** is poisoned:

`1000 / 0.5 = 2000 examples`

---

### **Conclusion**

The more the data is corrupted, the more total data we need to keep model performance stable.


