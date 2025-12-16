
# MLOps Assignments



## Tech Stack, Cloud, Concepts

```Python```
```GCP```
```DVC```
```Feast```
```MlFlow```
```Kubernetes```
```Docker```
```Github```
```CI/CD using Github Actions```
```Logging```
```Tracing```
```Monitoring```
```Governance```
```Poisoning```
```Explainability using SHAP```
```Drift Detection using Evidently```
```LLMOPs```




## GCP Project

[Project URL](https://console.cloud.google.com/welcome?project=sheetgpt-385916)

## Week 1

Setting up the ML pipeline for IRIS Classifier in Vertex AI platform using GCS as demonstrated in the lecture (**Hands-on: Introduction to Google Cloud, Vertex AI**) in your GCP account.

### Tasks

- Activate your GCP Trial
- Setup Vertex AI Workbench (Enable appropriate services / APIs as required)
- Store Training Data in Google Storage Bucket
- Fetch the data from Google Storage Bucket and successfully execute the IRIS Machine Learning training pipeline
- Store the output artifacts (models, logs, etc.) in Google Cloud Storage bucket with folders organized by their training execution timestamp
- Create a new script for inference and run the inference on eval set after fetching the models from GCS output artifacts bucket
- Run this training and inference for 2 times resulting in two output artifact folders in Google Cloud Storage bucket
- (Optional) Run this pipeline for two versions of data provided in GitHub data folder

This assignment is critical for your MLOps learning as this pipeline will be the basis for incorporating more features and tools as the course progresses.

### References

Find required reference IPython Notebook and IRIS dataset at GitHub:

- **GitHub Repository:** https://github.com/IITMBSMLOps/ga_resources
- **Branch:** week_1


### Solution video
[Week1 Video](https://drive.google.com/file/d/1lE-1XJx2WZhbDEXIdzMOmJO3JZroum8S/view?usp=drive_link)
## Week2

Incorporate DVC for the local data into the homework pipeline.  
Setup DVC in the IRIS pipeline set up as part of the Week-2 assignment.

### Tasks

- Setup the Git repository
- Configure DVC to use Google Cloud Storage bucket as remote storage
- Augment the IRIS data to simulate data additions and start training
- Demonstrate storing data and model files as part of DVC
- Demonstrate the ability to traverse through data versions effortlessly using `dvc checkout`
- DVC command sheet – [here](https://drive.google.com/file/d/1G-9IaZiE4rP-gRaQ51wGNjJ_gCYWVdJS/view)

### Solution Video

[Week2 Video](https://drive.google.com/file/d/10ePVEVVdKL_Y1ZMPVYECz96_JeqUUmsE/view?usp=sharing)
## Week3

Incorporate Feast into the homework pipeline.

## Tasks

- Setup Feast – Feature Store in existing IRIS pipeline
- Fetch features from Feature Store at the point of training and inference
- Optional: Use BigQuery for offline/online Feature Store backend (SQLite is also acceptable)

## References

Find required reference IPython Notebook and IRIS dataset at GitHub:

- **GitHub Repository:** https://github.com/IITMBSMLOps/ga_resources
- **Branch:** week_3

## Solution Video

[Week3 Video](https://drive.google.com/file/d/16Vp0deMgLLgS_dmFTf9ivl9EOZ9ZKW7_/view?usp=sharing)
## Week4

- Setup IRIS homework pipeline into a GitHub repository with two branches: `dev` and `main`
- Create evaluation and data validation unit tests using `pytest` or `unittest`
- For evaluation and testing, configure Continuous Integration (CI) with GitHub Actions to fetch the model and data needed for evaluation from DVC configured in Week-3
- Push inclusion of `pytest` code changes to `dev` branch and raise a Pull Request to `main` branch
- Every branch should have its own CI on push or PR merge
- Run a sanity test using GitHub Actions, printing a report as a comment using `cml`
- Lecture reference – GitHub commands – [here](https://github.com/IITMBSMLOps/ga_resources/tree/week_4)

## Solution Video
Did not attempt
## Week5

Integrate MLFlow into the homework pipeline by introducing hyperparameter tuning as part of the training loop.

## Tasks

- Log experiment parameters, evaluation metrics, and models using MLFlow
- Demonstrate comparing two experiments using metric visualization in the MLFlow portal
- Remove existing model logging dependency from DVC
- Modify the evaluation pipeline to fetch and utilize the latest/best model from the MLFlow model registry
- (Optional) Modify CI to fetch and utilize the latest/best model from the MLFlow model registry to run sanity checks

## Solution Video

[Week5 Video](https://drive.google.com/file/d/1dMVrHZ162oc8ytrmI3wYW7FOaPSelz-u/view?usp=sharing)

## Week6

Building on top of last week’s Continuous Integration with GitHub Actions.

Develop and integrate Continuous Deployment using GitHub Actions for building the IRIS API using Docker and deploying onto Kubernetes (K8s).

## Tasks

- Explain the difference between a Kubernetes Pod and a Docker container as part of the screencast
- Use GitHub workflows/actions to build the Docker image using a `Dockerfile`
- Push the image to Google Artifact Registry
- Setup GCP service account as needed
- Deploy the application using Google Kubernetes Engine from GitHub Actions

## Solution Video
[Week6 Video Part1](https://drive.google.com/file/d/1oOVHquI8FubgRRZowtf1zHtYTZ6iDYlb/view?usp=drive_link)

[Week6 Video Part2](https://drive.google.com/file/d/14R3XV3Nnu0BivuoMmQmddh1JpnsMwiaV/view?usp=drive_link)
## Week7

Building on last week’s CI/CD pipeline.

This week, we will be scaling the homework IRIS classification pipeline to handle multiple concurrent inferences and observe bottlenecks.

## Tasks

- Extend your existing GitHub CI/CD workflow to stress test the deployment
- Use `wrk` to simulate a high number (>1000) of requests after successful deployment
- Demonstrate Kubernetes auto scaling with `max_pods: 3` and default pod availability of `1`
- Observe bottlenecks when auto scaling is restricted to `1` pod and request concurrency is increased from `1000` to `2000`

## Solution Video

[Week7 Video](https://drive.google.com/file/d/18JDS-XDFm2hi7Sc0mgT-91da0yBO9e8H/view?usp=drive_link)
## Week8

Integrate data poisoning for IRIS using randomly generated numbers at various levels(5%,10%,50%) and explain the validation outcomes when trained on such data using MLFlow

Give your thoughts on how to mitigate such a poisoning attacks and how data quantity requirements evolve when data quality is affected

## Solution Video

[Week8 Video](https://drive.google.com/file/d/1XvmdYBc2FIj95EGNxYRRjh2SZ725YMvv/view?usp=drive_link)
## Week9

Introduce a “location” attribute in IRIS dataset with values 0 and 1 assigned randomly. 

Incorporate fairlearn explainer with location as sensitive attribute.

Explain in simple words what do the SHAP full dataset explainer plots (similar to what was shown in the demo) for class virginica mean.

## Solution Video

[Week9 Video](https://drive.google.com/file/d/1qChCta4bkomSr9waqib_oYMURPAS1oRQ/view?usp=drive_link)
## Week10

Use Vertex AI to train an IRIS classifier using a cost-efficient model from Gemini.

## Tasks

- Convert the data into a compatible and optimal format for LLM fine-tuning, incorporating appropriate evaluation metrics
- Validate improvements using evaluation metrics
- Train two versions of the model and compare performance:
  - **v1:** Raw IRIS data without any pre-processing
  - **v2:** IRIS data converted into descriptive form to exploit LLM capabilities

## Solution Video

[Week10 Video](https://drive.google.com/file/d/19fKcVdlf1SroVx-yQx2NZ4IR3bH4S0CS/view?usp=drive_link)

[Week10 Colab](https://colab.research.google.com/drive/1p-HMbeNOBm4JpdE0wkYZE_3WJnvGZuDq)