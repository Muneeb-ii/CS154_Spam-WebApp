[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/t-XuvzOz)
# Project 7 WebApp

### Name

Muneeb Azfar Nafees

### Introspection

_Describe the challenges you faced and what you learned_

### Dataset

_Which dataset did you choose for the classification task?_

### Resources

_List the people and resources you used to complete the project_


### *DO NOT EDIT BELOW THIS LINE*
---

## Goal

The goals of this project are:

* Host your ML model as a web app for others to try it out. 


## Description

In this project, you will host your ML model as a web app. This project contains all the files you need to create a web app on your own.

In this project, the web app will be local. However, if you are interested in hosting it online so that others can see the app, you can either use GCP or AWS which provide some free tiers. But, a simple option will be Heroku. [This](https://blog.bolajiayodeji.com/how-to-deploy-a-machine-learning-model-to-the-web) article provides some guidance. Although, you can search the web for some alternatives. 

The main tasks are as follows:

1. Convert the Jupyter notebook into a Python script, similar to the `model.py`. 
2. Run the `model.py` to create `.pkl` objects. These objects are saving the model and vectorizer in binary code.
3. Run the web app module `main.py` by running `python app/main.py`
4. Open your web browser and go to http://0.0.0.0:8000


## Pre-Requisites

A couple of things to note before you start

1. Install the following dependencies - `fastapi, uvicorn, python-multipart` using pip.
2. Ensure your model is able to predict with decent accuracy.
3. Before running the `main.py` file. Ensure you have new `model` and `vectorizer` objects `.pkl` saved.
4. Read the `model.py` and compare it with last week's Jupyter notebook to understand the differences. 