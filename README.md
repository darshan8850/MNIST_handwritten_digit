## MNIST Handwritten Digit Classification

This repository contains scripts for training a convolutional neural network (CNN) model on the MNIST dataset, serving the trained model using Flask, and deploying the inference service on Kubernetes. MLflow is used for experiment tracking during model training.

### Project Structure

mnist_classification/
│
├── train.py # Script for training the CNN model and logging with MLflow
├── inference.py # Script for serving the trained model using Flask
├── Dockerfile # Dockerfile for building the inference image
├── deployment.yaml # Kubernetes deployment configuration
├── service.yaml # Kubernetes service configuration
├── requirements.txt # Python dependencies
├── mlruns/ # MLflow tracking directory (contains experiment logs)
├── model.h5 # Trained CNN model saved in HDF5 format
