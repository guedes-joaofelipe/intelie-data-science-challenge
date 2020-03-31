# Intelie Data Science Challenge

This repository contains my personal solution to the [Intelie](http://www.intelie.com.br/) data science challenge. The challenge is composed of 2 parts (both of them to be solved in *python*):

- A data structure-related challenge
- A neural network-based recommender system

## Data Structure Challenge

Consider an information model, where a register is represented by a tuple. A tuple (or list) in this context is called a *fact*. 

Examples of a fact: `('joão', 'idade', 18, True)`

In this representation, the **entity (E)** `joão` has an **attribute (A)** `idade` with **value (V)** `18`. 

In order to indicate the removal (or retraction) of an information, the 4th element associated to the tuple can be `False` to represent that such entity no longer has the associated attribute value. 

As commonly seen in entity modeling, the attributes of an entity can have cardinality *1* or *N* (many). 

The goal of this challenge is to write a function that returns which are the current facts about those entities. In other words, what are the information that are valid in the current moment. The function has to receive `facts` (all known facts) and `schema` as arguments. 

The proposed solution and its unitary tests can be found in the `/Python/desafio.py` file.

## Neural Network-based Recommender System

The second part of the challenge is to build a recommender system for the *MovieLens 1M* dataset using a neural network. All explanations about hyperparameter choosing and model tuning need to be addressed. 

The solution was developed using an Autoencoder-based neural network and both theoretical and experimental explanations can be found in the `RecommenderSystem.ipynb` notebook. The original dataset can be found in the `Dataset`folder and the Autoencoder class is written in the `Sources/recommender` module. 