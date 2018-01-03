# Article Categorization
Software that categorize articles (among themes like politics, science, etc.) based on a training on wikipedia articles

![Screenshot of the interface](/screenshots/interface.png)

## General description

Machine learning software I implemented for my second-year bachelor project (TIPE, for french readers), when I had just started machine learning. The goal of this software is to predict the category of a given article, among "science", "sport", "politcs" and "art". For that, I created a GUI where you can choose your training set (a set of articles in text format) and evaluate the algorithm on either a selected article from the web or on a test set. The starting example of training set is a dataset of wikipedia articles.

## Technical description

There was basically four parts in the production of this project:
* Getting and parsing the wikipedia articles: python scripts
* Creating a feature-extractor to turn each article into a vector of numbers (NLP part): I used pystemmer to extract the tf-idf of each article
* Finding a machine learning algorithm to predict the categories (ML part): I used SVM since the subject of the bachelor project was SVM
* Creating the interface: tkinter

## Run the code

First install all the libraries in the file *requirements.txt*, then type the command:
```bash
python3 run.py
```
