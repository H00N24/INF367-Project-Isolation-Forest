## INF367 Project 1

Implementation and testing of Isolation Forest anomaly detection method

### Solution
Project assigment and are in `Project 1.ipynb` jupyter notebook.

### Prerequisities
* python >=3.7
* python venv/virtualenv library
* python libraries specified in `requiriments.txt`

### Commands for installing & running the notebook
* Creating and activating python virtual enviroment
```sh
$ python -m virtualenv venv # python -m venv venv
$ source venv/bin/activate
```
* Installing python libraries and ipykernel for jupyter notebook
```sh
[venv]$ pip install -r requiriments.txt
[venv]$ python -m ipykernel install --user --name inf367-project-1 --display-name "INF367 Project 1"
```
* Unziping data and running the notebook
```sh
[venv]$ unzip credit-card-fraud.zip
[venv]$ jupyter notebook
```
