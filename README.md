### Important Notice
This is a project for the Lucerne University of Applied Science and Arts (HSLU) in the module Data Science Project 2 (DSPRO2). This project was developed by Alvin and Jakob.


## How to use the project
### Prerequisites
Installed Python 3.8 or higher.

### Installation
1. Clone the repository
2. Create a venv with `python -m venv venv`
3. Activate the venv with `source venv/bin/activate`
4. Install the requirements with `pip install -r requirements.txt`

### 1. Scrape the newest data from Xeno Canto
Run the following script to scrape data from xeno-canto.org: `python scrape.py`

### 2. Create a data split
Run the following script to create a data split: `python split.py`

For further information run: `python split.py --help`

### 3. Train the model
Run the following script to train the model: `python train.py`

The script validates all entries before training the model.   

For further information run: `python train.py --help`

## Logs
The logs are stored in the `logs` directory. To view the logs in real-time, run the following commands:
- Linux/Mac: `tail -f logs/app.log`
- Windows: `Get-Content logs/app.log -Wait`
- Integrated in PyCharm [Guide](https://www.jetbrains.com/help/pycharm/setting-log-options.html)