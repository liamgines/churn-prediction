Churn Prediction
========
A notebook for predicting which customers are going to churn (i.e. when customers stop using a product/service). 

![Sample Graph](graph.png)

Dataset
--------
The dataset used is highly imbalanced and contains information about a bank's customers, including whether the customer left the bank by closing their account or stayed with the bank and remained a customer.

Installation
--------
First, ensure you have Python 3 installed.

Then, run the following commands:
```
git clone <repository_url>
cd churn-prediction
py -m pip install --upgrade pip
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Next, find where it says `Or copy and paste one of these URLs:` in Command Prompt and open one of them in a browser.

The URL that you open should look like `http://localhost:{port}/tree?token={token}` or `http://127.0.0.1:{port}/tree?token={token}`.

Once you've opened the URL in a browser, you should be able to click the `.ipynb` file to open the notebook and run the code from there.
