Crypto Trader Classifier with Logistic Regression
Overview
Welcome to the repository for our project on classifying cryptocurrency traders using logistic regression. This tool aims to categorize traders into three groups: Good Trader, Average Trader, and Bad Trader, based on their trading performance metrics.

Features Used
Base Cumulative Return: Total return over time.
Portfolio Return: Percentage return on the entire portfolio.
Daily Sharpe Ratio: Risk-adjusted performance measure.
Number of Trades: Total trades executed.
Unique Tokens Traded: Diversity of assets traded.

Methodology
Data Collection: Gathered using Flipside Crypto's SQL terminal, focusing on traders with volumes over $10 million.
Analysis and Modeling: Utilized Python for:
Statistical analysis
Feature engineering
Model training with logistic regression

Key Findings
Correlation Analysis: Identified Portfolio Return (0.2718) and Base Cumulative Return (0.2387) as the most influential features.
Model Performance: Achieved an accuracy of 97.87% in trader classification.

Project Structure
/data: Contains CSV files with raw and processed trading data.
/src: Source code for data analysis, model training, and testing.
/api: Flask application for model predictions.
Dockerfile: For containerizing the application.
requirements.txt: Python dependencies.

How to Use
Clone the Repository:
bash
git clone https://github.com/apostleoffinance/web3_ml.git
cd web3_ml
Setup Environment:
bash
pip install -r requirements.txt
Run the Model:
Locally:
bash
python src/train_model.py
With Docker:
bash
docker build -t crypto-trader-classifier .
docker run -p 5000:5000 crypto-trader-classifier
API Usage: After running the Flask server, you can make POST requests to /predict with trader data in JSON format to get classifications.

Applications
Copy-Trading: Mimic strategies of high-performing traders.
Risk Management: Avoid strategies of underperforming traders.
Portfolio Optimization: Use data-driven insights for better asset allocation.

Technologies
Machine Learning: Python, scikit-learn
API Development: Flask
Containerization: Docker
Version Control: GitHub

Further Reading
Check out the Flipside Crypto Dashboard for the SQL queries used in this project: Flipside Dashboard

Contribution
Feel free to fork this project, make your contributions, and submit pull requests. Any improvements and ideas are welcome!

License
This project is open-sourced under the MIT License (LICENSE).

Thank you for visiting this repo, and happy trading analysis!
