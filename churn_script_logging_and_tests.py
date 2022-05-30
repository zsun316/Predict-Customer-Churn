import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import 
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err



def test_eda(import_data, perform_eda):
	'''
	test perform eda function
	'''

	df = import_data("./data/bank_data.csv")
	perform_eda(df)
	pth_churn = './images/eda/churn_distribution.png'
	pth_age = './images/eda/customer_age_distribution.png'
	pth_heatmap = './images/eda/heatmap.png'
	pth_marital = './images/eda/marital_status_distribution.png'
	pth_transaction = './images/eda/total_transaction_distribution.png'

	try:
		assert os.path.exists(pth_churn) == True and os.path.getsize(pth_churn) > 0
		logging.info("Check churn_distribution.png: SUCCESS")
	except AssertionError as err:
		logging.error("Check churn_distribution.png: The file wasn't found")
		raise err

	try:
		assert os.path.exists(pth_age) == True and os.path.getsize(pth_age) > 0
		logging.info("Check customer_age_distribution.png: SUCCESS")
	except AssertionError as err:
		logging.error("Check customer_age_distribution.png: The file wasn't found")
		raise err

	try:
		assert os.path.exists(pth_heatmap) == True and os.path.getsize(pth_heatmap) > 0
		logging.info("Check heatmap.png: SUCCESS")
	except AssertionError as err:
		logging.error("Check heatmap.png: The file wasn't found")
		raise err

	try:
		assert os.path.exists(pth_marital) == True and os.path.getsize(pth_marital) > 0
		logging.info("Check marital_status_distribution.png: SUCCESS")
	except AssertionError as err:
		logging.error("Check marital_status_distribution.png: The file wasn't found")
		raise err

	try:
		assert os.path.exists(pth_transaction) == True and os.path.getsize(pth_transaction) > 0
		logging.info("Check total_transaction_distribution.png: SUCCESS")
	except AssertionError as err:
		logging.error("Check total_transaction_distribution.png: The file wasn't found")
		raise err


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	test_import(cl.import_data)
	test_eda(cl.import_data, cl.perform_eda)









