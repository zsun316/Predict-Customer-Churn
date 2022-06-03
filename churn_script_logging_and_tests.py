"""
Script Description: Test and monitor churn.library.py
Name: Zhahan Sun
Date: Jun 3
"""
import os
import logging
import churn_library as cl
import joblib

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


def test_encoder_helper(import_data, encoder_helper):
	'''
	test encoder helper
	'''
	df = import_data("./data/bank_data.csv")
	categories = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
	responseName = 'Churn'
	encoded_df = encoder_helper(df, categories, responseName)

	try:
		assert encoded_df.shape[0] > 0
		assert encoded_df.shape[1] > 0
		logging.info('encoded df dimension correct.')
	except AssertionError as err:
		logging.error("Testing test_encoder_helper: The file doesn't appear to have rows and columns")
		raise err

def test_perform_feature_engineering(import_data, encoder_helper, 
	                                 perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	df = import_data("./data/bank_data.csv")
	categories = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
	responseName = 'Churn'

	df = encoder_helper(df, categories, responseName)
	mp = perform_feature_engineering(df, responseName)
	try:
		assert mp['X_train'].shape[0] > 0
		assert mp['X_train'].shape[1] > 0
		assert mp['X_test'].shape[0] > 0
		assert mp['X_test'].shape[1] > 0
		assert mp['y_train'].size > 0
		assert mp['y_test'].size > 0
		logging.info("Testing perform_feature_engineering: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_feature_engineering: failed")
		raise err


def test_train_models(import_data, encoder_helper,
                      perform_feature_engineering,
                      classification_report_image,
                      feature_importance_plot,
	                  train_models):
	'''
	test train_models
	'''
	df = import_data("./data/bank_data.csv")
	categories = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
	responseName = 'Churn'

	df = encoder_helper(df, categories, responseName)
	mp = perform_feature_engineering(df, responseName)
    
	X_train, X_test, y_train, y_test = mp['X_train'], mp['X_test'], mp['y_train'], mp['y_test']
	train_models(X_train, X_test, y_train, y_test)
	rfc_model = joblib.load('./models/rfc_model.pkl')
	lr_model = joblib.load('./models/logistic_model.pkl')
	y_train_preds_rf = rfc_model.predict(X_train)
	y_test_preds_rf = rfc_model.predict(X_test)
	y_train_preds_lr = lr_model.predict(X_train)
	y_test_preds_lr = lr_model.predict(X_test)

	classification_report_image(y_train,y_test,y_train_preds_lr,y_train_preds_rf,y_test_preds_lr,y_test_preds_rf)
	pth_feat = './images/results/feature_importances.png'
	feature_importance_plot(rfc_model, mp['X'], pth_feat)

	pth_rf_res = './images/results/rf_results.png'
	pth_logistic_res = './images/results/logistic_results.png'
	pth_roc_auc = './images/results/roc_curve_result.png'
	try:
		assert os.path.exists(pth_feat) == True and os.path.getsize(pth_feat) > 0
		logging.info("Check feature_importance.png: SUCCESS")
	except AssertionError as err:
		logging.error("Check feature_importance.png: The file wasn't found")
		raise err

	try:
		assert os.path.exists(pth_rf_res) == True and os.path.getsize(pth_rf_res) > 0
		logging.info("Check rf_results.png: SUCCESS")
	except AssertionError as err:
		logging.error("Check rf_results.png: The file wasn't found")
		raise err

	try:
		assert os.path.exists(pth_logistic_res) == True and os.path.getsize(pth_logistic_res) > 0
		logging.info("Check logistic_results.png: SUCCESS")
	except AssertionError as err:
		logging.error("Check logistic_results.png: The file wasn't found")
		raise err

	try:
		assert os.path.exists(pth_roc_auc) == True and os.path.getsize(pth_roc_auc) > 0
		logging.info("Check roc_auc_result.png: SUCCESS")
	except AssertionError as err:
		logging.error("Check roc_auc_result.png: The file wasn't found")
		raise err



if __name__ == "__main__":
	test_import(cl.import_data)
	test_eda(cl.import_data, cl.perform_eda)
	test_encoder_helper(cl.import_data, cl.encoder_helper)
	test_perform_feature_engineering(cl.import_data, cl.encoder_helper, cl.perform_feature_engineering)
	test_train_models(cl.import_data, cl.encoder_helper,
                      cl.perform_feature_engineering,
                      cl.classification_report_image,
                      cl.feature_importance_plot,
	                  cl.train_models)









