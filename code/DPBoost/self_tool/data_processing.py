
import pandas as pd
import numpy as np

from self_tool import metrics_science
from sklearn import datasets

"""
test:
    1. e-shop clothing 2008                    (165474, 12)   max: 82,     min: 18,      k: 32.0,      b: 50.0
    2. Seoul Bike Sharing Demand               (8760, 13)     max: 23,     min: 0,       k: 11.5,      b: 11.5
    3. syc_r                                   (100000, 100)  max: 811.9818875499986 min: -881.3931153651708 k: 846.6875014575847 b: -34.7056139075861
    
    1. credit card clients                     (30000, 25)
    2. Internet Firewall                       (65532, 12)  
    3. Shill Bidding                           (6321, 13)
    4. Winnipeg                                (325834, 175)
    5. covtype(aaai)                           (581012, 54) 
    6. cod(aaai)                               (59535, 8)
    7. syc_c                                   (100000, 100)
    
experience:
    1. AI4I 2020 Predictive Maintenance       (10000, 12)    max: 76.6,   min: 3.8,     k: 36.4,      b: 40.2
    2. Bias correction                        (7588, 23)     max: 5.1782, min: 0.0985,  k: 2.53985,   b: 2.63835
    3. superconduct                           (21263, 82)    max: 185.0,  min: 0.00021, k: 92.499895, b: 92.500105
    4. abalone(aaai)                          (4177, 8)      max: 29      min：1        K: 14.0       b: 15.0
    5. YearPredictionMSD(aaai)                (463715, 90)   max: 2011    min: 1922     K: 44.5       b: 1966.5
        Yearc                                 (100000, 90)   max: 2010    min:1922      K:44          b:1966
"""
pass # aaai_1  cod 数据集
# cod
# total_data = pd.read_csv('F:/XGBoost_science/data/classification/cod/total_tem', sep=' ', header=None)
# for row in range(total_data.shape[0]):
#     for column in range(1, total_data.shape[1]):
#         value_list = str(total_data.iloc[row, column]).split(sep=':')
#         total_data.iloc[row, column] = float(value_list[1])
# metrics_science.draw_bar(total_data)
# total_data.dropna(inplace=True)
# total_data = total_data.reset_index(drop=True)
# metrics_science.draw_bar(total_data)
# total_data = metrics_science.upset_dataset(total_data)
# total_data.to_csv('F:/XGBoost_science/data/classification/cod/cod', index=False, sep=',')

# X_total = pd.read_csv('F:/XGBoost_science/data/classification/cod/cod')
# y = pd.DataFrame(X_total.loc[:, '0'])
# y.columns = ['class']
#
# y.to_csv('F:/XGBoost_science/data/classification/cod/cod_y', index=False)
# y.to_csv('F:/XGBoost_science/data/classification/cod/map_cod_y', index=False)
#
# X_total.drop(['0'], axis=1, inplace=True)
# X_total.to_csv('F:/XGBoost_science/data/classification/cod/cod_X', index=False)
# X = metrics_science.continue_to_dispersed(X_total)
# X.to_csv('F:/XGBoost_science/data/classification/cod/map_cod_X', index=False)

pass # aaai_2 covtype 从sklearn 读取数据集
# datax = datasets.fetch_covtype()['data']
# datay = datasets.fetch_covtype()['target']
# X = pd.DataFrame(datax)
# y = pd.DataFrame(datay)
# X.to_csv('F:/XGBoost_science/data/classification/covtype/covtype_x', index=False, sep=',')
# y.to_csv('F:/XGBoost_science/data/classification/covtype/covtype_y', index=False)

# y_temp = pd.read_csv('F:/XGBoost_science/data/classification/covtype/covtype_y_t')
# label_list = np.unique(y_temp)
# unique_num = len(label_list)
# every_label_num = [0] * unique_num
# for i in range(y_temp.shape[0]):
#     for label_index in range(unique_num):
#         if y_temp.iloc[i, 0] == label_list[label_index]:
#             every_label_num[label_index] += 1
#             break
# print(label_list)
# print(every_label_num)

# X_temp = pd.read_csv('F:/XGBoost_science/data/classification/covtype/covtype_x_t')
# y_temp = pd.read_csv('F:/XGBoost_science/data/classification/covtype/covtype_y_t')
# y_temp.columns = ['class']
# X_total = pd.concat([X_temp, y_temp], axis=1)
#
# metrics_science.draw_bar(X_total)
# X_total.dropna(inplace=True)
# X_total = X_total.reset_index(drop=True)
# metrics_science.draw_bar(X_total)
#
# X_upset = metrics_science.upset_dataset(X_total)
# y = pd.DataFrame(X_total.loc[:, 'class'])
# y.columns = ['class']
# for index in range(y.shape[0]):
#     if y.iloc[index, 0] != 2:
#         y.iloc[index, 0] = 1              no no no no no no

# y.to_csv('F:/XGBoost_science/data/classification/covtype/covtype_y', index=False)
# y = metrics_science.y_label_mapping(y, binary_classification=True)
# y.to_csv('F:/XGBoost_science/data/classification/covtype/map_covtype_y', index=False)
#
# X_total.drop(['class'], axis=1, inplace=True)
# X_total.to_csv('F:/XGBoost_science/data/classification/covtype/covtype_X', index=False)
# X = metrics_science.continue_to_dispersed(X_total)
# metrics_science.draw_bar(X)
# X.to_csv('F:/XGBoost_science/data/classification/covtype/map_covtype_X', index=False)

pass # test_1 处理 e-shop clothing 2008.csv 数据集
# ['year', 'month', 'day', 'order', 'country', 'session ID', 'page 1 (main category)', 'page 2 (clothing model)',
# 'colour', 'location', 'model photography', 'price', 'price 2', 'page']
# X = pd.read_csv('F:/dataset/regression/e-shop data and description/e-shop clothing 2008.csv', sep=';')
# X.dropna(inplace=True)
# X = X.reset_index(drop=True)
# X = metrics_science.mapping_feature_value(X, ['page 2 (clothing model)'])
# X = metrics_science.upset_dataset(X)
# y = pd.DataFrame(X.loc[:, 'price'])
# y.columns = ['price']
# y.to_csv('F:/XGBoost/data/e-shop clothing 2008/e-shop clothing_y', index=False)
# y = pd.read_csv('F:/XGBoost/data/science/regression/e-shop clothing 2008/e-shop clothing_y')
# y, max, min, k, b = metrics_science.y_label_mapping(y)
# print('max: {}, min: {}, k: {}, b: {}'.format(max, min, k, b)) # max: 82, min: 18, k: 32.0, b: 50.0
# print(y)
# y.to_csv('F:/XGBoost/data/science/regression/e-shop clothing 2008/map_e-shop clothing_y', index=False)
# X.drop(['year', 'price'], axis=1, inplace=True)
# X.to_csv('F:/XGBoost/data/e-shop clothing 2008/e-shop clothing_X', index=False, sep=',')
# metrics_science.draw_bar(X)
# X = metrics_science.continue_to_dispersed(X)
# metrics_science.draw_bar(X)
# X.to_csv('F:/XGBoost/data/e-shop clothing 2008/map_e-shop clothing_X', index=False)

pass # test_2 处理 abalone 数据集
# X = pd.read_csv('F:/my_science/dataset/not science/abalone/abalone.data', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'class'])
# y = pd.DataFrame(X.iloc[:, 8])
# y.columns = ['class']
# y.to_csv('F:/XGBoost_science/data/regression/abalone/abalone_y', index=False)
# y, max, min, k, b = metrics_science.y_label_mapping(y)
# y.to_csv('F:/XGBoost_science/data/regression/abalone/map_abalone_y', index=False)
# print(y, max, min, k, b)
# X.drop(['class'], axis=1, inplace=True)
# X = metrics_science.mapping_feature_value(X, ['a'])
# X.to_csv('F:/XGBoost_science/data/regression/abalone/abalone_X', index=False)
# X = metrics_science.continue_to_dispersed(X)
# X.to_csv('F:/XGBoost_science/data/regression/abalone/map_abalone_X', index=False)

pass # test_3 处理 YearPredictionMSD 数据集
X = pd.read_csv('F:/my_science/dataset/not science/YearPredictionMSD/YearPredictionMSD.txt', header=None, nrows=50000)
print(X.shape)
y = pd.DataFrame(X.iloc[:, 0])
y.columns = ['Year']
y.to_csv('F:/XGBoost_science/data/regression/Yearc/Yearc_y', index=False)
y, max, min, k, b = metrics_science.y_label_mapping(y)
y.to_csv('F:/XGBoost_science/data/regression/Yearc/map_Yearc_y', index=False)
print(y, max, min, k, b)
X.drop([0], axis=1, inplace=True)
X.to_csv('F:/XGBoost_science/data/regression/Yearc/Yearc_X', index=False)
X = metrics_science.continue_to_dispersed(X)
X.to_csv('F:/XGBoost_science/data/regression/Yearc/map_Yearc_X', index=False)

pass # test_4 处理 AI4I 2020 Predictive Maintenance 数据集
# ['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
#   'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
# X = pd.read_csv('F:/XGBoost/data/AI4I 2020 Predictive Maintenance/AI4I 2020 Predictive Maintenance')
# print(X.columns)
# metrics_science.draw_bar(X)
# X.dropna(inplace=True)
# X = X.reset_index(drop=True)
# X = metrics_science.mapping_feature_value(X, ['Type'])
# print(X)
# X = metrics_science.upset_dataset(X)
# print(X)
# y = pd.DataFrame(X.loc[:, 'Torque [Nm]'])
# y.columns = ['Torque']
# y.to_csv('F:/XGBoost/data/AI4I 2020 Predictive Maintenance/AI4I 2020 Predictive Maintenance_y', index=False)
# y, max, min, k, b = metrics_science.reg_label_mapping(y)
# print('max: {}, min: {}, k: {}, b: {}'.format(max, min, k, b)) # max: 76.6, min: 3.8, k: 36.4, b: 40.199999999999996
# y.to_csv('F:/XGBoost/data/AI4I 2020 Predictive Maintenance/map_AI4I 2020 Predictive Maintenance_y', index=False)
# X.drop(['UDI', 'Product ID'], axis=1, inplace=True)
# X.to_csv('F:/XGBoost/data/AI4I 2020 Predictive Maintenance/AI4I 2020 Predictive Maintenance_X', index=False)
# metrics_science.draw_bar(X)
# X = metrics_science.continue_to_dispersed(X)
# metrics_science.draw_bar(X)
# X.to_csv('F:/XGBoost/data/AI4I 2020 Predictive Maintenance/map_AI4I 2020 Predictive Maintenance_X', index=False)

pass # regression_1 Bias correction of numerical prediction model temperature forecast 数据集
# ['station', 'Date', 'Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin','LDAPS_RHmax', 'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse',
# 'LDAPS_WS','LDAPS_LH', 'LDAPS_CC1', 'LDAPS_CC2', 'LDAPS_CC3', 'LDAPS_CC4','LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3',
# 'LDAPS_PPT4', 'lat', 'lon','DEM', 'Slope', 'Solar radiation', 'Next_Tmax', 'Next_Tmin']
# X = pd.read_csv('F:/dataset/regression/Bias correction of numerical prediction model temperature forecast/Bias_correction_ucl.csv')
# print(X.columns)
# X.dropna(inplace=True)
# X = X.reset_index(drop=True)
# X = metrics_science.mapping_feature_value(X, ['Date'])
# print(X)
# X = metrics_science.upset_dataset(X)
# print(X)
# y = pd.DataFrame(X.loc[:, 'Slope'])
# y.columns = ['Slope']
# y.to_csv('F:/XGBoost/data/Bias correction/Bias correction_y', index=False)
# y, max, min, k, b = metrics_science.reg_label_mapping(y)
# print('max: {}, min: {}, k: {}, b: {}'.format(max, min, k, b)) # max: 5.1782, min: 0.0985, k: 2.5398500000000004, b: 2.63835
# y.to_csv('F:/XGBoost/data/Bias correction/map_Bias correction_y', index=False)
# X.drop(['station', 'Slope'], axis=1, inplace=True)
# X.to_csv('F:/XGBoost/data/Bias correction/Bias correction_X', index=False)
# metrics_science.draw_bar(X)
# X = metrics_science.continue_to_dispersed(X)
# metrics_science.draw_bar(X)
# X.to_csv('F:/XGBoost/data/Bias correction/map_Bias correction_X', index=False)

pass # regression_2 Seoul Bike Sharing Demand 数据集
# ['Date', 'Rented Bike Count', 'Hour', 'Temperature', 'Humidity', 'Wind speed', 'Visibility', 'Dew point temperature',
# 'Solar Radiation',  'Rainfall', 'Snowfall', 'Seasons', 'Holiday', 'Functioning Day']
# X = pd.read_csv('F:/XGBoost/data/science/regression/Seoul Bike Sharing Demand/SeoulBikeData')
# pd.set_option('display.max_columns', None)
# print(X) # 8760
# print(X.columns)
# metrics_science.draw_bar(X)
# X.dropna(inplace=True)
# X = X.reset_index(drop=True)
# X = metrics_science.mapping_feature_value(X, ['Date', 'Seasons', 'Holiday', 'Functioning Day'])
# print(X)
# X = metrics_science.upset_dataset(X)
# print(X)
# y = pd.DataFrame(X.loc[:, 'Hour'])
# y.columns = ['Hour']
# y.to_csv('F:/XGBoost/data/science/regression/Seoul Bike Sharing Demand/SeoulBikeData_y', index=False)
# y, max, min, k, b = metrics_science.reg_label_mapping(y)
# print('max: {}, min: {}, k: {}, b: {}'.format(max, min, k, b)) # max: 23, min: 0, k: 11.5, b: 11.5
# y.to_csv('F:/XGBoost/data/science/regression/Seoul Bike Sharing Demand/map_SeoulBikeData_y', index=False)
# X.drop(['Hour'], axis=1, inplace=True)
# X.to_csv('F:/XGBoost/data/science/regression/Seoul Bike Sharing Demand/SeoulBikeData_X', index=False)
# metrics_science.draw_bar(X)
# X = metrics_science.continue_to_dispersed(X)
# metrics_science.draw_bar(X)
# X.to_csv('F:/XGBoost/data/science/regression/Seoul Bike Sharing Demand/map_SeoulBikeData_X', index=False)

pass # regression_3 superconduct 数据集
# X = pd.read_csv('F:/dataset/regression/Superconductivty/superconduct/train.csv')
# X.dropna(inplace=True)
# X = X.reset_index(drop=True)
# X = metrics_science.upset_dataset(X)
# y = pd.DataFrame(X.loc[:, 'critical_temp'])
# y.columns = ['critical_temp']
# y.to_csv('F:/XGBoost/data/science/regression/Superconductivty/superconduct_y', index=False)
# y, max, min, k, b = metrics_science.y_label_mapping(y)
# print('max: {}, min: {}, k: {}, b: {}'.format(max, min, k, b)) # max: 185.0, min: 0.00021, k: 92.499895, b: 92.500105
# y.to_csv('F:/XGBoost/data/science/regression/Superconductivty/map_superconduct_y', index=False)
# X.drop(['critical_temp'], axis=1, inplace=True)
# X.to_csv('F:/XGBoost/data/science/regression/Superconductivty/superconduct_X', index=False)
# metrics_science.draw_bar(X)
# X = metrics_science.continue_to_dispersed(X)
# metrics_science.draw_bar(X)
# X.to_csv('F:/XGBoost/data/science/regression/Superconductivty/map_superconduct_X', index=False)

pass # regression_4  数据集

pass # classification_1 credit card clients 数据集
# ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
# 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
# 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default payment next month']
# X = pd.read_excel('F:/dataset/classification/default of credit card clients.xls', header=1)
# print(X.shape)
# metrics_science.draw_bar(X)
# X.dropna(inplace=True)
# X = X.reset_index(drop=True)
# X = metrics_science.upset_dataset(X)
# print(X)
# y = pd.DataFrame(X.loc[:, 'default payment next month'])
# y.columns = ['default payment next month']
# y.to_csv('F:/XGBoost/data/science/classification/clients/clients_y', index=False)
# y = metrics_science.y_label_mapping(y, binary_classification=True)
# y.to_csv('F:/XGBoost/data/science/classification/clients/map_clients_y', index=False)
# X.drop(['ID', 'default payment next month'], axis=1, inplace=True)
# X.to_csv('F:/XGBoost/data/science/classification/clients/clients_X', index=False)
# metrics_science.draw_bar(X)
# X = metrics_science.continue_to_dispersed(X)
# metrics_science.draw_bar(X)
# X.to_csv('F:/XGBoost/data/science/classification/clients/map_clients_X', index=False)
# y = pd.read_csv('F:/XGBoost/data/science/classification/clients/clients_y')
# y = metrics_science.y_label_mapping(y, binary_classification=True)
# y.to_csv('F:/XGBoost/data/science/classification/clients/map_clients_y_0.5', index=False)

pass # classification_2 Internet Firewall 数据集
# ['Source Port', 'Destination Port', 'NAT Source Port', 'NAT Destination Port', 'Action', 'Bytes', 'Bytes Sent',
# 'Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received']
# X = pd.read_csv('F:/dataset/classification/Internet Firewall/log2.csv')
# print(X.columns)
# print(np.unique(X['Action'])) # ['allow' 'deny' 'drop' 'reset-both'] [37640, 14987, 12851, 54]
# for i in range(X.shape[0]):
#     if X.loc[i, 'Action'] == 'allow' or X.loc[i, 'Action'] == 'reset-both':
#         X.loc[i, 'Action'] = 1
#     else:
#         X.loc[i, 'Action'] = -1
# metrics_science.draw_bar(X, title='Internet Firewall')
# X.dropna(inplace=True)
# X = X.reset_index(drop=True)
# X = metrics_science.upset_dataset(X)
# print(X)
# y = pd.DataFrame(X.loc[:, 'Action'])
# y.columns = ['Action']
# y.to_csv('F:/XGBoost/data/science/classification/Action/Action_y', index=False)
# y.to_csv('F:/XGBoost/data/science/classification/Action/map_Action_y', index=False)
# X.drop(['Action'], axis=1, inplace=True)
# X.to_csv('F:/XGBoost/data/science/classification/Action/Action_X', index=False)
# X = metrics_science.continue_to_dispersed(X)
# metrics_science.draw_bar(X, title='Action')
# X.to_csv('F:/XGBoost/data/science/classification/Action/map_Action_X', index=False)
# y = pd.read_csv('F:/XGBoost/data/science/classification/Internet/Internet_y')
# y = metrics_science.y_label_mapping(y, binary_classification=True)
# y.to_csv('F:/XGBoost/data/science/classification/Internet/map_Internet_y_0.5', index=False)

pass # classification_3 Shill Bidding 数据集
# ['Record_ID', 'Auction_ID', 'Bidder_ID', 'Bidder_Tendency', 'Bidding_Ratio', 'Successive_Outbidding', 'Last_Bidding',
# 'Auction_Bids', 'Starting_Price_Average', 'Early_Bidding', 'Winning_Ratio', 'Auction_Duration', 'Class']
# X = pd.read_csv('F:/dataset/classification/Shill Bidding/Shill Bidding Dataset.csv')
# print(X)
# print(X.columns)
# metrics_science.draw_bar(X)
# X.dropna(inplace=True)
# X = X.reset_index(drop=True)
# X = metrics_science.mapping_feature_value(X, ['Bidder_ID'])
# X = metrics_science.upset_dataset(X)
# print(X)
# y = pd.DataFrame(X.loc[:, 'Class'])
# y.columns = ['Class']
# y.to_csv('F:/XGBoost/data/science/classification/Shill/Shill_y', index=False)
# y = metrics_science.y_label_mapping(y, binary_classification=True)
# y.to_csv('F:/XGBoost/data/science/classification/Shill/map_Shill_y', index=False)
# X.drop(['Record_ID', 'Class'], axis=1, inplace=True)
# X.to_csv('F:/XGBoost/data/science/classification/Shill/Shill_X', index=False)
# metrics_science.draw_bar(X)
# X = metrics_science.continue_to_dispersed(X)
# metrics_science.draw_bar(X)
# X.to_csv('F:/XGBoost/data/science/classification/Shill/map_Shill_X', index=False)

# y = pd.read_csv('F:/XGBoost/data/science/classification/Shill/Shill_y')
# y = metrics_science.y_label_mapping(y, binary_classification=True)
# y.to_csv('F:/XGBoost/data/science/classification/Shill/map_Shill_y_0.5', index=False)

pass # classification_4 Winnipeg 数据集
# ['label', 'f1', ..., 'f174']
# X = pd.read_csv('F:/dataset/classification/Crop mapping using fused optical-radar/WinnipegDataset.txt')
# print(X)
# metrics_science.draw_bar(X)
# X.dropna(inplace=True)
# X = X.reset_index(drop=True)
# for i in range(X.shape[0]):
#     if X.loc[0, 'label'] < 5:
#         X.loc[0, 'label'] = -1
#     else:
#         X.loc[0, 'label'] = 1
# X = metrics_science.upset_dataset(X)
# print(X)
# y = pd.DataFrame(X.loc[:, 'label'])
# y.columns = ['label']
# y.to_csv('F:/XGBoost/data/science/classification/Winnipeg/Winnipeg_y', index=False)
# y.to_csv('F:/XGBoost/data/science/classification/Winnipeg/map_Winnipeg_y', index=False)
# X.drop(['label'], axis=1, inplace=True)
# X.to_csv('F:/XGBoost/data/science/classification/Winnipeg/Winnipeg_X', index=False)
# X = metrics_science.continue_to_dispersed(X)
# metrics_science.draw_bar(X, title='Winnipeg')
# X.to_csv('F:/XGBoost/data/science/classification/Winnipeg/map_Winnipeg_X', index=False)

# y = pd.read_csv('F:/XGBoost/data/science/classification/Winnipeg/Winnipeg_y')
# for i in range(y.shape[0]):
#     if y.loc[i, 'label'] < 5:
#         y.loc[i, 'label'] = -1
#     else:
#         y.loc[i, 'label'] = 1
# y.to_csv('F:/XGBoost/data/science/classification/Winnipeg/map_Winnipeg_y', index=False)
# y = metrics_science.y_label_mapping(y, binary_classification=True)
# y.to_csv('F:/XGBoost/data/science/classification/Winnipeg/map_Winnipeg_y_0.5', index=False)










