from self_tool import metrics_science

metrics_science.run_experiment(core_str='abalone', xgb_g_lap=True, save_test_result=True, kFold=True, k=14, b=15,
                              internal_rate=0.2, pri_list=[5])











"""
internal_rate=[0.05, 0.1, 0.3, 0.5]
for i in range(len(internal_rate)):
    metrics_science.run_experiment(core_str='abalone',#数据集
                                   xgb=True,
                                   save_test_result=True,
                                   kFold=True,
                                   k=14, b=15,
                                   internal_rate=internal_rate[i])
"""
"""
[5.582620826813635, 4.569230695591291, 3.2150141445315783, 2.6815047335221975, 2.62073269298435]
[6.021316172338419, 4.83825301911101, 3.486599557786044, 3.0323083896037875, 2.633062356363698]
[8.025467990411872, 4.978644516101807, 3.4025327828871488, 3.1348917542329175, 2.571025859303875]
[8.957626509858596, 7.011752473284385, 3.9038410630665465, 3.1021750317095, 2.7940997370234357]
"""