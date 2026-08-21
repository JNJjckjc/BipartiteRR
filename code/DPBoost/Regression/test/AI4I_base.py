from self_tool import metrics_science

# 测试不加差分隐私的xgboost中每棵树运行时间
metrics_science.run_experiment(core_str='AI4I', kFold=True, Base=True, all_tree=50)















