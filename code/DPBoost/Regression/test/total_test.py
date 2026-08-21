from self_tool import metrics_science

# metrics_science.run_experiment(core_str='AI4I', Base=True, kFold=True)
# metrics_science.run_experiment(core_str='abalone', Base=True, kFold=True)
# metrics_science.run_experiment(core_str='Bias', Base=True, kFold=True)
# metrics_science.run_experiment(core_str='superconduct', Base=True, kFold=True)
#
# metrics_science.run_experiment(core_str='AI4I', xgb=True, AAAI=True, Gbdt=True, pri_list=[1], kFold=True, k=36.4, b=40.2)
# metrics_science.run_experiment(core_str='abalone', xgb=True, AAAI=True, Gbdt=True, pri_list=[1], kFold=True, k=14, b=15)
# metrics_science.run_experiment(core_str='superconduct', xgb=True, AAAI=True, Gbdt=True, pri_list=[1], kFold=True, k=92.499895, b=92.500105)

metrics_science.run_experiment(core_str='abalone', Gbdt=True, kFold=True, k=14, b=15)
metrics_science.run_experiment(core_str='Bias', Gbdt=True, kFold=True, k=2.53985, b=2.63835)
metrics_science.run_experiment(core_str='AI4I', Gbdt=True, kFold=True, k=36.4, b=40.2)
metrics_science.run_experiment(core_str='superconduct', Gbdt=True, kFold=True, k=92.499895, b=92.500105)
