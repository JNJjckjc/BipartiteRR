from self_tool import metrics_science

metrics_science.run_experiment(core_str='superconduct',
                               xgb_g_brr=True,
                               save_test_result=True,
                               kFold=True,
                               k=92.499895, b=92.500105,
                              internal_rate=0.2,
                               pri_list=[5])

# internal_rate=[0.01, 0.05, 0.1, 0.3, 0.5]
# for i in range(len(internal_rate)):
#     metrics_science.run_experiment(core_str='superconduct',
#                                    xgb=True,
#                                    save_test_result=True,
#                                    kFold=True,
#                                    k=92.499895, b=92.500105,
#                                    internal_rate=internal_rate[i])
"""
metrics_science.run_experiment(core_str='superconduct',
                               Gbdt=True,
                               kFold=True,
                               pri_list=[4],
                               k=92.5, b=92.5)
"""