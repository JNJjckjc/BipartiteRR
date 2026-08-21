from self_tool import metrics_science

metrics_science.run_experiment(core_str='Bias',
                               xgb_g_lap=True,
                               kFold=True,
                               # save_test_result=True,
                               k=2.53985, b=2.63835,
                               pri_list=[1],#隐私预算表
                               internal_rate=0.1)


