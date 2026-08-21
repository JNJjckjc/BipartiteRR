from self_tool import metrics_science#23w29

# metrics_science.run_experiment(core_str='AI4I',
#                                Gbdt=True,
#                                kFold=True,
#                                pri_list=[4],
#                                k=36.4, b=40.2) # [7.630751504612187]

# internal_rate=[0.01, 0.05, 0.1, 0.3, 0.5]
# for i in range(len(internal_rate)):
#     metrics_science.run_experiment(core_str='AI4I',
#                                    xgb=True,
#                                    save_test_result=True,
#                                    kFold=True,
#                                    k=36.4, b=40.2,
#                                    internal_rate=internal_rate[i])

# metrics_science.run_experiment(core_str='abalone',
#                                Gbdt=True,
#                                kFold=True,
#                                pri_list=[4],
#                                k=14, b=15) # [4.597036427150593]

# metrics_science.run_experiment(core_str='Bias',
#                                Gbdt=True,
#                                kFold=True,
#                                pri_list=[4],
#                                k=2.53985, b=2.63835) # 3.4651344826306874 2.2266670379947664 1.2426640742045882

# metrics_science.run_experiment(core_str='AI4I',
#                                xgb=True,
#                                kFold=True,
#                                internal_rate=0.5,
#                                k=36.4, b=40.2)

"""
平均取样
abalone:
RESULT_xgb: [48.471383578322325, 27.0672591510854, 9.716762791952956, 7.200791969644193, 5.240983726049164, 4.324451812569744]
DEL_samples_rate_xgb: [0.4138119309430321, 0.2348554697624548, 0.030584001500319462, 0.011730040655146665, 0.001974954910707169, 0.0007779772591262717]
AI4I:
RESULT_xgb: [126.81003029303967, 57.95274201201316, 25.249880546544677, 17.31750890831522, 12.603209263517128, 9.783912119900801]
DEL_samples_rate_xgb: [0.425075, 0.196325, 0.0314, 0.00585, 0.00034999999999999994, 0.0]
"""

metrics_science.run_experiment(core_str='AI4I',
                               # xgb=True,
                               AAAI=True,
                               # Base=True,
                               kFold=True,
                               all_tree=20,
                               k=36.4, b=40.2)

metrics_science.run_experiment(core_str='AI4I',
                               # xgb=True,
                               AAAI=True,
                               # Base=True,
                               kFold=True,
                               all_tree=40,
                               k=36.4, b=40.2)

"""
all_tree=20
RESULT_xgb: [16.73541697014717, 9.135786723766007, 6.396615280588675, 5.763800568178593, 4.797120158305796, 4.390313529262997]
DEL_samples_rate_xgb: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
RESULT_AAAI: [30.707833298819935, 19.4253516230121, 9.843454801132042, 8.50476739271953, 6.6933882902688895, 6.359712334438096]
DEL_samples_rate_AAAI: [0.0, 5e-05, 0.0, 0.0, 0.0, 0.0]
2.26816695336361

all_tree=40
RESULT_xgb: [11.907779076223347, 9.429816083091193, 5.982235545882633, 5.209318281668894, 5.090835737769566, 3.7025985252826565]
DEL_samples_rate_xgb: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
RESULT_AAAI: [40.53522868794526, 17.818461008412648, 10.00648920218672, 8.02792338118679, 7.165691275163255, 6.169137402448035]
DEL_samples_rate_AAAI: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
1.1590018490880714
"""




