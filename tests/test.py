def get_df(size=(3, 200)):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame()
    for i in range(size[0]):
        df['f{}'.format(i+1)] = 100 * np.random.randn(size[1])
    df['target'] = np.random.randint(0, 2, size[1])
    df['predicted'] = np.random.randint(0, 1000, size[1]) / 1000
    df['date'] = np.random.randint(0, 4, size[1])
    return df


# def test_psi():
#     from riskpy.utilities.common_metrics import psi
#     df1 = get_df()
#     df2 = get_df()
#     p1 = psi(df1, df2, 'target')
#     p2 = psi(df1, df2)
#     return p1, p2
# test_psi()
#
#
# def test_gini():
#     from riskpy.utilities.common_metrics import gini
#     df = get_df()
#     return gini(y_true=df['target'], y_pred=df['predicted'])
# test_gini()
#
#
# def test_gini_dynamic():
#     from riskpy.graphs.graphs import plot_pd_gini_in_time
#     plot_pd_gini_in_time(get_df(), 'target', 'predicted', 'date', name='', zone_borders=[0, 0.4, 0.6, 1], size=[17, 10], colorful=True)
#     plot_pd_gini_in_time(get_df(), 'target', 'predicted', 'date', name='name', zone_borders=[0, 0.3, 0.8, 1], size=[297, 210], colorful=False, grid=True)
# test_gini_dynamic()
#
#
# def test_roc():
#     from riskpy.graphs.graphs import roc
#     df = get_df()
#     roc(y_true=df['target'], y_pred=df['predicted'])
# test_roc()
#
#
# def test_vif():
#     from riskpy.modeling.modeler import VIF
#     df = get_df()
#     vif = VIF()
#     vif.vif(df)
#     vif.fit(df, verbose=False)
#     vif.transform(df, verbose=False)
#     vif._end_columns
#     vif._init_columns
#     vif.treshhold
#     vif._drop_cols
# test_vif()
#
#
# def test_correlations():
#     from riskpy.modeling.modeler import Correlations
#     cor = Correlations(threshold=0.95).fit(get_df(), drop_cols=['target'], verbose=True, plot=True, figsize=(10, 10),
#                                            title='title')
#     cor.transform(get_df())
#     cor = Correlations(threshold=0.95).fit(get_df(), drop_cols=['target'], verbose=True, plot=True)
#     cor.transform(get_df())
#     cor = Correlations().fit(get_df())
#     cor.transform(get_df())
# test_correlations()


def test_modeler():
    from riskpy.modeling.modeler import Modeler
    import numpy as np
    m = Modeler(train_dataset=get_df(), test_dataset=get_df(), target='target', oot_dataset=get_df(),  period='date',
            kind='class', mode='sklearn').fit(['f1', 'f2', 'f3'], disp=False, n_iter=100)
    print(m.get_score('train'))
    print(m.get_score('test'))
    print(m.get_score('oot'))

    print(m.predict('train', True)[1][:3])
    print(m.predict('test')[:3])
    print(m.predict('oot')[-3:])

    print(np.mean(m.cv_score(['f1', 'f2', 'f3'])[0]), np.mean(m.cv_score(['f1', 'f2', 'f3'])[1]))
    return m._fitted_model
test_modeler()
