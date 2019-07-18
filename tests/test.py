def get_df(size=(5, 2000)):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame()
    for i in range(size[0]):
        df['f{}'.format(i+1)] = 100 * np.random.randn(size[1])
    df['target'] = np.random.randint(0, 2, size[1])
    df['predicted'] = np.random.randint(0, 1000, size[1]) / 1000
    df['date'] = np.random.randint(0, 4, size[1])
    return df


def test_psi():
    from riskpy.utilities.common_metrics import psi
    df1 = get_df()
    df2 = get_df()
    p1 = psi(df1, df2, 'target')
    p2 = psi(df1, df2)
    return p1, p2
test_psi()


def test_gini():
    from riskpy.utilities.common_metrics import gini
    df = get_df()
    return gini(y_true=df['target'], y_pred=df['predicted'])
test_gini()


def test_gini_dynamic():
    from riskpy.graphs.graphs import plot_pd_gini_in_time
    plot_pd_gini_in_time(get_df(), 'target', 'predicted', 'date', name='', zone_borders=[0, 0.4, 0.6, 1], size=[17, 10], colorful=True)
    plot_pd_gini_in_time(get_df(), 'target', 'predicted', 'date', name='name', zone_borders=[0, 0.3, 0.8, 1], size=[297, 210], colorful=False, grid=True)
test_gini_dynamic()


def test_roc():
    from riskpy.graphs.graphs import roc
    df = get_df()
    roc(y_true=df['target'], y_pred=df['predicted'])
test_roc()


def test_vif():
    from riskpy.modeling.modeler import VIF
    df = get_df()
    vif = VIF()
    vif.vif(df)
    vif.fit(df, verbose=False)
    vif.transform(df, verbose=False)
    vif._end_columns
    vif._init_columns
    vif.treshhold
    vif._drop_cols
test_vif()


def test_correlations():
    from riskpy.modeling.modeler import Correlations
    corr = Correlations(get_df(), get_df(), get_df(), target='target')
    corr.corr(get_corr_tables=True, all_tables=True)

    corr.fit()
    corr.transform()

    corr.get_train(verbose=False)
    corr.get_test(verbose=False)
    corr.get_oot(verbose=False)

    corr = Correlations(get_df(), get_df(), get_df(), target='target')
    corr.corr(get_corr_tables=False, all_tables=True)

    corr.fit()
    corr.transform()

    corr.get_train(verbose=False)
    corr.get_test(verbose=False)
    corr.get_oot(verbose=False)

    corr = Correlations(get_df(), get_df(), get_df(), target='target')
    corr.corr(get_corr_tables=True, all_tables=False)

    corr.fit()
    corr.transform()

    corr.get_train(verbose=False)
    corr.get_test(verbose=False)
    corr.get_oot(verbose=False)

    corr._train
    corr._test
    corr._oot
    corr._target
    corr._train_corr
    corr._test_corr
    corr._oot_corr
    corr._col_to_del
test_correlations()


