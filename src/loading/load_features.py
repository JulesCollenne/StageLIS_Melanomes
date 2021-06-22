from loading.DataLoader import DataLoader


def get_features():
    ohe = OneHotEncoder()

    featuresNames = ['beta_quad_RGB', 'beta_quad_HSV', 'beta_quad_Lab', 'beta_quad_YCrCb',
                     'histos', 'histos_HSV', 'histos_Lab', 'histos_YCrCb',
                     'SPML_4_2', 'SPML_4_2_HSV', 'SPML_4_2_cielab', 'SPML_4_2_ycbcr',
                     'SPO', 'SP60_4', 'beta']

    loader = DataLoader(featuresNames)
    X_train, y_train, X_test, y_test = loader.load(centering=True)

    ohe.fit(y_train.reshape((-1, 1)))
    y_trainOH = ohe.transform(y_train.reshape((-1, 1))).toarray()
    y_testOH = ohe.transform(y_test.reshape((-1, 1))).toarray()
    return X_train, y_trainOH, X_test, y_testOH
