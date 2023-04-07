import logging

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('risk_factor_analysis')


def prepare_features(df):
    for i in range(24):
        col_name = f'sen_{i}'
        df[f'day_m1_{col_name}'] = df.groupby('ticker')[col_name].shift(1)
        col_name = f'euc_w_{i}'
        df[f'day_m1_{col_name}'] = df.groupby('ticker')[col_name].shift(1)
        df.drop(columns=[f'euc_w_sen_{i}'], inplace=True)

    return df


def create_and_test_binary_classification_model(hourly_weight_sim_df, return_df):
    def predict(X_train, X_test, y_train, y_test, model):
        print(model)
        logger.info('Fitting model...')
        model.fit(X_train, y_train)
        logger.info('Predicting...')
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        logger.info('Calculating metrics...')
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        confusion = confusion_matrix(y_test, y_pred_test)
        logger.info(model)
        logger.info(f"Train accuracy: {train_accuracy}")
        logger.info(f"Test accuracy: {test_accuracy}")
        logger.info(f"Confusion matrix:\n{confusion}")

    df = hourly_weight_sim_df.merge(return_df, left_index=True, right_index=True)
    df = prepare_features(df)
    df['target'] = df['return'] > 0.0
    df.drop(columns=['return'], inplace=True)
    df.dropna(inplace=True)
    print(df)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    logger.debug('Scaling data...')
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    predict(X_train, X_test, y_train, y_test,
            MLPClassifier(solver='adam', hidden_layer_sizes=(200, 200), alpha=0.00001, max_iter=200,
                          early_stopping=True, n_iter_no_change=20, verbose=True))

    return df
