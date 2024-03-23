import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np


class LGBMRegressor:
    """
    Regressor Class of LightGBM

    Document URL:
        * https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api
    """

    def __init__(self):
        self.model = None
        self.callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(100),
        ]

    def train(
        self,
        params,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        feature_name="auto",
        num_boost_round=100,
        cv=0,
        seed=42,
    ):
        """
        Args:
            X_train: pd.Series
            y_train: pd.Series
        """
        assert cv >= 0, "cv must be an integer value greater than or equal to 0.."
        assert cv > 0 or (
            X_val is not None and y_val is not None
        ), "If cv=0, specify X_val, y_val.."
        # training dataset
        train_data = lgb.Dataset(
            data=X_train,
            label=y_train,
            feature_name=feature_name,
            free_raw_data=False,
        )

        # validation dataset
        valid_data = lgb.Dataset(
            data=X_val,
            label=y_val,
            feature_name=feature_name,
        )

        if cv == 0:
            self.model = lgb.train(
                params=params,
                train_set=train_data,
                valid_sets=[valid_data],
                num_boost_round=num_boost_round,
                callbacks=self.callbacks,
            )
        else:
            self.model = lgb.cv(
                params=params,
                train_set=train_data,
                num_boost_round=num_boost_round,
                callbacks=self.callbacks,
                nfold=cv,
                seed=seed,
                stratified=False,  # It is required to set False for regression
                return_cvbooster=True,  # Default: False
            )

    def predict(self, X):
        """
        Predict test data

        Args:
            X: ndarray, (n_sample, n_features)
                input data after preprocessing

        Return:
            prediction: ndarray, (n_sample,)
        """
        return self.model.predict(np.array(X), num_iteration=self.model.best_iteration)

    def plot_learning_curve(self, X, y, model_list, savepath=None):
        """
        Learning Curve

        Args:
            X: ndarray, (n_sample, n_features)
                input data after preprocessing
            y: ndarray, (n_sample,)
                target data
            model_list: list[model instance]
                model instance list
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        for model in model_list:
            model.model.plot_metric(
                ax=ax,
                figsize=(10, 10),
                title="Learning Curve",
                xlabel="Iterations",
                ylabel="RMSE",
            )
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()


if __name__ == "__main__":
    pass
