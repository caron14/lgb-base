import lightgbm as lgb
import numpy as np


class LGBM_Regressor:
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
        学習曲線の描画

        Args:
            X: ndarray, (n_sample, n_features)
                前処理後の説明変数データ
            y: ndarray, (n_sample,)
                目的変数ラベル
            model_list: list[model instance]
                モデルインスタンスのリスト
        """
        figsize = (10, 15)
        nrows, ncols = len(model_list) // 2 + len(model_list) % 2, 2
        # nrows, ncols = 3, 2
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharey=True,
        )

        # F1値のscorer objectを作成
        scorer = make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred))
        # 設定
        common_params = {
            "X": X,
            "y": y,
            "train_sizes": np.linspace(0.1, 1.0, 5),
            "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
            "score_type": "both",
            "scoring": scorer,
            "score_name": "F1",
            "line_kw": {"marker": "o"},
            "std_display_style": "fill_between",
        }

        model_idx = 0
        for i in range(nrows):
            for j in range(ncols):
                assert f"{model_list[model_idx].__class__.__name__}" in [
                    "SVC",
                    "LogisticRegression",
                    "LGBMClassifier",
                ], "サポート手法外です.."

                LearningCurveDisplay.from_estimator(
                    model_list[model_idx], **common_params, ax=ax[i][j]
                )
                handles, label = ax[i][j].get_legend_handles_labels()
                ax[i][j].legend(handles[:2], ["train", "val"], loc="lower right")
                ax[i][j].set_ylim(0.5, 1)
                model_idx += 1
                # Error対策: モデルリストのindex外参照
                if model_idx >= self.config.n_kfold:
                    break

        if savepath is not None:
            plt.savefig(savepath)
        plt.close()


if __name__ == "__main__":
    pass
