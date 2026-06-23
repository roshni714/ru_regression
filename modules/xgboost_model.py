import numpy as np
import loss
from xgboost import callback
from xgboost import XGBRegressor
from data_loaders.dataloader_utils import unpack_tensors
from loss import compute_losses


class XGBoostModel:
    def __init__(self, loss, epochs, seed, y_mean, y_scale=1.0):
        self.loss = loss
        self.epochs = int(epochs)
        self.seed = seed
        self.y_mean = y_mean
        self.y_scale = y_scale
        self.model = None

    def _get_objective(self):

        def weighted_mse_eval(dtrain, preds, weights):
            labels = dtrain
            l = ((preds - labels)**2 * weights).sum()/weights.sum()

            return l
        
        def weighted_poisson_nll_eval(dtrain, preds, weights):
            labels = dtrain
            l = ((-preds * labels + np.exp(preds)) * weights).sum()/weights.sum()
            return l
        

        def weighted_mse_loss(labels, preds, weights):
            grad = 2 * (preds - labels) * weights
            hess = 2 * np.ones_like(labels) * weights
            return grad, hess
                
        def weighted_poisson_nll_loss(labels, preds, weights):
            grad = (np.exp(preds) - labels) * weights
            hess = np.exp(preds) * weights
            return grad, hess
        
        if self.loss == "squared_loss":
            return weighted_mse_loss, weighted_mse_eval
        if self.loss == "poisson_nll":
            return weighted_poisson_nll_loss, weighted_poisson_nll_eval
        raise ValueError("Unsupported loss '{}' for xgboost".format(self.loss))

    def fit(self, train_loader, val_loader):
        X_train, y_train, r_train = unpack_tensors(train_loader)
        X_val, y_val, r_val = unpack_tensors(val_loader)

        if r_train is None:
            r_train = np.ones_like(y_train)
        if r_val is None:
            r_val = np.ones_like(y_val)

        objective, eval_metric = self._get_objective()

        if X_train.shape[1] == 1:
            max_depth = 3
        else:
            max_depth = 6

        self.model = XGBRegressor(
            objective= lambda dtrain, preds: objective(dtrain, preds, r_train.flatten()),
            random_state=self.seed,
            n_estimators=5000,
            learning_rate=0.001,
            eval_metric= lambda dtrain, preds: eval_metric(dtrain, preds, r_val.flatten()),
            disable_default_eval_metric=True,
            max_depth=max_depth,
        )

        self.model.fit(
            X_train,
            y_train.flatten(),
            verbose=True,
            eval_set=[(X_val, y_val.flatten())],
            early_stopping_rounds=20,
            )
        
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError(
                "XGBoost model must be fit before calling predict"
            )
        return self.model.predict(X, iteration_range=(0, self.model.best_iteration + 1)).reshape(-1, 1)

    def summarize_test(self, test_loader, n_bootstrap=500):
        print("Testing...")
        X, y_true, r = unpack_tensors(test_loader)
        y_hat = self.predict(X)

        inner_loss, mse_loss = compute_losses(
            y_hat=y_hat,
            y_true=y_true,
            y_mean=self.y_mean,
            loss_name=self.loss,
            y_scale=self.y_scale
        )
        ru_loss = inner_loss

        n = y_true.shape[0]
        inner_vals = np.zeros(n_bootstrap)
        mse_vals = np.zeros(n_bootstrap)
        ru_vals = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            if r is None:
                inner_vals[i] = inner_loss[idx].mean()
                mse_vals[i] = mse_loss[idx].mean()
                ru_vals[i] = ru_loss[idx].mean()
            else:
                w = r[idx]
                w_sum = np.sum(w)
                inner_vals[i] = np.sum(inner_loss[idx] * w) / w_sum
                mse_vals[i] = np.sum(mse_loss[idx] * w) / w_sum
                ru_vals[i] = np.sum(ru_loss[idx] * w) / w_sum

        if r is None:
            test_loss = inner_loss.mean()
            test_mse = mse_loss.mean()
            test_ru = ru_loss.mean()
        else:
            w_sum = np.sum(r)
            test_loss = np.sum(inner_loss * r) / w_sum
            test_mse = np.sum(mse_loss * r) / w_sum
            test_ru = np.sum(ru_loss * r) / w_sum

        return {
            "test_ru_loss": float(test_ru),
            "test_ru_loss_se": float(ru_vals.std()),
            "test_loss": float(test_loss),
            "test_loss_se": float(inner_vals.std()),
            "test_mse": float(test_mse),
            "test_mse_se": float(mse_vals.std()),
        }
