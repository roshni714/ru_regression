import numpy as np
import loss
from xgboost import XGBRegressor
from data_loaders.dataloader_utils import unpack_tensors
from loss import compute_losses


class XGBoostModel:
    def __init__(self, loss, epochs, seed):
        self.loss = loss
        self.epochs = int(epochs)
        self.seed = seed
        self.model = None

    def _get_objective(self):

        def weighted_mse_eval(preds, dtrain, weights):
            labels = dtrain
            l = ((preds - labels)**2 * weights).mean()
            return l
        
        def weighted_poisson_nll_eval(preds, dtrain, weights):
            labels = dtrain
            l = (loss.poisson_nll_loss(preds, labels) * weights).mean()
            return l
        

        def weighted_mse_loss(preds, labels, weights):
            grad = 2 * (preds - labels) * weights
            hess = 2 * weights
            return grad, hess
                
        def weighted_poisson_nll_loss(preds, labels, weights):
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
            r_train = np.ones_like(y_train)/len(y_train)
        if r_val is None:
            r_val = np.ones_like(y_val)/len(y_val)

        objective, eval_metric = self._get_objective()

        self.model = XGBRegressor(
            objective=objective,
            n_estimators=self.epochs,
            random_state=self.seed,
            eval_metric= lambda preds, dtrain: eval_metric(preds, dtrain, r_val.flatten()))

        self.model.fit(
            X_train,
            y_train.flatten(),
            sample_weight=r_train,
            eval_set=[(X_val, y_val.flatten())],
            verbose=False,
            )

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError(
                "XGBoost model must be fit before calling predict"
            )
        return self.model.predict(X).reshape(-1, 1)

    def summarize_test(self, test_loader, n_bootstrap=500):
        X, y_true, r = unpack_tensors(test_loader)
        y_hat = self.predict(X)

        inner_loss, mse_loss = compute_losses(
            y_hat=y_hat,
            y_true=y_true,
            loss_name=self.loss,
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
