import csv
import os


def write_result(results_file, result):
    """Writes results to a csv file."""
    with open(results_file, "a+", newline="") as csvfile:
        field_names = result.keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.stat(results_file).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)


def report_regression(
    dataset, method, loss, p_train, gamma, seed, save, X, y, alpha, save_dir="results"
):
    if "ru_reg" not in method:
        gamma = 1
    all_res = []
    for i in range(len(X)):
        results = {"X": X[i].item(), "y": y[i].item(), "alpha": alpha[i].item()}
        all_res.append(results)
    results_file = (
        save_dir
        + "/"
        + save
        + "_{}_{}_{}_{}_p_train_{}_seed_{}".format(
            dataset, method, gamma, loss, p_train, seed
        )
        + ".csv"
    )

    for i in range(len(all_res)):
        write_result(results_file, all_res[i])


def report_results(
    results,
    dataset,
    n_train,
    d,
    p_train,
    method,
    loss,
    gamma,
    use_train_weights,
    model_class,
    seed,
    save,
    save_dir="results",
):
    if "ru_reg" not in method:
        gamma = 1
    results_file = save_dir + "/" + save + ".csv"

    for i in range(len(results)):
        if dataset in ["survey", "mimic"]:
            full_result = {
                "dataset": dataset,
                "method": method,
                "gamma": gamma,
                "loss": loss,
                "use_train_weights": use_train_weights,
                "model_class": model_class,
                "seed": seed,
            }
        else:
            full_result = {
                "dataset": dataset,
                "n_train": n_train,
                "d": d,
                "p_train": p_train,
                "method": method,
                "gamma": gamma,
                "loss": loss,
                "model_class": model_class,
                "use_train_weights": use_train_weights,
                "seed": seed,
            }
        full_result.update(results[i])
        write_result(results_file, full_result)
