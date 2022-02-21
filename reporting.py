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


def report_results(model, dataset, loss, gamma, seed, save, save_dir="results"):
    result = {
        "dataset": dataset,
        "test_loss": model.test_loss,
        "loss": loss,
        "gamma": gamma,
        "seed": seed,
    }
    results_file = save_dir + "/" + save + ".csv"
    write_result(results_file, result)
