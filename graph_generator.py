import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_data(algorithm_name):
    summary_file = f"weka-results/metrics-summary/{algorithm_name}_summary.csv"
    confusion_matrix_file = (
        f"weka-results/confusion-matrix/{algorithm_name}_confusion_matrix.csv"
    )

    summary_df = pd.read_csv(summary_file)
    confusion_matrix_df = pd.read_csv(confusion_matrix_file, index_col=0)
    return summary_df, confusion_matrix_df


def process_summary_df(summary_df):
    summary_df["Valor"] = (
        summary_df["Valor"]
        .str.replace("%", "")
        .str.replace(",", ".")
        .astype(float)
    )
    return summary_df


def plot_execution_time(execution_times):
    plt.figure(figsize=(14, 7))
    execution_times_df = pd.DataFrame(
        list(execution_times.items()),
        columns=["Algorithm", "Time Taken (seconds)"],
    )
    sns.barplot(
        x="Algorithm", y="Time Taken (seconds)", data=execution_times_df
    )
    plt.title("Execution Time Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Time Taken (seconds)")
    plt.xticks(rotation=45)
    plt.show()


def plot_accuracy_error(accuracy_error):
    plt.figure(figsize=(14, 7))

    # Precisão
    plt.subplot(121)
    sns.barplot(x="Algorithm", y="Accuracy", data=accuracy_error)
    plt.title("Accuracy Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)

    # Erro
    plt.subplot(122)
    sns.barplot(x="Algorithm", y="Error", data=accuracy_error)
    plt.title("Error Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Error")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_combined_confusion_matrix(confusion_matrices):
    combined_matrix = pd.concat(
        [df for df in confusion_matrices.values()], ignore_index=True
    )
    plt.figure(figsize=(14, 7))
    sns.heatmap(combined_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Combined Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


algorithm_names = [
    "1-decision_table",
    "2-jrip",
    "3-one_r",
    "4-part",
    "5-zero_r",
    "6-decision_stump",
    "7-hoeffding_tree",
    "8-j48",
    "9-lmt",
    "10-random_forest",
    "11-random_tree",
    "12-rep_tree",
]

execution_times = {}
accuracy_error = {"Algorithm": [], "Accuracy": [], "Error": []}
confusion_matrices = {}

for algorithm_name in algorithm_names:
    summary_df, confusion_matrix_df = load_data(algorithm_name)
    summary_df = process_summary_df(summary_df)

    time_taken_row = summary_df.loc[
        summary_df["Métrica"] == "Time taken (in seconds)"
    ]
    if not time_taken_row.empty:
        execution_times[algorithm_name] = time_taken_row["Valor"].values[0]
    else:
        print(f"No 'Time taken' entry found for {algorithm_name}. Skipping.")
        continue

    accuracy_row = summary_df.loc[
        summary_df["Métrica"] == "Correctly Classified Instances"
    ]
    if not accuracy_row.empty:
        accuracy = (
            accuracy_row["Porcentagem sobre o total"]
            .values[0]
            .replace("%", "")
            .replace(",", ".")
        )
        accuracy_error.loc[len(accuracy_error)] = [
            algorithm_name,
            float(accuracy),
            None,
        ]  # None para Error, pois estamos apenas lidando com Accuracy aqui
    else:
        print(f"No 'Accuracy' entry found for {algorithm_name}. Skipping.")
        continue

    error_row = summary_df.loc[
        summary_df["Métrica"] == "Incorrectly Classified Instances"
    ]
    if not error_row.empty:
        error = (
            error_row["Porcentagem sobre o total"]
            .values[0]
            .replace("%", "")
            .replace(",", ".")
        )
        accuracy_error.loc[len(accuracy_error)] = [
            algorithm_name,
            None,
            float(error),
        ]  # None para Accuracy, pois estamos apenas lidando com Error aqui
    else:
        print(f"No 'Error' entry found for {algorithm_name}. Skipping.")
        continue

    confusion_matrices[algorithm_name] = confusion_matrix_df

# Plotar o tempo de execução
plot_execution_time(execution_times)

# Plotar precisão e erros
plot_accuracy_error(pd.DataFrame(accuracy_error))

# Plotar matriz de confusão combinada
plot_combined_confusion_matrix(confusion_matrices)
