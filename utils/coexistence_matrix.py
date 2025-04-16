import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot(object_columns: list, data_path: str, output_dir: str):
    # Absolute Path Conversion
    data_path = os.path.abspath(data_path)
    output_dir = os.path.abspath(output_dir)

    report_df = pd.read_csv(data_path)

    # Convert ground truth object counts to binary (presence/absence of each object)
    binary_data = report_df[object_columns].apply(lambda col: np.where(col > 0, 1, 0))

    # Calculate the co-existence matrix for ground truth
    coexistence_matrix = binary_data.T.dot(binary_data)

    # Plot the co-existence matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(coexistence_matrix, annot=True, fmt='d', cmap="coolwarm", cbar=True)
    plt.xlabel('Object ID')
    plt.ylabel('Object ID')
    plt.title('Co-existence Matrix')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{output_dir}/coexistence_matrix.png')
