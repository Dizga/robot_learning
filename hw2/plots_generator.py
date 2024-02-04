import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_plots(base_dir='hw2/data'):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'log_data.csv':
                csv_path = os.path.join(root, file)
                log_data = pd.read_csv(csv_path)
                
                eval_returns = log_data['trainer/Eval_AverageReturn']
                train_returns = log_data['trainer/Train_AverageReturn']
                
                x_values = np.arange(1,1+len(eval_returns))

                plt.plot(x_values, eval_returns, 'go-', label='Evaluation Returns')
                plt.plot(x_values, train_returns, 'bo-', label='Training Returns')

                plt.legend()
                plt.title('Training and Evaluation Returns Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Average Return')
                plt.xticks(x_values)

                plt.show()

if __name__ == '__main__':
    generate_plots()
