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

def generate_cem_v_rs_plot(base_dir='hw2/data'):

    rs_file = os.path.join(base_dir, 'hw2_q5_cheetah_random/log_data.csv')
    cem2_file = os.path.join(base_dir, 'hw2_q5_cheetah_cem_2/log_data.csv')
    cem4_file = os.path.join(base_dir, 'hw2_q5_cheetah_cem_4/log_data.csv')

    rs_returns = pd.read_csv(rs_file)['trainer/Eval_AverageReturn']
    cem2_returns = pd.read_csv(cem2_file)['trainer/Eval_AverageReturn']
    cem4_returns = pd.read_csv(cem4_file)['trainer/Eval_AverageReturn']

    x_values = np.arange(1,1+len(rs_returns))

    plt.plot(x_values, rs_returns, label='Random shooting')
    plt.plot(x_values, cem2_returns, label='CEM 2')
    plt.plot(x_values, cem4_returns, label='CEM 4')

    plt.legend()
    plt.title('Evaluation Returns Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Average Return')
    plt.xticks(x_values)

    plt.show()

if __name__ == '__main__':
    generate_plots()
    generate_cem_v_rs_plot()
