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

                plt.plot(x_values, eval_returns, 'o-', label='Evaluation Returns')
                plt.plot(x_values, train_returns, 'o-', label='Training Returns')

                plt.legend()
                plt.title('Training and Evaluation Returns Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Average Return')
                plt.xticks(x_values)

                plt.show()

def generate_q4_plot(base_dir='hw2/data'):

    horizon5_path = os.path.join(base_dir, 'hw2_q4_reacher_horizon5/log_data.csv')
    horizon15_path = os.path.join(base_dir, 'hw2_q4_reacher_horizon15/log_data.csv')
    horizon30_path = os.path.join(base_dir, 'hw2_q4_reacher_horizon30/log_data.csv')

    numseq100_path = os.path.join(base_dir, 'hw2_q4_reacher_numseq100/log_data.csv')
    numseq1000_path = os.path.join(base_dir, 'hw2_q4_reacher_numseq1000/log_data.csv')

    ensemble1_path = os.path.join(base_dir, 'hw2_q4_reacher_ensemble1/log_data.csv')
    ensemble3_path = os.path.join(base_dir, 'hw2_q4_reacher_ensemble3/log_data.csv')

    horizon5_returns = pd.read_csv(horizon5_path)['trainer/Eval_AverageReturn']
    horizon15_returns = pd.read_csv(horizon15_path)['trainer/Eval_AverageReturn']
    horizon30_returns = pd.read_csv(horizon30_path)['trainer/Eval_AverageReturn']

    x_values = np.arange(1,1+len(horizon5_returns))

    plt.plot(x_values, horizon5_returns, label='5 steps')
    plt.plot(x_values, horizon15_returns, label='15 steps')
    plt.plot(x_values, horizon30_returns, label='30 steps')

    plt.legend()
    plt.title('Return with differents horizon steps')
    plt.xlabel('Epochs')
    plt.ylabel('Average Return')
    plt.xticks(x_values)

    plt.show()
    plt.clf()

    numseq100_returns = pd.read_csv(numseq100_path)['trainer/Eval_AverageReturn']
    numseq1000_returns = pd.read_csv(numseq1000_path)['trainer/Eval_AverageReturn']

    plt.plot(x_values, numseq100_returns, label='100 sequences')
    plt.plot(x_values, numseq1000_returns, label='1000 sequences')

    plt.legend()
    plt.title('Return with differents actions sequences')
    plt.xlabel('Epochs')
    plt.ylabel('Average Return')
    plt.xticks(x_values)

    plt.show()
    plt.clf()

    ensemble1_returns = pd.read_csv(ensemble1_path)['trainer/Eval_AverageReturn']
    ensemble3_returns = pd.read_csv(ensemble3_path)['trainer/Eval_AverageReturn']

    plt.plot(x_values, ensemble1_returns, label='No ensemble')
    plt.plot(x_values, ensemble3_returns, label='Ensemble of 3')

    plt.legend()
    plt.title('Return with differents ensembles')
    plt.xlabel('Epochs')
    plt.ylabel('Average Return')
    plt.xticks(x_values)

    plt.show()



def generate_q5_plot(base_dir='hw2/data'):

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
    generate_q4_plot()
    generate_q5_plot()
