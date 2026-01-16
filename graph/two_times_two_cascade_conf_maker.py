import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = {
    'RF_RF_1': [[6148253, 52], [481, 292845]],
    'RF_RF_2': [[262958, 14], [2, 29923]]
}

conf_labels = {
    'RF_RF_1': ['normal', 'botnet'],
    'RF_RF_2': ['botnet', 'botnet_spam']
}

for key, value in conf_matrix.items():
    print(f"{key}: {value}")
    plt.figure(figsize=(4, 3))
    sns.heatmap(value, annot=True, fmt='d', cmap='Blues',
                xticklabels=conf_labels[key],
                yticklabels=conf_labels[key])
    
    # Add axis labels
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    # plt.show()
    # Optional: 
    plt.savefig(f'graph/cascade result 2 x 2/{key}.png', bbox_inches='tight')
    plt.close()