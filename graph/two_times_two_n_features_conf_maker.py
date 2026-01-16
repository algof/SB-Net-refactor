import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = {
    'n_3_1': [[6148254, 51], [58, 293268]],
    'n_3_2': [[263376, 12], [2, 29929]]
}

conf_labels = {
    'n_3_1': ['normal', 'botnet'],
    'n_3_2': ['botnet', 'botnet_spam']
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
    plt.savefig(f'graph/n features 2 x 2/{key}.png', bbox_inches='tight')
    plt.close()