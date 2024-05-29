if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    import numpy as np
    from typing import List
    
    
    def get_data(key: str,
                 content: List):
        return [element for item in content for element in item.get(key, 0)]
    
    with open('loss.json', 'r') as file:
        contents = json.load(file)
        train_step_loss = get_data("train_step_loss", contents)
        train_precision_score = get_data("train_precision_score", contents)
        train_recall_score = get_data("train_recall_score", contents)
        train_F1_score = get_data("train_F1_score", contents)
        valid_step_loss = get_data("valid_step_loss", contents)
        valid_precision_score = get_data("valid_precision_score", contents)
        valid_recall_score = get_data("valid_recall_score", contents)
        valid_F1_score = get_data("valid_F1_score", contents)
    
    # Create x-axis values (epoch numbers)
    epochs = list(range(len(contents)))
    train_steps = len(train_step_loss) // len(contents)
    valid_steps = len(valid_step_loss) // len(contents)
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axs[0, 0].plot(np.arange(len(train_step_loss)), train_step_loss, label='Train Loss')
    axs[0, 0].set_title('Train Loss')
    axs[0, 0].set_xlabel('Steps')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Validation loss
    axs[0, 1].plot(np.arange(len(valid_step_loss)), valid_step_loss, label='Validation Loss')
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Steps')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    
    # Training F1 score
    axs[1, 0].plot(np.arange(len(train_F1_score)), train_F1_score, label='Train F1 Score')
    axs[1, 0].set_title('Train F1 Score')
    axs[1, 0].set_xlabel('Steps')
    axs[1, 0].set_ylabel('F1 Score')
    axs[1, 0].legend()
    
    # Validation F1 score
    axs[1, 1].plot(np.arange(len(valid_F1_score)), valid_F1_score, label='Validation F1 Score')
    axs[1, 1].set_title('Validation F1 Score')
    axs[1, 1].set_xlabel('Steps')
    axs[1, 1].set_ylabel('F1 Score')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()