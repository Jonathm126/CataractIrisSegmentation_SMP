import matplotlib.pyplot as plt

def plot_losses(train_losses, valid_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Loss')
    plt.title('Dice Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()