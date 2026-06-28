import matplotlib.pyplot as plt
import re

def parse_loss_file(file_path='loss_history.txt'):
    train_losses = []
    val_losses = []

    train_regex = re.compile(r"Train:([\d.]+)")
    val_regex = re.compile(r"Val:([\d.]+)")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                train_match = train_regex.search(line)
                val_match = val_regex.search(line)
                
                if train_match:
                    train_losses.append(float(train_match.group(1)))
                if val_match:
                    val_losses.append(float(val_match.group(1)))
    except FileNotFoundError:
        return [], []
                
    return train_losses, val_losses

def plot_loss_history(train_losses, val_losses):
    if not train_losses or not val_losses:
        return

    epochs = range(1, len(train_losses) + 1)
    
    # Tạo một figure chứa 2 subplot (1 hàng, 2 cột)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Training Results', fontsize=16)

    # --- Biểu đồ Training Loss (bên trái) ---
    ax1.plot(epochs, train_losses, 'b-o', markersize=4)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.grid(True)
    
    # --- Biểu đồ Validation Loss (bên phải) ---
    ax2.plot(epochs, val_losses, 'r-o', markersize=4)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Điều chỉnh layout để có chỗ cho tiêu đề chung
    
    plt.show()

if __name__ == '__main__':
    train_loss_history, val_loss_history = parse_loss_file()
    plot_loss_history(train_loss_history, val_loss_history)