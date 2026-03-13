from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load dataset
(train_images, train_labels), _ = mnist.load_data()

# Count digits
digit_counts = pd.Series(train_labels).value_counts()

# Plot distribution
sns.barplot(x=digit_counts.index, y=digit_counts.values)
plt.title("Digit Distribution")
plt.show()

plt.figure(figsize=(6,6))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i], cmap="gray")
    plt.title(f"Label: {train_labels[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()