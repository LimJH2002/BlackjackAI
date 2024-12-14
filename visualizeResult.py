import matplotlib.pyplot as plt
import os
import ast

if __name__ == "__main__":
    with open("mobilenet_results.txt", mode="r") as file:
        lines = file.readlines()
        train_losses = ast.literal_eval(lines[0][len("Training losses: ") :].strip())
        val_accuracies = ast.literal_eval(lines[1][len("Testing accuracies: ") :].strip())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig("mobilenet_results.png")
    plt.close()