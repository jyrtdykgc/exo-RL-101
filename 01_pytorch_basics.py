"""
Part 1: PyTorch Basics
=======================
Goal: train a tiny model to learn the line y = 2x + 1 from noisy data.

This is the "hello world" of PyTorch. Every ML model you'll ever train,
including the RL agent in Part 3, follows this exact same pattern:

    1. Make a prediction (forward pass)
    2. Measure how wrong it was (loss)
    3. Figure out which direction to adjust each parameter (backward pass)
    4. Nudge the parameters a little in that direction (optimizer step)
    5. Repeat

Fill in the 4 TODOs below, then run:  python 01_pytorch_basics.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. Make some fake data: y = 2x + 1, plus a little random noise
# ---------------------------------------------------------------------------
x = torch.linspace(-5, 5, 100).unsqueeze(1) # shape (100, 1)
true_y = 2 * x + 1
noise = torch.randn_like(true_y) * 1.5
y = true_y + noise

# ---------------------------------------------------------------------------
# 2. Define the model
# ---------------------------------------------------------------------------
# nn.Linear(1, 1) is a model with exactly two learnable numbers: 
# a slope (weight) and an intercept (bias). Training will adjust both until the line
# fits the data. Every PyTorch models is built the same way:
# a class with layers defined in __init__ and the forward pass in forward().
class LineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # `forward` is where you describe what the model does to its
        # input. We only have one layer, self.linear (defined above), and
        # we want to run `x` through it and send back whatever it outputs.
        #
        # A layer is called just like a function: you put your input
        # inside parentheses right after its name (see the cheat sheet's
        # "Calling a layer or model" section for the general pattern).
        #
        # TODO 1: replace the line below with a `return` statement that
        # calls self.linear on x and returns the output
        return self.linear(x)


model = LineModel()

# ---------------------------------------------------------------------------
# 3. Loss function and optimizer
# ---------------------------------------------------------------------------
# The loss function tells us how far the model's predictions are from
# the correct answers. A smaller loss means the model is doing better.

# MSE (Mean Squared Error) is commonly used for regression problems,
# where the model is predicting a number.
loss_fn = nn.MSELoss()

# The optimizer is responsible for improving the model.
# It uses the gradients calculated by .backward() to update the model's
# weight and bias so that the loss gets smaller over time.

# In this project, we use SGD (Stochastic Gradient Descent) method to make small adjustments 
# to the model's parameters after each training step.

# lr (learning rate) controls how big each adjustment is.
# Here, 0.01 means the optimizer takes relatively small steps.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ---------------------------------------------------------------------------
# 4. The training loop
# ---------------------------------------------------------------------------
losses = []
num_epochs = 100

for epoch in range(num_epochs):
    # Step 1: pass forward. Use the model to make predictions for our inputs.
    # The model uses its current parameters to produce these predictions.
    # As training progresses, we expect the predictions to improve.
    predictions = model(x)

    # Step 2: calculate the loss. 
    # The loss tells us how far the model's predictions are from the correct answers. '
    # `loss_fn` takes two arguments:
    #   1. `predictions` — what the model predicted
    #   2. `y`           — the correct answers (targets)
    # It returns a single tensor containing the loss.

    # TODO 2: call `loss_fn` with `predictions` and `y` as the second argument. 
    # to calculate the loss. Store the result in a variable named `loss`.
    loss = loss_fn(predictions, y)

    # Step 3: Backward pass + parameter update.
    # Clear gradients from the previous training step. PyTorch accumulates
    # gradients by default, so we reset them before computing new ones.
    optimizer.zero_grad()

    # Step 4. Compute gradients for every trainable parameter.
    # A gradient tells us how changing a parameter would change the loss:
    #   - Positive gradient  -> decreasing the parameter would reduce the loss.
    #   - Negative gradient  -> increasing the parameter would reduce the loss.
    # The larger the gradient, the bigger the parameter's effect on the loss.

    # PyTorch uses backpropagation to calculate these gradients efficiently,
    # starting from the loss and working backward through the neural network.
    


    # Step 5: Backward pass + update:
    # First, clear the gradients from the previous training step.
    # TODO 3: call `.backward()` on `loss`
    loss.backward()

    # Step 6: Update the model's parameters.
    # The optimizer uses the gradients computed in the backward pass to
    # adjust each trainable parameter. This is the step where the model
    # learns by taking a small step toward reducing the loss.
    # TODO 4: call `.step()` on `optimizer`
    optimizer.step()

    # Step 7: Save the loss so we can look at how training progressed later.
    # `.item()` converts the single-value tensor into a Python number.
    losses.append(loss.item())

    # Print the loss every 20 epochs so we can monitor training progress
    # without printing something after every single iteration.
    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch + 1:3d}/{num_epochs}  loss = {loss.item():.3f}")


# ---------------------------------------------------------------------------
# 5. Check what the model learned
# ---------------------------------------------------------------------------
learned_weight = model.linear.weight.item()
learned_bias = model.linear.bias.item()
print(f"\nTrue line:    y = 2.00x + 1.00")
print(f"Learned line: y = {learned_weight:.2f}x + {learned_bias:.2f}")

# ---------------------------------------------------------------------------
# 6. Plot it
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(losses)
ax1.set_title("Loss over training")
ax1.set_xlabel("epoch")
ax1.set_ylabel("MSE loss")

ax2.scatter(x.numpy(), y.numpy(), s=10, label="noisy data")
with torch.no_grad():
    ax2.plot(x.numpy(), model(x).numpy(), color="red", label="learned line")
ax2.set_title("Learned fit")
ax2.legend()

plt.tight_layout()
plt.savefig("pytorch_basics_result.png")
print("\nSaved plot to pytorch_basics_result.png")
plt.show()
