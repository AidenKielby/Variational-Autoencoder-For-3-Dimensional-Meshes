"""
Test: Multi-Output Backpropagation
===================================
Task: Classify a 2D point into one of 3 zones based on its (x, y) coordinates.
 
Zone rules (easy to learn, but requires multi-output softmax):
  - Zone 0 (Red):   x < 0.33
  - Zone 1 (Green): 0.33 <= x < 0.66
  - Zone 2 (Blue):  x >= 0.66
 
Inputs:  [x, y]  (both in range 0-1, y is a mild distractor)
Outputs: [p_zone0, p_zone1, p_zone2]  (softmax probabilities)
 
This tests that:
  1. Softmax output layer is chosen (outputNeurons > 1)
  2. Cross-entropy-style gradients (o - t) flow back correctly
  3. All 3 output neurons and their weights get updated
"""
 
import random
import math
from Network import NeuralNetwork  
 
 
# ──────────────────────────────────────────────
# 1. Data generation
# ──────────────────────────────────────────────
 
def generate_sample():
    x = random.random()
    y = random.random()
    if x < 0.33:
        label = [1, 0, 0]
    elif x < 0.66:
        label = [0, 1, 0]
    else:
        label = [0, 0, 1]
    return [x, y], label
 
 
def generate_dataset(n):
    return [generate_sample() for _ in range(n)]
 
 
# ──────────────────────────────────────────────
# 2. Helpers
# ──────────────────────────────────────────────
 
def argmax(lst):
    return lst.index(max(lst))
 
 
def accuracy(net, dataset):
    correct = 0
    for inputs, label in dataset:
        net.addInputs(inputs)
        net.forwardPass()
        if argmax(net.neurons[-1]) == argmax(label):
            correct += 1
    return correct / len(dataset)
 
 
def avg_loss(net, dataset):
    total = 0
    for inputs, label in dataset:
        net.addInputs(inputs)
        net.forwardPass()
        output = net.neurons[-1]
        total += -sum(t * math.log(o + 1e-9) for o, t in zip(output, label))
    return total / len(dataset)
 
 
# ──────────────────────────────────────────────
# 3. Training
# ──────────────────────────────────────────────
 
def train(net, dataset, epochs, lr, batch_log=200):
    print(f"\n{'='*55}")
    print(f"  Network: 2 inputs → 2 hidden layers × 8 neurons → 3 outputs")
    print(f"  Dataset: {len(dataset)} samples  |  LR: {lr}  |  Epochs: {epochs}")
    print(f"{'='*55}\n")
 
    loss_history = []
 
    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)
        for inputs, label in dataset:
            net.addInputs(inputs)
            net.forwardPass()
            net.backpropagate(label, lr, 0.85)
 
        if epoch % batch_log == 0 or epoch == 1:
            loss = avg_loss(net, dataset)
            acc  = accuracy(net, dataset)
            loss_history.append((epoch, loss, acc))
            print(f"  Epoch {epoch:>5}  |  Loss: {loss:.4f}  |  Accuracy: {acc*100:.1f}%")
 
    return loss_history
 
 
# ──────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────
 
def evaluate(net, n=500):
    test_data = generate_dataset(n)
    acc = accuracy(net, test_data)
    print(f"\n{'='*55}")
    print(f"  Test accuracy on {n} unseen samples: {acc*100:.1f}%")
    print(f"{'='*55}\n")
 
    print("  Sample predictions (x, y  →  predicted / actual):")
    for inputs, label in test_data[:8]:
        net.addInputs(inputs)
        net.forwardPass()
        probs = net.neurons[-1]
        pred  = argmax(probs)
        true  = argmax(label)
        mark  = "✓" if pred == true else "✗"
        print(f"    x={inputs[0]:.2f} y={inputs[1]:.2f}  "
              f"→  pred={pred} ({probs[pred]:.2f})  true={true}  {mark}")
    return acc
 
 
# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────
 
if __name__ == "__main__":
    random.seed(42)
 
    # Network: 2 inputs, 2 hidden layers, 8 neurons each, 3 outputs
    net = NeuralNetwork(
        inputNeurons=2,
        hiddenLayers=2,
        neuronsPerHiddenLayer=8,
        outputNeurons=3
    )
 
    train_data = generate_dataset(1000)
 
    history = train(net, train_data, epochs=2000, lr=0.005, batch_log=200)
 
    final_acc = evaluate(net, n=500)
 
    # Quick pass/fail verdict
    print("\n" + "="*55)
    if final_acc >= 0.90:
        print("  ✅  PASS — multi-output backprop is working correctly!")
        print(f"      Final test accuracy: {final_acc*100:.1f}%  (threshold: 90%)")
    else:
        print("  ❌  FAIL — accuracy below 90%, something may be off.")
        print(f"      Final test accuracy: {final_acc*100:.1f}%  (threshold: 90%)")
    print("="*55 + "\n")