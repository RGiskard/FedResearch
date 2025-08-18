# -*- coding: utf-8 -*-
"""
Paper: Quantum-Federated Privacy (Q-FedPriv) - Experimento con LeNet-5 y FedProx
Conferencia: UCC 2025
Descripción: Compara FedAvg, FedProx y Q-FedPriv usando PyTorch y Qiskit con LeNet-5.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import copy

# ==============================================================================
# 1. HIPERPARÁMETROS
# ==============================================================================
print("1. Configurando los hiperparámetros...")
NUM_CLIENTS = 10
NUM_ROUNDS = 20 # Aumentado para mejor convergencia
EPOCHS_PER_CLIENT = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Se usara:{DEVICE}")
# Parámetro para FedProx (mu > 0)
FEDPROX_MU = 0.01
# Parámetro para Q-FedPriv
NUM_SHOTS = 1024

# ==============================================================================
# 2. DATOS Y MODELO (LE-NET 5)
# ==============================================================================
print("2. Cargando datos y definiendo el modelo LeNet-5...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Valores estándar de normalización para MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Distribuyendo datos de forma No-IID entre {NUM_CLIENTS} clientes...")
client_data_indices = [[] for _ in range(NUM_CLIENTS)]
for i, (_, label) in enumerate(train_dataset):
    client_data_indices[label % NUM_CLIENTS].append(i)

client_loaders = []
for indices in client_data_indices:
    subset = torch.utils.data.Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    client_loaders.append(loader)

# Modelo LeNet-5 para MNIST
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), # Padding para mantener dimensiones
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_stack = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = torch.flatten(x, 1)
        logits = self.fc_stack(x)
        return logits

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()
    return correct / len(loader.dataset)

# ==============================================================================
# 3. PROTOCOLO CUÁNTICO Q-FedPriv (Sin cambios)
# ==============================================================================
def encode_gradient_in_circuit(gradient_vector):
    clipped_grad = np.clip(gradient_vector, -1.0, 1.0)
    circuit = QuantumCircuit(len(clipped_grad))
    for i, grad_val in enumerate(clipped_grad):
        angle = np.arcsin(grad_val)
        circuit.ry(2 * angle, i)
    return circuit

def aggregate_gradients_quantum(circuits, shots=1024):
    simulator = AerSimulator()
    estimated_gradients = []
    for circuit in circuits:
        circuit.measure_all()
        result = simulator.run(circuit, shots=shots).result()
        counts = result.get_counts(circuit)
        estimated_grad = np.zeros(circuit.num_qubits)
        for outcome, count in counts.items():
            prob = count / shots
            for i, bit in enumerate(reversed(outcome)):
                sign = 1 if bit == '0' else -1
                estimated_grad[i] += sign * prob
        estimated_gradients.append(estimated_grad)
    return np.mean(estimated_gradients, axis=0)

# ==============================================================================
# 4. IMPLEMENTACIÓN DE LOS MÉTODOS FEDERADOS
# ==============================================================================

# --- Entrenamiento de Cliente para FedAvg y Q-FedPriv ---
def train_client_standard(model, loader, epochs):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(data), target)
            loss.backward()
            optimizer.step()

# --- Entrenamiento de Cliente para FedProx ---
def train_client_fedprox(model, global_model, loader, epochs, mu):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    global_params = list(global_model.parameters())

    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            
            # Pérdida estándar
            loss = loss_fn(model(data), target)
            
            # Término proximal de FedProx
            proximal_term = 0.0
            for local_param, global_param in zip(model.parameters(), global_params):
                proximal_term += torch.pow(torch.norm(local_param - global_param), 2)
            loss += (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()

# --- Bucle de ejecución para FedAvg ---
def run_fedavg():
    print("\n--- Ejecutando FedAvg (LeNet-5) ---")
    global_model = LeNet5().to(DEVICE)
    accuracies = []
    for round_num in range(NUM_ROUNDS):
        local_weights = []
        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            train_client_standard(local_model, client_loaders[i], EPOCHS_PER_CLIENT)
            local_weights.append(copy.deepcopy(local_model.state_dict()))
        
        global_state_dict = global_model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.stack([local_weights[i][key] for i in range(NUM_CLIENTS)]).mean(0)
        global_model.load_state_dict(global_state_dict)
        
        acc = evaluate_model(global_model, test_loader)
        accuracies.append(acc)
        print(f"  Ronda {round_num+1}, Precisión: {acc:.4f}")
    return accuracies

# --- Bucle de ejecución para FedProx ---
def run_fedprox():
    print("\n--- Ejecutando FedProx (LeNet-5) ---")
    global_model = LeNet5().to(DEVICE)
    accuracies = []
    for round_num in range(NUM_ROUNDS):
        local_weights = []
        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            train_client_fedprox(local_model, global_model, client_loaders[i], EPOCHS_PER_CLIENT, FEDPROX_MU)
            local_weights.append(copy.deepcopy(local_model.state_dict()))

        global_state_dict = global_model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.stack([local_weights[i][key] for i in range(NUM_CLIENTS)]).mean(0)
        global_model.load_state_dict(global_state_dict)

        acc = evaluate_model(global_model, test_loader)
        accuracies.append(acc)
        print(f"  Ronda {round_num+1}, Precisión: {acc:.4f}")
    return accuracies

# --- Bucle de ejecución para Q-FedPriv ---
def run_qfedpriv():
    print("\n--- Ejecutando Q-FedPriv (LeNet-5) ---")
    global_model = LeNet5().to(DEVICE)
    accuracies = []
    
    total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    # Limitar el número de parámetros a codificar para que sea manejable
    params_to_encode = min(total_params, 2000)
    print(f"  El modelo tiene {total_params} parámetros. Codificando los primeros {params_to_encode}.")

    for round_num in range(NUM_ROUNDS):
        global_weights_dict = copy.deepcopy(global_model.state_dict())
        client_gradient_circuits = []
        
        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            train_client_standard(local_model, client_loaders[i], EPOCHS_PER_CLIENT)
            
            local_update = []
            with torch.no_grad():
                for key in global_weights_dict.keys():
                    update = local_model.state_dict()[key] - global_weights_dict[key]
                    local_update.append(update.view(-1))
            
            flat_update_np = torch.cat(local_update).cpu().numpy()[:params_to_encode]
            client_gradient_circuits.append(encode_gradient_in_circuit(flat_update_np))
        
        aggregated_flat_gradient = aggregate_gradients_quantum(client_gradient_circuits, shots=NUM_SHOTS)
        
        with torch.no_grad():
            pointer = 0
            new_global_weights = global_model.state_dict()
            grad_tensor = torch.from_numpy(aggregated_flat_gradient).float().to(DEVICE)
            
            for key in new_global_weights.keys():
                param = new_global_weights[key]
                if not param.requires_grad: continue
                num_elements = param.numel()
                if pointer < len(grad_tensor):
                    update_slice = grad_tensor[pointer : pointer + num_elements]
                    if update_slice.numel() == num_elements:
                        param -= update_slice.reshape(param.shape) * LEARNING_RATE
                pointer += num_elements
            global_model.load_state_dict(new_global_weights)

        acc = evaluate_model(global_model, test_loader)
        accuracies.append(acc)
        print(f"  Ronda {round_num+1}, Precisión: {acc:.4f}")
    return accuracies

# ==============================================================================
# 5. EJECUCIÓN Y VISUALIZACIÓN COMPARATIVA
# ==============================================================================
if __name__ == '__main__':
    acc_standard = run_fedavg()
    acc_fedprox = run_fedprox()
    acc_qfedpriv = run_qfedpriv()
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, NUM_ROUNDS + 1), acc_standard, marker='o', linestyle='-', label='FedAvg (Baseline)')
    plt.plot(range(1, NUM_ROUNDS + 1), acc_fedprox, marker='s', linestyle='--', label='FedProx (Baseline)')
    plt.plot(range(1, NUM_ROUNDS + 1), acc_qfedpriv, marker='x', linestyle=':', label='Q-FedPriv (Nuestra Propuesta)')
    
    plt.title('Comparativa de Precisión con LeNet-5 en MNIST (No-IID)')
    plt.xlabel('Ronda de Comunicación')
    plt.ylabel('Precisión en el Test Set')
    plt.xticks(range(0, NUM_ROUNDS + 1, 2))
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()