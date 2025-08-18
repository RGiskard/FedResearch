# -*- coding: utf-8 -*-
"""
Paper: Q-Fed(Compress/Hybrid) - Estrategia Híbrida
Conferencia: UCC 2025
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import copy

# ==============================================================================
# 1. HIPERPARÁMETROS Y CONFIGURACIÓN
# ==============================================================================
print("1. Configurando los hiperparámetros...")
CONFIG = {
    "num_clients": 10,
    "num_rounds": 75,
    "epochs_per_client": 5,
    "batch_size": 32,
    "lr": 0.005,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "fedprox_mu": 0.01,
    # --- NUEVOS PARÁMETROS DE OPTIMIZADOR ---
    "client_optimizer": "adam", # Puede ser "sgd" o "adam"
    "client_momentum": 0.9,    # Solo se usa si el optimizador es "sgd"
    # Parámetros para Q-FedCompress y Q-FedHybrid
    "q_comm_dimension": 24,#16,
    "q_shots": 1024,
    "server_lr": 0.001,
    "hybrid_quantum_fraction": 0.2
}
print(f"Se usará el dispositivo: {CONFIG['device']}")

# ==============================================================================
# 2. DATOS Y MODELO (LE-NET 5) - Sin cambios
# ==============================================================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
client_data_indices = [[] for _ in range(CONFIG['num_clients'])]
for i, (_, label) in enumerate(train_dataset): client_data_indices[label % CONFIG['num_clients']].append(i)
client_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=CONFIG['batch_size'], shuffle=True) for indices in client_data_indices]

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_stack = nn.Sequential(nn.Conv2d(1, 6, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(6, 16, 5, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_stack = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))
    def forward(self, x):
        x = self.conv_stack(x)
        return self.fc_stack(torch.flatten(x, 1))

# ==============================================================================
# 3. CLASES PARA CLIENTE Y SERVIDOR
# ==============================================================================
class Client:
    def __init__(self, client_id, loader, config):
        self.loader = loader
        self.config = config
        self.model = LeNet5().to(config['device'])

    def train(self, global_model_params, strategy='FedAvg'):
        self.model.load_state_dict(copy.deepcopy(global_model_params))
        self.model.train()
        
        # --- LÓGICA DE SELECCIÓN DE OPTIMIZADOR ---
        if self.config['client_optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config['client_momentum'])
        elif self.config['client_optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        else:
            raise ValueError("Optimizador de cliente no reconocido. Usar 'sgd' o 'adam'.")
            
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.config['epochs_per_client']):
            for data, target in self.loader:
                data, target = data.to(self.config['device']), target.to(self.config['device'])
                optimizer.zero_grad()
                loss = loss_fn(self.model(data), target)
                if strategy == 'FedProx':
                    global_model_for_prox = LeNet5().to(self.config['device'])
                    global_model_for_prox.load_state_dict(global_model_params)
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), list(global_model_for_prox.parameters())):
                        proximal_term += torch.pow(torch.norm(local_param - global_param), 2)
                    loss += (self.config['fedprox_mu'] / 2) * proximal_term
                loss.backward()
                optimizer.step()
        return self.model.state_dict()

class Server:
    def __init__(self, config):
        self.model = LeNet5().to(config['device'])
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['server_lr'])
        self.quantum_simulator = AerSimulator()
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.projection_matrix = torch.randn(
            config['q_comm_dimension'], self.total_params
        ).to(config['device'])

    def aggregate_weights(self, local_weights_list):
        global_state_dict = self.model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.stack([w[key] for w in local_weights_list]).mean(0)
        self.model.load_state_dict(global_state_dict)
    
    def apply_update_with_adam(self, update_vector):
        self.optimizer.zero_grad()
        pointer = 0
        for param in self.model.parameters():
            if not param.requires_grad: continue
            num_elements = param.numel()
            param.grad = -update_vector[pointer : pointer + num_elements].reshape(param.shape)
            pointer += num_elements
        self.optimizer.step()

    def aggregate_hybrid_updates(self, classical_updates, compressed_updates):
        final_update = torch.zeros(self.total_params, device=self.config['device'])
        num_classical = len(classical_updates)
        num_quantum = len(compressed_updates)
        total_clients = num_classical + num_quantum
        
        if num_classical > 0:
            avg_classical_update = torch.stack(classical_updates).mean(dim=0)
            final_update += avg_classical_update * (num_classical / total_clients)
        
        if num_quantum > 0:
            noisy_sketches = [self._quantum_channel_simulation(sketch) for sketch in compressed_updates]
            avg_sketch = np.mean(noisy_sketches, axis=0)
            avg_sketch_tensor = torch.from_numpy(avg_sketch).float().to(self.config['device'])
            approx_quantum_update = self.projection_matrix.T @ avg_sketch_tensor
            final_update += approx_quantum_update * (num_quantum / total_clients)
            
        self.apply_update_with_adam(final_update)

    def _quantum_channel_simulation(self, vector):
        norm = np.linalg.norm(vector)
        vector_normalized = vector / norm if norm > 0 else vector
        clipped_vector = np.clip(vector_normalized, -1.0, 1.0)
        circuit = QuantumCircuit(len(clipped_vector))
        for i, val in enumerate(clipped_vector):
            angle = np.arcsin(val)
            circuit.ry(2 * angle, i)
        circuit.measure_all()
        result = self.quantum_simulator.run(circuit, shots=self.config['q_shots']).result()
        counts = result.get_counts()
        estimated_vector = np.zeros(circuit.num_qubits)
        for outcome, count in counts.items():
            prob = count / self.config['q_shots']
            for i, bit in enumerate(reversed(outcome)):
                estimated_vector[i] += (1 if bit == '0' else -1) * prob
        return estimated_vector * norm

    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.config['device']), target.to(self.config['device'])
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return correct / total

# ==============================================================================
# 4. FUNCIÓN DE EJECUCIÓN GENÉRICA
# ==============================================================================
def run_experiment(strategy, config, clients, server):
    print(f"\n--- Ejecutando Experimento: {strategy} ---")
    accuracies = []
    server.model = LeNet5().to(config['device'])
    server.optimizer = torch.optim.Adam(server.model.parameters(), lr=config['server_lr'])

    for round_num in range(config['num_rounds']):
        global_params = copy.deepcopy(server.model.state_dict())
        
        if strategy in ['FedAvg', 'FedProx']:
            local_weights = [client.train(global_params, strategy) for client in clients]
            server.aggregate_weights(local_weights)
        
        else: # Estrategias cuánticas (Q-FedCompress y Q-FedHybrid)
            classical_updates = []
            quantum_compressed_updates = []
            
            if strategy == 'Q-FedHybrid':
                num_quantum = int(config['num_clients'] * config['hybrid_quantum_fraction'])
                quantum_indices = np.random.choice(config['num_clients'], num_quantum, replace=False)
            else: # Q-FedCompress puro
                quantum_indices = list(range(config['num_clients']))

            for i, client in enumerate(clients):
                local_weights = client.train(global_params, 'FedAvg')
                with torch.no_grad():
                    update_tensors = [local_weights[name] - global_params[name] for name, param in server.model.named_parameters() if param.requires_grad]
                    flat_update = torch.cat([t.view(-1) for t in update_tensors])
                
                if i in quantum_indices:
                    compressed_update = server.projection_matrix @ flat_update
                    quantum_compressed_updates.append(compressed_update.cpu().numpy())
                else: 
                    classical_updates.append(flat_update)

            server.aggregate_hybrid_updates(classical_updates, quantum_compressed_updates)
        
        acc = server.evaluate(test_loader)
        accuracies.append(acc)
        print(f"  Ronda {round_num+1}, Precisión: {acc:.4f}")
        
    return accuracies

# ==============================================================================
# 5. EJECUCIÓN, MÉTRICAS Y VISUALIZACIÓN
# ==============================================================================
if __name__ == '__main__':
    server_instance = Server(CONFIG)
    clients_pool = [Client(i, client_loaders[i], CONFIG) for i in range(CONFIG['num_clients'])]
    
    # Renombramos Q-FedCompress a Q-FedHybrid con fracción 1.0 para reutilizar el código
    CONFIG_COMPRESS_ONLY = CONFIG.copy()
    CONFIG_COMPRESS_ONLY['hybrid_quantum_fraction'] = 1.0
    
    acc_standard = run_experiment('FedAvg', CONFIG, clients_pool, server_instance)
    acc_fedprox = run_experiment('FedProx', CONFIG, clients_pool, server_instance)
    acc_qcompress = run_experiment('Q-FedHybrid', CONFIG_COMPRESS_ONLY, clients_pool, server_instance)
    acc_qhybrid = run_experiment('Q-FedHybrid', CONFIG, clients_pool, server_instance)
    
    # --- Cálculo de Métricas de Comunicación ---
    total_params = server_instance.total_params
    classical_overhead = total_params * 32
    quantum_overhead = CONFIG['q_comm_dimension']
    hybrid_overhead = (1 - CONFIG['hybrid_quantum_fraction']) * classical_overhead + CONFIG['hybrid_quantum_fraction'] * quantum_overhead
    
    print("\n--- Métricas de Sobrecarga de Comunicación Promedio por Cliente por Ronda ---")
    print(f"  FedAvg / FedProx: {classical_overhead / 1e6:.2f} Megabits")
    print(f"  Q-FedCompress:    {quantum_overhead} Qubits")
    print(f"  Q-FedHybrid ({int(CONFIG['hybrid_quantum_fraction']*100)}% cuántico): {hybrid_overhead / 1e6:.4f} Megabits (promedio ponderado)")

    # --- Gráfico 1: Precisión ---
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, CONFIG['num_rounds'] + 1), acc_standard, marker='o', linestyle='-', label='FedAvg')
    plt.plot(range(1, CONFIG['num_rounds'] + 1), acc_fedprox, marker='s', linestyle='--', label='FedProx')
    plt.plot(range(1, CONFIG['num_rounds'] + 1), acc_qcompress, marker='x', linestyle=':', label=f'Q-FedCompress (100% cuántico)')
    plt.plot(range(1, CONFIG['num_rounds'] + 1), acc_qhybrid, marker='D', linestyle='-.', label=f'Q-FedHybrid ({int(CONFIG["hybrid_quantum_fraction"]*100)}% cuántico)')
    
    plt.title('Comparativa de Precisión con LeNet-5 en MNIST (No-IID)')
    plt.xlabel('Ronda de Comunicación')
    plt.ylabel('Precisión en el Test Set')
    plt.xticks(range(0, CONFIG['num_rounds'] + 1, 2))
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig('accuracy_comparison_hybrid.png')
    print("\nGráfico de precisión guardado como 'accuracy_comparison_hybrid.png'")

    # --- Gráfico 2: Sobrecarga de Comunicación ---
    plt.figure(figsize=(10, 7))
    methods = ['FedAvg / FedProx', 'Q-FedHybrid', 'Q-FedCompress']
    values = [classical_overhead, hybrid_overhead, quantum_overhead]
    bars = plt.bar(methods, values, color=['#1f77b4', '#2ca02c', '#ff7f0e'])
    plt.ylabel('Sobrecarga de Comunicación (Escala Log, bits eq.)')
    plt.title('Comparación de Costo de Comunicación')
    plt.yscale('log')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:,.0f}', va='bottom', ha='center')
    plt.savefig('communication_overhead_hybrid.png')
    print("Gráfico de sobrecarga de comunicación guardado como 'communication_overhead_hybrid.png'")