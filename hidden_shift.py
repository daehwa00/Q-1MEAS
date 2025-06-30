import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

min_qubits = 2
max_qubits = 6
n_samples = 100
test_size = 0.3
poly_degrees = list(range(2, max_qubits + 1))

quantum_accs = []
classical_accs = {deg: [] for deg in poly_degrees}

for n_qubits in tqdm(
    range(min_qubits, max_qubits + 1), desc="n_qubits loop", leave=False
):
    # 1. 입력 데이터(n_samples개, n_qubits차원) 생성
    X = np.random.randint(0, 2, (n_samples, n_qubits))

    # 2. Hidden shift 벡터 s 생성
    rng = np.random.default_rng(seed=42 + n_qubits)  # n_qubits별로 고유 시드
    s = rng.integers(0, 2, n_qubits)

    # 3. 레이블: parity(x XOR s)
    y = np.array([1 if np.sum(x ^ s) % 2 == 0 else -1 for x in X])

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 5. Parity feature map
    def parity_feature_map_circuit(x):
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            if x[i]:
                qc.rx(np.pi, i)
        qc.h(range(n_qubits))
        return qc

    def compute_kernel_matrix(X1, X2):
        N1, N2 = len(X1), len(X2)
        K = np.zeros((N1, N2))
        for i in tqdm(range(N1), desc=f"Kernel row (n_qubits={n_qubits})", leave=False):
            for j in range(N2):
                qc1 = parity_feature_map_circuit(X1[i])
                qc2 = parity_feature_map_circuit(X2[j])
                sv1 = Statevector.from_instruction(qc1)
                sv2 = Statevector.from_instruction(qc2)
                K[i, j] = np.abs(np.dot(np.conjugate(sv1.data), sv2.data)) ** 2
        return K

    # 6. Quantum SVM
    K_train = compute_kernel_matrix(X_train, X_train)
    K_test = compute_kernel_matrix(X_test, X_train)
    qsvm = SVC(kernel="precomputed", C=1e6)
    qsvm.fit(K_train, y_train)
    y_pred = qsvm.predict(K_test)
    q_acc = accuracy_score(y_test, y_pred)
    quantum_accs.append(q_acc)

    # 7. Classical SVM (poly kernel, degree=2~n_qubits)
    for deg in poly_degrees:
        if deg > n_qubits:
            classical_accs[deg].append(np.nan)
            continue
        csvm = SVC(kernel="poly", degree=deg, C=1e6)
        csvm.fit(X_train, y_train)
        y_pred_classical = csvm.predict(X_test)
        acc = accuracy_score(y_test, y_pred_classical)
        classical_accs[deg].append(acc)

# ----- Plotting -----
plt.figure(figsize=(8, 6))
x_vals = np.arange(min_qubits, max_qubits + 1)
plt.plot(
    x_vals,
    quantum_accs,
    marker="o",
    label="Quantum Kernel SVM",
    linewidth=2,
    color="black",
)

for deg, accs in classical_accs.items():
    plt.plot(x_vals, accs, marker="s", label=f"Poly degree={deg}", linestyle="--")

plt.title("Hidden Shift Classification: Quantum vs Classical Poly SVM")
plt.xlabel("n_qubits (problem dimension)")
plt.ylabel("Test Accuracy")
plt.xticks(x_vals)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figure/hidden_shift_scaling.png")

# ----- Table Output -----
print("\nHidden Shift Classification: Quantum vs Classical Poly SVM")
header = ["n_qubits", "Quantum"] + [f"Poly d={deg}" for deg in poly_degrees]
row_format = "{:>9} " + "{:>10} " * (len(header) - 1)
print(row_format.format(*header))
for idx, n_qubits in enumerate(range(min_qubits, max_qubits + 1)):
    row = [n_qubits, f"{quantum_accs[idx]:.3f}"]
    for deg in poly_degrees:
        acc = classical_accs[deg][idx]
        if np.isnan(acc):
            row.append("   -   ")
        else:
            row.append(f"{acc:.3f}")
    print(row_format.format(*row))
# plt.show()
