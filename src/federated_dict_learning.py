import os
import glob
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

# =============================================================================
# ΡΥΘΜΙΣΕΙΣ PATH
# =============================================================================
DATASET_PATH = "./dataset"
# =============================================================================
# 1. Φόρτωση και Προεπεξεργασία (FLOAT64)
# =============================================================================

def load_and_preprocess_dataset(root_folder):
    print(f"--- [ΒΗΜΑ 1] Φόρτωση εικόνων από: {root_folder} ---", flush=True)
    patches = []
    image_files = glob.glob(os.path.join(root_folder, "**", "*.pgm"), recursive=True)
    
    if not image_files:
        print("ΣΦΑΛΜΑ: Δεν βρέθηκαν εικόνες.", flush=True)
        return None

    print(f"   -> Βρέθηκαν {len(image_files)} εικόνες. Ξεκινάει η επεξεργασία...", flush=True)

    count = 0
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img_arr = np.array(img.convert('L'), dtype=np.float64) / 255.0
            
            h_crop = min(img_arr.shape[0], 88)
            w_crop = min(img_arr.shape[1], 112)
            img_cropped = img_arr[:h_crop, :w_crop]
            
            for i in range(0, h_crop, 8):
                for j in range(0, w_crop, 8):
                    patch = img_cropped[i:i+8, j:j+8]
                    if patch.shape != (8, 8): continue
                    
                    vec = patch.flatten()
                    vec = vec - np.mean(vec)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-10:
                        vec = vec / norm
                    patches.append(vec)
            count += 1
        except:
            pass

    try:
        X = np.array(patches).T
    except ValueError as e:
        print("Σφάλμα διαστάσεων:", e)
        return None

    print(f"   -> Ολοκληρώθηκε. Δημιουργήθηκε πίνακας X: {X.shape}", flush=True)
    return X

# =============================================================================
# 2. Αλγόριθμοι (OMP Optimized & Classic MOD)
# =============================================================================

def normalize_dictionary(D):
    norms = np.linalg.norm(D, axis=0)
    norms[norms < 1e-10] = 1.0 
    return D / norms

def batch_omp(D, X, T):
    """ OMP με solve  """
    n, m = D.shape
    _, N = X.shape
    A = np.zeros((m, N))
    Dt = D.T
    
    for i in range(N):
        x = X[:, i]
        r = x.copy()
        idx = []
        for _ in range(T):
            best_idx = np.argmax(np.abs(Dt @ r))
            if best_idx not in idx: idx.append(best_idx)
            
            D_sub = D[:, idx]
            try:
                a_sub = np.linalg.solve(D_sub.T @ D_sub, D_sub.T @ x)
            except:
                a_sub, _, _, _ = np.linalg.lstsq(D_sub, x, rcond=None)
            r = x - D_sub @ a_sub
        A[idx, i] = a_sub
    return A

def mod_update(X, A, mu=1e-6):
    """ MOD με inv/pinv  """
    m = A.shape[0]
    AA_T = A @ A.T
    XA_T = X @ A.T
    try:
        term_inv = np.linalg.inv(AA_T + mu * np.eye(m))
    except:
        term_inv = np.linalg.pinv(AA_T + mu * np.eye(m))
    return normalize_dictionary(XA_T @ term_inv)

# =============================================================================
# 3. Μάθηση (Centralized & Federated)
# =============================================================================

def centralized_learning(X, m, T, L):
    """ Ζητούμενο 1α & 2 """
    print(f"\n--- Centralized Learning (m={m}, T={T}) ---", flush=True)
    np.random.seed(42)
    idx = np.random.choice(X.shape[1], m, replace=False)
    D = normalize_dictionary(X[:, idx])
    n, N = X.shape
    errors = []
    
    st = time.time()
    for l in range(L):
        A = batch_omp(D, X, T)
        D = mod_update(X, A)
        err = np.linalg.norm(X - D @ A, 'fro')**2 / (n * N)
        errors.append(err)
        print(f"   Iter {l+1:02d}/{L} | MSE: {err:.6f}", flush=True)
    
    print(f"   -> Χρόνος: {time.time()-st:.1f}s")
    return errors

def federated_learning(X, K, m, T, R, mode='iid'):
    """ 
    Ζητούμενα 1β, 3, 4, 5 & 6.
    Επιστρέφει: errors, D_global (τελικό), D_locals (τελικά)
    """
    print(f"\n--- Federated Learning (K={K}, m={m}, T={T}, mode={mode}) ---", flush=True)
    n, N = X.shape
    
    # Αρχικοποίηση
    np.random.seed(42)
    idx = np.random.choice(X.shape[1], m, replace=False)
    D_global = normalize_dictionary(X[:, idx])
    
    # Splitting
    if mode == 'iid':
        indices = np.random.permutation(N)
        X_sorted = X[:, indices]
    else:
        X_sorted = X 
        
    samples_per_user = N // K
    user_datasets = [X_sorted[:, k*samples_per_user:(k+1)*samples_per_user] for k in range(K)]
    
    fed_errors = []
    weights = [1.0/K] * K
    
    # Για αποθήκευση των τοπικών λεξικών του τελευταίου γύρου
    last_D_locals = []

    st = time.time()
    for r in range(R):
        D_locals = []
        total_err = 0
        for k in range(K):
            X_k = user_datasets[k]
            A_k = batch_omp(D_global, X_k, T)
            err_k = np.linalg.norm(X_k - D_global @ A_k, 'fro')**2 / (n * samples_per_user)
            total_err += weights[k] * err_k
            D_locals.append(mod_update(X_k, A_k))
        
        # FedAvg
        D_avg = np.zeros_like(D_global)
        for k in range(K): D_avg += weights[k] * D_locals[k]
        D_global = normalize_dictionary(D_avg)
        
        fed_errors.append(total_err)
        last_D_locals = D_locals # Ανανέωση για να έχουμε τα τελευταία
        print(f"   Round {r+1:02d}/{R} | Global MSE: {total_err:.6f}", flush=True)
        
    print(f"   -> Χρόνος: {time.time()-st:.1f}s")
    
    # ΕΠΙΣΤΡΟΦΗ ΚΑΙ ΤΩΝ ΛΕΞΙΚΩΝ ΓΙΑ ΤΟ ΖΗΤΟΥΜΕΝΟ 6
    return fed_errors, D_global, last_D_locals

# =============================================================================
# 4. Γραφήματα & Πίνακες (Plots & Tables)
# =============================================================================

def plot_req2_convergence(cent_errors):
    """ Γράφημα 1: Σύγκλιση Centralized """
    plt.close('all')
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cent_errors)+1), cent_errors, 'b-o', label='Centralized')
    plt.xlabel('Iterations (L)')
    plt.ylabel('MSE')
    plt.title('Ζητούμενο 2: Σύγκλιση Κεντρικοποιημένης Μάθησης')
    plt.grid(True)
    plt.legend()
    plt.savefig('req2_centralized_convergence.png')
    print("   [PLOT] Αποθηκεύτηκε: req2_centralized_convergence.png")

def plot_req2_fed_vs_baseline(baseline, fed_errors_k5):
    """ Γράφημα 2: Federated K=5 vs Baseline """
    plt.close('all')
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fed_errors_k5)+1), fed_errors_k5, 'r-s', label='Federated (K=5)')
    plt.axhline(y=baseline, color='b', linestyle='--', linewidth=2, label=f'Baseline (MSE={baseline:.5f})')
    plt.xlabel('Rounds (R)')
    plt.ylabel('MSE')
    plt.title('Ζητούμενο 2: Federated (K=5) vs Centralized Baseline')
    plt.grid(True)
    plt.legend()
    plt.savefig('req2_federated_vs_baseline.png')
    print("   [PLOT] Αποθηκεύτηκε: req2_federated_vs_baseline.png")

def plot_req3_impact_k(baseline, fed_results):
    """ Γράφημα 3: Επίδραση K=5, 10, 20 """
    plt.close('all')
    plt.figure(figsize=(10, 6))
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline (MSE={baseline:.5f})')
    
    colors = ['r', 'g', 'b']
    for (K, errs), c in zip(fed_results.items(), colors):
        plt.plot(range(1, len(errs)+1), errs, marker='o', markersize=4, color=c, label=f'FedAvg K={K}')
        
    plt.xlabel('Rounds (R)')
    plt.ylabel('MSE')
    plt.title('Ζητούμενο 3: Επίδραση Πλήθους Χρηστών (K)')
    plt.grid(True)
    plt.legend()
    plt.savefig('req3_impact_of_K.png')
    print("   [PLOT] Αποθηκεύτηκε: req3_impact_of_K.png")

def print_table_req3(baseline, fed_results):
    print("\n" + "="*60)
    print(f"{'Μέθοδος':<25} | {'Χρήστες (K)':<12} | {'Τελικό MSE':<10}")
    print("-" * 60)
    print(f"{'Centralized (Baseline)':<25} | {'-':<12} | {baseline:.6f}")
    print("-" * 60)
    for K, errors in fed_results.items():
        print(f"{'Federated (FedAvg)':<25} | {K:<12} | {errors[-1]:.6f}")
    print("="*60 + "\n")

def plot_req4_params(results_req4):
    """ Γράφημα 4: Σύγκριση Παραμέτρων """
    plt.close('all')
    plt.figure(figsize=(10, 6))
    colors = ['m', 'c']
    
    for i, ((m, T), errs) in enumerate(results_req4.items()):
        plt.plot(range(1, len(errs)+1), errs, marker='s', markersize=4, color=colors[i], label=f'm={m}, T={T}')
        
    plt.xlabel('Rounds (R)')
    plt.ylabel('MSE')
    plt.title('Ζητούμενο 4: Σύγκριση Παραμέτρων (K=10)')
    plt.grid(True)
    plt.legend()
    plt.savefig('req4_parameters.png')
    print("   [PLOT] Αποθηκεύτηκε: req4_parameters.png")

def plot_req5_iid_vs_noniid(results_req5):
    """ Γράφημα 5: IID vs Non-IID """
    plt.close('all')
    plt.figure(figsize=(10, 6))
    
    styles = {'iid': ('g', 'o'), 'non-iid': ('orange', 'x')}
    
    for mode, errs in results_req5.items():
        c, m = styles[mode]
        plt.plot(range(1, len(errs)+1), errs, color=c, marker=m, markersize=5, label=f'Scenario: {mode.upper()}')
        
    plt.xlabel('Rounds (R)')
    plt.ylabel('MSE')
    plt.title('Ζητούμενο 5: I.I.D. vs Non-I.I.D. Distribution (K=10)')
    plt.grid(True)
    plt.legend()
    plt.savefig('req5_iid_vs_noniid.png')
    print("   [PLOT] Αποθηκεύτηκε: req5_iid_vs_noniid.png")

def plot_req6_atoms(data_req6):
    """
    Ζητούμενο 6: Οπτικοποίηση Atoms.
    data_req6: Dictionary με κλειδιά (K, mode) και τιμές (D_global, D_locals)
    """
    plt.close('all')
    
    # 3 Atoms που θα εμφανίσουμε τυχαία
    atom_indices = [0, 10, 20]
    
    # Δημιουργία 4 ξεχωριστών παραθύρων ένα για κάθε σενάριο
    for (K, mode), (D_glob, D_locs) in data_req6.items():
        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        fig.suptitle(f'Atoms: K={K}, {mode.upper()}', fontsize=12)
        
        # Επιλογή δύο χρηστών (Πρώτος και Τελευταίος για μέγιστη διαφορά στο Non-IID)
        user_indices = [0, K-1] 
        dictionaries = [D_locs[user_indices[0]], D_locs[user_indices[1]], D_glob]
        labels = [f'User {user_indices[0]}', f'User {user_indices[1]}', 'Global']
        
        for row, (D, lbl) in enumerate(zip(dictionaries, labels)):
            for col, atom_idx in enumerate(atom_indices):
                # Reshape (64,) -> (8,8)
                atom = D[:, atom_idx].reshape(8, 8)
                
                ax = axes[row, col]
                ax.imshow(atom, cmap='gray')
                ax.axis('off')
                
                if col == 1: # Τίτλος στη μεσαία στήλη
                    ax.set_title(lbl, fontsize=10)

        plt.tight_layout()
        fname = f'req6_atoms_K{K}_{mode}.png'
        plt.savefig(fname)
        print(f"   [PLOT] Αποθηκεύτηκε: {fname}")

# =============================================================================
# MAIN EXECUTION (Unified 1-6)
# =============================================================================

if __name__ == "__main__":
    # Καταγραφή Αρχικού Χρόνου
    total_start_time = time.time()
    
    X = load_and_preprocess_dataset(DATASET_PATH)
    
    if X is not None:
        # ---------------------------------------------------------
        # Ζητούμενα 1 & 2: Κεντρικοποιημένη (m=128, T=10)
        # ---------------------------------------------------------
        print("\n=== ΕΚΤΕΛΕΣΗ ΖΗΤΟΥΜΕΝΩΝ 1 & 2 ===")
        m_base, T_base = 128, 10
        L, R = 20, 30
        
        cent_errors = centralized_learning(X, m=m_base, T=T_base, L=L)
        baseline_mse = cent_errors[-1]
        
        plot_req2_convergence(cent_errors)
        
        # ---------------------------------------------------------
        # Ζητούμενο 3: Επίδραση του K (m=128, T=10, IID)
        # ---------------------------------------------------------
        print("\n=== ΕΚΤΕΛΕΣΗ ΖΗΤΟΥΜΕΝΟΥ 3 (Impact of K) ===")
        fed_results_k = {}
        # Χρειαζόμαστε τα αποτελέσματα του K=5 για το γράφημα 2
        err_k5 = None 
        
        for K in [5, 10, 20]:
            # Παίρνουμε ΜΟΝΟ τα errors (με το _)
            errs, _, _ = federated_learning(X, K, m_base, T_base, R, mode='iid')
            fed_results_k[K] = errs
            if K == 5: err_k5 = errs
        
        plot_req2_fed_vs_baseline(baseline_mse, err_k5)
        plot_req3_impact_k(baseline_mse, fed_results_k)
        print_table_req3(baseline_mse, fed_results_k)
        
        # ---------------------------------------------------------
        # Ζητούμενο 4: Επίδραση Παραμέτρων (K=10, IID)
        # ---------------------------------------------------------
        print("\n=== ΕΚΤΕΛΕΣΗ ΖΗΤΟΥΜΕΝΟΥ 4 (Parameters) ===")
        scenarios = [(64, 6), (128, 10)]
        fed_results_params = {}
        K_fixed = 10
        
        for (m, T) in scenarios:
            errs, _, _ = federated_learning(X, K_fixed, m, T, R, mode='iid')
            fed_results_params[(m, T)] = errs
            
        plot_req4_params(fed_results_params)
        
        # ---------------------------------------------------------
        # Ζητούμενο 5: IID vs Non-IID (K=10, m=128, T=10)
        # ---------------------------------------------------------
        print("\n=== ΕΚΤΕΛΕΣΗ ΖΗΤΟΥΜΕΝΟΥ 5 (Distribution) ===")
        modes = ['iid', 'non-iid']
        fed_results_dist = {}
        
        for mode in modes:
            errs, _, _ = federated_learning(X, K_fixed, m_base, T_base, R, mode=mode)
            fed_results_dist[mode] = errs
            
        plot_req5_iid_vs_noniid(fed_results_dist)
        
        # ---------------------------------------------------------
        # Ζητούμενο 6: Οπτικοποίηση Atoms (K=5 & 20, IID & Non-IID)
        # ---------------------------------------------------------
        print("\n=== ΕΚΤΕΛΕΣΗ ΖΗΤΟΥΜΕΝΟΥ 6 (Atoms Visualization) ===")
        data_req6 = {} # Dictionary για να αποθηκεύσουμε τα Dictionaries
        
        # Τρέχουμε 4 σενάρια
        scenarios_req6 = [
            (5, 'iid'), (5, 'non-iid'),
            (20, 'iid'), (20, 'non-iid')
        ]
        
        for (K, mode) in scenarios_req6:
            # Τώρα χρειαζόμαστε τα D_global και D_locals
            _, D_glob, D_locs = federated_learning(X, K, m_base, T_base, R, mode=mode)
            data_req6[(K, mode)] = (D_glob, D_locs)
            
        plot_req6_atoms(data_req6)
        
        print("\n=== ΟΛΑ ΤΑ ΖΗΤΟΥΜΕΝΑ (1-6) ΟΛΟΚΛΗΡΩΘΗΚΑΝ ΕΠΙΤΥΧΩΣ ===")
        
    # Καταγραφή Τελικού Χρόνου & Εκτύπωση
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nΣυνολικός Χρόνος Εκτέλεσης: {total_duration:.2f} δευτερόλεπτα")
