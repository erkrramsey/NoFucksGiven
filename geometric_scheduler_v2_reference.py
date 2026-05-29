import itertools, json, math, hashlib, statistics, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path('/mnt/data') if Path('/mnt/data').exists() else Path('.')
FIGDIR = ROOT / 'geo_v2_figures'
FIGDIR.mkdir(exist_ok=True)
MASTER_SEED = 20260528
RNG = np.random.default_rng(MASTER_SEED)
TWO53 = 2**53
print('seed', MASTER_SEED, 'figure_dir', FIGDIR)



def grid_coords(rows, cols):
    return np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=np.int64)

def manhattan_metric(coords, scale=1):
    coords = np.asarray(coords, dtype=np.int64)
    diff = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
    return (scale * diff).astype(np.int64)

def rack_fabric_metric(rows=4, racks_per_row=4, slots_per_rack=8,
                       intra_slot=1, rack_penalty=7, row_penalty=23, pod_penalty=41):
    # Hierarchical physical metric: slot distance plus rack/row penalties.
    # Returns (D, coords, metadata).
    coords = []
    meta = []
    for row in range(rows):
        for rack in range(racks_per_row):
            for slot in range(slots_per_rack):
                coords.append((row, rack, slot))
                meta.append({'row': row, 'rack': rack, 'slot': slot})
    coords = np.array(coords, dtype=np.int64)
    n = len(coords)
    D = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            ri, rki, si = coords[i]
            rj, rkj, sj = coords[j]
            if ri == rj and rki == rkj:
                D[i, j] = intra_slot * abs(si - sj)
            elif ri == rj:
                D[i, j] = rack_penalty + abs(rki-rkj)*2 + (si+sj)//4
            else:
                # cross-row / cross-pod proxy: row jump plus rack and slot ingress/egress cost
                D[i, j] = row_penalty + abs(ri-rj)*pod_penalty + abs(rki-rkj)*3 + (si+sj)//3
    return D.astype(np.int64), coords, meta

def symmetric_zero_diag(A):
    A = np.asarray(A)
    A = ((A + A.T) // 2).astype(np.int64)
    np.fill_diagonal(A, 0)
    return A

def lattice_locality_weights(rows, cols, near=1000, far=8, decay=2.1):
    coords = grid_coords(rows, cols)
    L = manhattan_metric(coords)
    W = np.rint(far + near / np.maximum(1, L)**decay).astype(np.int64)
    np.fill_diagonal(W, 0)
    return symmetric_zero_diag(W), coords

def clustered_weights(n, clusters=4, within=900, between=25, jitter=20, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(clusters), int(np.ceil(n/clusters)))[:n]
    rng.shuffle(labels)
    W = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i+1, n):
            base = within if labels[i] == labels[j] else between
            val = base + int(rng.integers(0, jitter+1))
            W[i, j] = W[j, i] = val
    return W, labels

def ring_expander_like_weights(n, degree=4, weight=100, background=20):
    W = np.full((n, n), background, dtype=np.int64)
    np.fill_diagonal(W, 0)
    for i in range(n):
        for d in range(1, degree//2 + 1):
            j = (i + d) % n
            k = (i - d) % n
            W[i, j] = W[j, i] = weight
            W[i, k] = W[k, i] = weight
    return symmetric_zero_diag(W)

def objective_int(W, D, perm):
    W = np.asarray(W, dtype=np.int64); D = np.asarray(D, dtype=np.int64)
    perm = np.asarray(perm, dtype=np.int64)
    n = len(perm)
    total = 0
    for i in range(n):
        pi = int(perm[i])
        for j in range(i+1, n):
            total += int(W[i, j]) * int(D[pi, int(perm[j])])
    return int(total)

def objective_float64(W, D, perm):
    Wf = np.asarray(W, dtype=np.float64)
    Df = np.asarray(D, dtype=np.float64)
    perm = np.asarray(perm, dtype=np.int64)
    M = Df[np.ix_(perm, perm)]
    return float(np.triu(Wf * M, 1).sum(dtype=np.float64))

def swap_delta_int(W, D, perm, a, b):
    if a == b:
        return 0
    W = np.asarray(W, dtype=np.int64); D = np.asarray(D, dtype=np.int64)
    perm = np.asarray(perm, dtype=np.int64)
    pa, pb = int(perm[a]), int(perm[b])
    delta = 0
    n = len(perm)
    for k in range(n):
        if k == a or k == b:
            continue
        pk = int(perm[k])
        delta += (int(W[a, k]) - int(W[b, k])) * (int(D[pb, pk]) - int(D[pa, pk]))
    return int(delta)

def apply_swap(perm, a, b):
    q = np.array(perm, dtype=np.int64).copy()
    q[a], q[b] = q[b], q[a]
    return q

def best_improving_swap(W, D, perm):
    n = len(perm)
    best = (0, None, None)
    for a in range(n-1):
        for b in range(a+1, n):
            d = swap_delta_int(W, D, perm, a, b)
            if d < best[0]:
                best = (d, a, b)
    return best

def local_swap_descent(W, D, perm, max_iter=10000, migration_cost=0):
    perm = np.array(perm, dtype=np.int64).copy()
    history = [objective_int(W, D, perm)]
    moves = []
    for step in range(max_iter):
        d, a, b = best_improving_swap(W, D, perm)
        adjusted = d + migration_cost
        if a is None or adjusted >= 0:
            break
        old = history[-1]
        perm = apply_swap(perm, a, b)
        new = objective_int(W, D, perm)
        assert new - old == d, (new, old, d, a, b)
        assert new <= old
        history.append(new)
        moves.append({'step': step, 'a': int(a), 'b': int(b), 'delta': int(d), 'adjusted_delta': int(adjusted), 'J': int(new)})
    return perm, history, moves

def placement_by_labels_into_rack_blocks(labels, slots_per_rack=8):
    # Greedy deterministic cluster packing: nodes in same logical label occupy contiguous physical slots.
    labels = np.asarray(labels)
    order = np.argsort(labels, kind='stable')
    perm = np.empty_like(order)
    for physical_slot, logical_node in enumerate(order):
        perm[logical_node] = physical_slot
    return perm.astype(np.int64)

def cost_summary(W, D, perm):
    return {'J': int(objective_int(W, D, perm)), 'J_float': objective_float64(W, D, perm)}



def xy_to_hilbert_index(x, y, bits):
    # Standard integer Hilbert d2xy inverse style mapping from (x,y) to d.
    # Valid for 0 <= x,y < 2**bits.
    x = int(x); y = int(y)
    d = 0
    s = 1 << (bits - 1)
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = (1 << bits) - 1 - x
                y = (1 << bits) - 1 - y
            x, y = y, x
        s //= 2
    return int(d)

def hilbert_order(side):
    bits = int(math.log2(side))
    assert 2**bits == side, 'side must be a power of two'
    indexed = []
    for y in range(side):
        for x in range(side):
            indexed.append((xy_to_hilbert_index(x, y, bits), y*side+x))
    return [idx for _, idx in sorted(indexed)]

def morton_index(x, y):
    x = int(x); y = int(y)
    z = 0; bit = 0
    while x or y:
        z |= (x & 1) << (2*bit)
        z |= (y & 1) << (2*bit + 1)
        x >>= 1; y >>= 1; bit += 1
    return z

def morton_order(side):
    indexed = []
    for y in range(side):
        for x in range(side):
            indexed.append((morton_index(x, y), y*side+x))
    return [idx for _, idx in sorted(indexed)]

def row_major_order(side):
    return list(range(side*side))

def snake_order(side):
    order = []
    for r in range(side):
        row = list(range(r*side, (r+1)*side))
        if r % 2:
            row.reverse()
        order += row
    return order

def order_to_perm(order):
    # order lists logical nodes along increasing physical slot; perm[logical] = physical slot.
    perm = np.empty(len(order), dtype=np.int64)
    for slot, logical in enumerate(order):
        perm[logical] = slot
    return perm

def one_dim_physical_metric(n, near_cost=1):
    coords = np.array([(i,) for i in range(n)], dtype=np.int64)
    return manhattan_metric(coords, scale=near_cost)



def test_swap_delta_identity(trials=300, n=16, seed=11):
    rng = np.random.default_rng(seed)
    W, _ = clustered_weights(n, clusters=4, within=700, between=17, jitter=13, seed=seed)
    D, _, _ = rack_fabric_metric(rows=2, racks_per_row=2, slots_per_rack=n//4)
    failures = []
    for t in range(trials):
        perm = rng.permutation(n).astype(np.int64)
        J0 = objective_int(W, D, perm)
        for _ in range(10):
            a, b = sorted(rng.choice(n, 2, replace=False).tolist())
            d = swap_delta_int(W, D, perm, a, b)
            q = apply_swap(perm, a, b)
            diff = objective_int(W, D, q) - J0
            if d != diff:
                failures.append((t, a, b, d, diff))
    return {'trials': trials, 'n': n, 'failures': failures, 'pass': len(failures) == 0}

t1 = test_swap_delta_identity()
print(json.dumps({k:v for k,v in t1.items() if k!='failures'}, indent=2))
assert t1['pass']



def test_float64_exactness(trials=500, n=16, seed=22):
    rng = np.random.default_rng(seed)
    W, _ = clustered_weights(n, clusters=4, within=90, between=2, jitter=3, seed=seed)
    D, _, _ = rack_fabric_metric(rows=2, racks_per_row=2, slots_per_rack=n//4,
                                 intra_slot=1, rack_penalty=5, row_penalty=11, pod_penalty=13)
    max_abs_err = 0.0
    max_J = 0
    for _ in range(trials):
        perm = rng.permutation(n).astype(np.int64)
        Ji = objective_int(W, D, perm)
        Jf = objective_float64(W, D, perm)
        max_J = max(max_J, Ji)
        err = abs(Jf - float(Ji))
        max_abs_err = max(max_abs_err, err)
        assert Ji < TWO53
        assert err == 0.0
    return {'pass': True, 'trials': trials, 'n': n, 'max_int_objective': int(max_J), 'max_abs_float_error': max_abs_err, 'two_to_53': TWO53}

t2 = test_float64_exactness()
print(json.dumps(t2, indent=2))



def exhaustive_optimum(W, D):
    n = W.shape[0]
    best_J = None; best_perm = None
    for perm in itertools.permutations(range(n)):
        J = objective_int(W, D, np.array(perm, dtype=np.int64))
        if best_J is None or J < best_J:
            best_J = J; best_perm = perm
    return int(best_J), np.array(best_perm, dtype=np.int64)

def test_exhaustive_small(seed=33):
    rng = np.random.default_rng(seed)
    n = 8
    W, labels = clustered_weights(n, clusters=2, within=120, between=5, jitter=2, seed=seed)
    coords = grid_coords(2,4)
    D = manhattan_metric(coords, scale=3)
    opt_J, opt_perm = exhaustive_optimum(W, D)
    starts = []
    for s in range(20):
        p0 = rng.permutation(n)
        pf, hist, moves = local_swap_descent(W, D, p0)
        starts.append({'start': int(hist[0]), 'final': int(hist[-1]), 'moves': len(moves), 'gap_to_global': int(hist[-1] - opt_J)})
        assert all(hist[i+1] <= hist[i] for i in range(len(hist)-1))
    best_final = min(x['final'] for x in starts)
    return {'pass': True, 'global_optimum': opt_J, 'best_local_final': int(best_final), 'best_gap': int(best_final - opt_J), 'runs': starts[:5], 'all_gaps': [x['gap_to_global'] for x in starts]}

t3 = test_exhaustive_small()
print(json.dumps({k:v for k,v in t3.items() if k not in ['all_gaps']}, indent=2))
assert t3['best_gap'] == 0



def compare_ribbon_orders(side=8, random_trials=200, seed=44):
    rng = np.random.default_rng(seed)
    n = side*side
    W, logical_coords = lattice_locality_weights(side, side, near=500, far=1, decay=2.2)
    D1 = one_dim_physical_metric(n)
    methods = {
        'hilbert': order_to_perm(hilbert_order(side)),
        'morton': order_to_perm(morton_order(side)),
        'snake': order_to_perm(snake_order(side)),
        'row_major': order_to_perm(row_major_order(side)),
    }
    scores = {name: objective_int(W, D1, perm) for name, perm in methods.items()}
    random_scores = []
    for _ in range(random_trials):
        random_scores.append(objective_int(W, D1, rng.permutation(n)))
    scores['random_mean'] = float(np.mean(random_scores))
    scores['random_min'] = int(np.min(random_scores))
    scores['random_max'] = int(np.max(random_scores))
    scores['random_std'] = float(np.std(random_scores))
    scores['hilbert_vs_random_mean_reduction'] = float((scores['random_mean'] - scores['hilbert']) / scores['random_mean'])
    scores['best_named'] = min(['hilbert','morton','snake','row_major'], key=lambda k: scores[k])
    return scores, random_scores

scores4, random_scores4 = compare_ribbon_orders()
print(json.dumps(scores4, indent=2))
assert scores4['hilbert'] < scores4['random_mean']



# Figure: ribbon placement comparison
names = ['hilbert','morton','snake','row_major','random_mean']
vals = [scores4[n] for n in names]
plt.figure(figsize=(7,4))
plt.bar(names, vals)
plt.xticks(rotation=30, ha='right')
plt.ylabel('weighted distance objective J')
plt.title('2D locality mapped onto 1D physical ribbon')
plt.tight_layout()
fig1 = FIGDIR / 'ribbon_order_comparison.png'
plt.savefig(fig1, dpi=180)
plt.close()
fig1



def rack_cluster_experiment(n=32, clusters=4, seed=55):
    rng = np.random.default_rng(seed + 1009)
    W, labels = clustered_weights(n, clusters=clusters, within=1000, between=20, jitter=30, seed=seed)
    D, coords, meta = rack_fabric_metric(rows=1, racks_per_row=4, slots_per_rack=8, rack_penalty=30, row_penalty=80, pod_penalty=120)
    random_perm = rng.permutation(n)
    packed_perm = placement_by_labels_into_rack_blocks(labels)
    opt_perm, hist, moves = local_swap_descent(W, D, random_perm, max_iter=200)
    packed_refined, packed_hist, packed_moves = local_swap_descent(W, D, packed_perm, max_iter=200)
    result = {
        'random_start': int(hist[0]),
        'random_local_final': int(hist[-1]),
        'random_local_moves': len(moves),
        'cluster_packed': int(objective_int(W, D, packed_perm)),
        'cluster_packed_refined': int(packed_hist[-1]),
        'cluster_packed_refinement_moves': len(packed_moves),
        'packed_vs_random_reduction': float((hist[0] - objective_int(W,D,packed_perm))/hist[0]),
        'local_vs_random_reduction': float((hist[0] - hist[-1])/hist[0]),
    }
    return result, hist, packed_hist, labels, W, D, random_perm, packed_perm

res5, hist5, phist5, labels5, W5, D5, random_perm5, packed_perm5 = rack_cluster_experiment()
print(json.dumps(res5, indent=2))
assert res5['cluster_packed'] < res5['random_start']
assert all(hist5[i+1] <= hist5[i] for i in range(len(hist5)-1))



plt.figure(figsize=(7,4))
plt.plot(hist5, marker='o', markersize=2, linewidth=1)
plt.xlabel('accepted swap')
plt.ylabel('objective J')
plt.title('Monotone local swap descent from random placement')
plt.tight_layout()
fig2 = FIGDIR / 'swap_descent_trajectory.png'
plt.savefig(fig2, dpi=180)
plt.close()
fig2



plt.figure(figsize=(6,4))
labels = ['random start', 'swap final', 'packed', 'packed refined']
vals = [res5['random_start'], res5['random_local_final'], res5['cluster_packed'], res5['cluster_packed_refined']]
plt.bar(labels, vals)
plt.xticks(rotation=25, ha='right')
plt.ylabel('weighted physical cost')
plt.title('Rack-aware geometric placement reduces hot-edge distance')
plt.tight_layout()
fig3 = FIGDIR / 'rack_cluster_comparison.png'
plt.savefig(fig3, dpi=180)
plt.close()
fig3



def dynamic_drift_experiment(n=32, seed=66, migration_cost=1000):
    rng = np.random.default_rng(seed + 1009)
    D, _, _ = rack_fabric_metric(rows=1, racks_per_row=4, slots_per_rack=8, rack_penalty=30, row_penalty=80, pod_penalty=120)
    W1, labels1 = clustered_weights(n, clusters=4, within=850, between=18, jitter=10, seed=seed)
    W2, labels2 = clustered_weights(n, clusters=4, within=850, between=18, jitter=10, seed=seed+1)
    p0 = placement_by_labels_into_rack_blocks(labels1)
    p1, hist1, moves1 = local_swap_descent(W1, D, p0, migration_cost=0)
    before_drift = objective_int(W2, D, p1)
    p2, hist2, moves2 = local_swap_descent(W2, D, p1, migration_cost=migration_cost)
    after_drift = hist2[-1]
    # Verify every accepted move cleared the migration threshold.
    assert all(m['delta'] + migration_cost < 0 for m in moves2)
    return {
        'J_on_W1_after_initial_placement': int(hist1[-1]),
        'J_on_W2_before_remap': int(before_drift),
        'J_on_W2_after_thresholded_remap': int(after_drift),
        'accepted_remap_moves': len(moves2),
        'migration_cost_per_swap': migration_cost,
        'drift_reduction': float((before_drift - after_drift)/before_drift) if before_drift else 0.0,
    }, hist2

res6, hist6 = dynamic_drift_experiment()
print(json.dumps(res6, indent=2))
assert res6['J_on_W2_after_thresholded_remap'] <= res6['J_on_W2_before_remap']



plt.figure(figsize=(7,4))
plt.plot(hist6, marker='o', markersize=2, linewidth=1)
plt.xlabel('accepted thresholded remap')
plt.ylabel('objective under drifted W')
plt.title('Dynamic remap with migration-cost threshold')
plt.tight_layout()
fig4 = FIGDIR / 'dynamic_drift_remap.png'
plt.savefig(fig4, dpi=180)
plt.close()
fig4



def null_case_tests(n=32, seed=77):
    rng = np.random.default_rng(seed)
    D, _, _ = rack_fabric_metric(rows=2, racks_per_row=4, slots_per_rack=4)
    W_uniform = np.ones((n,n), dtype=np.int64) * 10
    np.fill_diagonal(W_uniform, 0)
    scores = [objective_int(W_uniform, D, rng.permutation(n)) for _ in range(50)]
    uniform_invariant = len(set(scores)) == 1
    W_exp = ring_expander_like_weights(n, degree=8, weight=100, background=80)
    exp_scores = [objective_int(W_exp, D, rng.permutation(n)) for _ in range(120)]
    p0 = rng.permutation(n)
    pf, hist, moves = local_swap_descent(W_exp, D, p0, max_iter=200)
    return {
        'uniform_invariant': uniform_invariant,
        'uniform_unique_scores': sorted(set(scores))[:3],
        'expander_random_mean': float(np.mean(exp_scores)),
        'expander_random_std': float(np.std(exp_scores)),
        'expander_local_reduction': float((hist[0]-hist[-1])/hist[0]) if hist[0] else 0.0,
        'expander_moves': len(moves)
    }

t7 = null_case_tests()
print(json.dumps(t7, indent=2))
assert t7['uniform_invariant']



evidence = {
    'version': 'geo-infra-v2-hardpush',
    'seed': MASTER_SEED,
    'tests': {
        'swap_delta_identity': {k:v for k,v in t1.items() if k!='failures'},
        'float64_exactness': t2,
        'exhaustive_small': {k:v for k,v in t3.items() if k not in ['runs','all_gaps']},
        'ribbon_orders': scores4,
        'rack_cluster': res5,
        'dynamic_drift': res6,
        'null_cases': t7,
    },
    'figures': [str(p) for p in sorted(FIGDIR.glob('*.png'))],
}
canonical = json.dumps(evidence, sort_keys=True, separators=(',', ':'))
root = hashlib.sha256(canonical.encode()).hexdigest()
evidence['evidence_root'] = root
evidence_path = ROOT / 'geometric_infra_v2_evidence.json'
evidence_path.write_text(json.dumps(evidence, indent=2))
print('evidence_root', root)
print('wrote', evidence_path)
assert all([
    evidence['tests']['swap_delta_identity']['pass'],
    evidence['tests']['float64_exactness']['pass'],
    evidence['tests']['exhaustive_small']['pass'],
    evidence['tests']['null_cases']['uniform_invariant'],
])

