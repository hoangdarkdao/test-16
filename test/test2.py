# fair_evaluation_bi_tsp.py
import json
import numpy as np
import multiprocessing
import time
import warnings
import random
from typing import List, Tuple

# ===================================================================
# 1. TẠO INSTANCE (seed cố định)
# ===================================================================
class GetData:
    def __init__(self, n_instance: int, n_cities: int):
        self.n_instance = n_instance
        self.n_cities = n_cities

    def generate_instances(self):
        np.random.seed(2025)               
        instances = []
        for _ in range(self.n_instance):
            coord1 = np.random.rand(self.n_cities, 2)
            coord2 = np.random.rand(self.n_cities, 2)
            coord = np.concatenate([coord1, coord2], axis=1)
            dist1 = np.linalg.norm(coord1[:, np.newaxis] - coord1, axis=2)
            dist2 = np.linalg.norm(coord2[:, np.newaxis] - coord2, axis=2)
            instances.append((coord, dist1, dist2))
        return instances


# ===================================================================
# 2. CÁC HÀM CƠ BẢN
# ===================================================================
def tour_cost(instance: np.ndarray, solution: np.ndarray, n: int) -> Tuple[float, float]:
    sol = solution.astype(int)
    c1 = c2 = 0.0
    for i in range(n):
        a, b = sol[i], sol[(i + 1) % n]
        c1 += np.linalg.norm(instance[a][:2] - instance[b][:2])
        c2 += np.linalg.norm(instance[a][2:] - instance[b][2:])
    return c1, c2


def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def random_solution(n: int) -> np.ndarray:
    s = np.arange(n)
    np.random.shuffle(s)
    return s


def check_constraint(sol: np.ndarray, n: int) -> bool:
    s = sol.astype(int, copy=False)
    return len(s) == n and len(set(s)) == n and s.min() >= 0 and s.max() < n


# ===================================================================
# 3. ƯỚC LƯỢNG Z_ideal & Z_nadir CỐ ĐỊNH (độc lập với mọi heuristic)
# ===================================================================
def compute_fixed_bounds(instances, n_cities: int, samples_per_inst: int = 5000):
    costs = []
    for coord, _, _ in instances:
        for _ in range(samples_per_inst):
            s = random_solution(n_cities)
            costs.append(tour_cost(coord, s, n_cities))
    arr = np.array(costs)
    return arr.min(axis=0), arr.max(axis=0)


# ===================================================================
# 4. ĐÁNH GIÁ MỘT HEURISTIC (dùng Z_ideal/nadir cố định)
# ===================================================================
def evaluate_fixed(instances,
                   n_cities: int,
                   ref_point: np.ndarray,
                   select_neighbor_func,
                   fixed_ideal: np.ndarray,
                   fixed_nadir: np.ndarray):
    hv_list = []
    time_list = []

    all_fronts = []
    for coord, dm1, dm2 in instances:
        start = time.time()
        Archive = [(random_solution(n_cities),
                    tour_cost(coord, random_solution(n_cities), n_cities))
                   for _ in range(100)]

        for _ in range(2000):
            neigh = select_neighbor_func(Archive, coord, dm1, dm2)
            if not check_constraint(neigh, n_cities):
                continue
            f = tour_cost(coord, neigh, n_cities)
            if not any(dominates(fb, f) for _, fb in Archive):
                Archive = [(s, fb) for s, fb in Archive if not dominates(f, fb)]
                Archive.append((neigh, f))

        end = time.time()
        front = np.array([f for _, f in Archive])
        all_fronts.append(front)
        time_list.append(end - start)

    # DÙNG CỐ ĐỊNH ideal/nadir → công bằng tuyệt đối
    from pymoo.indicators.hv import Hypervolume
    hv_indicator = Hypervolume(ref_point=ref_point,
                               norm_ref_point=False,
                               zero_to_one=True,
                               ideal=fixed_ideal,
                               nadir=fixed_nadir)

    for front in all_fronts:
        hv_list.append(-hv_indicator(front))   # âm để càng tốt càng gần 0

    return np.mean(hv_list), np.mean(time_list)


# ===================================================================
# 5. WORKER CHO multiprocessing
# ===================================================================
def worker(code_str: str,
           instances,
           n_cities: int,
           ref_point: np.ndarray,
           fixed_ideal: np.ndarray,
           fixed_nadir: np.ndarray,
           queue):
    try:
        local_ns = {}
        exec(code_str, {
            "np": np,
            "random": random,
            "List": List,
            "Tuple": Tuple,
            "time": time,
            "warnings": warnings
        }, local_ns)

        func = local_ns["select_neighbor"]
        hv, t = evaluate_fixed(instances, n_cities, ref_point, func,
                               fixed_ideal, fixed_nadir)
        queue.put([float(hv), float(t)])
    except Exception as e:
        import traceback
        queue.put(f"Error: {e}\n{traceback.format_exc()}")


# ===================================================================
# 6. MAIN
# ===================================================================
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    random.seed(42)

    # Đọc danh sách heuristic
    with open("test/samples_1~300.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    problem_sizes = [200]          # Thêm bao nhiêu size cũng được
    n_instances = 4
    ref_point = np.array([1.1, 1.1])

    results = {}

    for ps in problem_sizes:
        print("\n" + "=" * 75)
        print(f"   ĐÁNH GIÁ HOÀN TOÀN CÔNG BẰNG – {ps} THÀNH PHỐ")
        print("=" * 75)

        # 1. Sinh instance cố định (một lần duy nhất)
        instances = GetData(n_instances, ps).generate_instances()
        print(f"→ Đã cố định {n_instances} instance (seed=2025)")

        # 2. Tính Z_ideal / Z_nadir cố định (độc lập với mọi heuristic)
        print("→ Đang ước lượng Z_ideal & Z_nadir cố định từ 20.000 tour ngẫu nhiên...")
        FIXED_IDEAL, FIXED_NADIR = compute_fixed_bounds(instances, ps, samples_per_inst=5000)
        print(f"   Z_ideal (CỐ ĐỊNH): {FIXED_IDEAL}")
        print(f"   Z_nadir (CỐ ĐỊNH): {FIXED_NADIR}")
        print("-" * 75)

        # 3. Đánh giá từng heuristic
        for idx, entry in enumerate(data):
            hid = idx + 1
            if "program" not in entry or not entry["program"].strip():
                continue

            code = entry["program"]
            desc = entry.get("algorithm", "No description")[:120]

            print(f"\n→ Heuristic {hid:3d}: {desc}...")

            q = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=worker,
                args=(code, instances, ps, ref_point, FIXED_IDEAL, FIXED_NADIR, q)
            )
            p.start()
            p.join(timeout=3600)

            if p.is_alive():
                p.terminate()
                p.join()
                score = "TIMEOUT"
                print("   → TIMEOUT")
            else:
                res = q.get()
                if isinstance(res, str) and res.startswith("Error"):
                    score = res.split("\n")[0]
                    print(f"   → {score}")
                else:
                    hv, t = res
                    score = [hv, t]
                    print(f"   → HV = {hv: .8f} | Time = {t: .2f}s")

            results.setdefault(hid, {})[ps] = score

        # Lưu kết quả sau mỗi size
        out_file = f"test/heuristic_scores_size_{ps}_FAIR.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n→ Đã lưu kết quả công bằng vào: {out_file}")

    print("\nHOÀN TẤT! Tất cả heuristic đã được đánh giá trên cùng một bộ instance và cùng Z_ideal/Z_nadir.")