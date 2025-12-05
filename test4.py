# test3.py – PHIÊN BẢN CUỐI CÙNG, CHẠY ĐÚNG VỚI LLM4AD
import importlib
import json
import numpy as np
import multiprocessing
import time
import warnings
import sys
from pathlib import Path

# ===================================================================
# CHỈ CẦN SỬA 2 DÒNG NÀY MỖI LẦN CHẠY
# ===================================================================
INPUT_JSON_FILES = [
    "logs/meoh/tsp_semo/nhv_runtime_20/v1/samples/samples_1~300.json",
    #"logs/momcts/tsp_semo/nhv_runtime_20/v30/samples/samples_301~600.json",
    # "test/samples_301~600.json",
]

PROBLEM = "bi_tsp"  # bi_tsp | bi_kp | bi_cvrp

# ===================================================================
# CẤU HÌNH TỰ ĐỘNG
# ===================================================================
CONFIG = {
    "bi_tsp":  {"eval": "llm4ad/task/optimization/bi_tsp_semo/evaluation.py",
                "inst": "llm4ad/task/optimization/bi_tsp_semo/get_instance.py",
                "sizes": [100], "n_inst": 4,  "ref": [1.1, 1.1]},
    "bi_kp":   {"eval": "llm4ad/task/optimization/bi_kp_semo/evaluation.py",
                "inst": "llm4ad/task/optimization/bi_kp_semo/get_instance.py",
                "sizes": [200], "n_inst": 10, "ref": [1.1, 1.1]},
    "bi_cvrp": {"eval": "llm4ad/task/optimization/bi_cvrp_semo/evaluation.py",
                "inst": "llm4ad/task/optimization/bi_cvrp_semo/get_instance.py",
                "sizes": [100], "n_inst": 5,  "ref": [1.1, 1.1]},
}

cfg = CONFIG[PROBLEM]
print(f"ĐÁNH GIÁ CÔNG BẰNG: {PROBLEM.upper()}")
print(f"Files: {INPUT_JSON_FILES}\n" + "="*80)

# ===================================================================
# IMPORT + PYTHONPATH
# ===================================================================
def import_mod(path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    root = Path(__file__).parent.parent.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        print(f"PYTHONPATH: {root}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod

eval_mod = import_mod(cfg["eval"])
inst_mod = import_mod(cfg["inst"])
print("Import evaluation.py và get_instance.py thành công!\n")

# ===================================================================
# WORKER – ĐÃ SỬA ĐÚNG THEO LLM4AD
# ===================================================================
def worker(code_str, instances, ps, ref_point, queue):
    try:
        import random
        local_ns = {}
        exec(code_str, {"np": np, "random": random}, local_ns)
        func = local_ns.get("select_neighbor")
        if func is None:
            raise NameError("Không tìm thấy hàm select_neighbor")

        # CHỈ TRUYỀN NHỮNG GÌ evaluate() CHẤP NHẬN
        hv, t = eval_mod.evaluate(
            instance_data=instances,
            n_instance=len(instances),
            problem_size=ps,
            ref_point=np.array(ref_point),
            eva=func
            # KHÔNG truyền fixed_ideal / fixed_nadir → đúng với llm4ad
        )
        queue.put([float(hv), float(t)])
    except Exception as e:
        import traceback
        msg = f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n[DEBUG] {msg}")
        queue.put(msg)

# ===================================================================
# MAIN
# ===================================================================
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    for json_file in INPUT_JSON_FILES:
        path = Path(json_file)
        if not path.exists():
            print(f"Không tìm thấy: {path} → bỏ qua")
            continue

        print(f"\nĐANG XỬ LÝ: {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = {}

        for ps in cfg["sizes"]:
            print(f"\n--- Size: {ps} ---")

            getter = inst_mod.GetData(cfg["n_inst"], ps)
            instances = getter.generate_instances()
            print(f"Đã tạo {cfg['n_inst']} instance cố định")

            for idx, entry in enumerate(data):
                hid = idx + 1
                if "program" not in entry or not entry["program"].strip():
                    continue

                code = entry["program"]
                # Sửa đổi tại dòng 127
                algorithm_value = entry.get("algorithm")

                # Kiểm tra nếu giá trị là None (hoặc là chuỗi rỗng sau khi strip() để an toàn hơn)
                if not algorithm_value or not str(algorithm_value).strip():
                    desc = "No name"
                else:
                    # Chuyển thành chuỗi (đề phòng giá trị là số hoặc object khác) rồi cắt
                    desc = str(algorithm_value)[:90]

                # Kiểm tra nếu giá trị là None (hoặc là chuỗi rỗng sau khi strip() để an toàn hơn)
                if not algorithm_value or not str(algorithm_value).strip():
                    desc = "No name"
                else:
                    # Chuyển thành chuỗi (đề phòng giá trị là số hoặc object khác) rồi cắt
                    desc = str(algorithm_value)[:90]

                print(f"Heuristic {hid:3d}: {desc}... ", end="", flush=True)

                q = multiprocessing.Queue()
                p = multiprocessing.Process(
                    target=worker,
                    args=(code, instances, ps, cfg["ref"], q)
                )
                p.start()
                p.join(timeout=3600)

                if p.is_alive():
                    p.terminate(); p.join()
                    score = "TIMEOUT"
                    print("TIMEOUT")
                else:
                    res = q.get()
                    if isinstance(res, str):
                        score = res.split("\n")[0]
                        print("ERROR")
                    else:
                        hv, t = res
                        score = [float(hv), float(t)]
                        print(f"HV = {hv:.6f} | Time = {t:.1f}s")

                results.setdefault(hid, {})[ps] = score

        # LƯU KẾT QUẢ
        out = path.stem + f"_FAIR_{PROBLEM.upper()}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nHOÀN TẤT → Đã lưu: {out}\n" + "-"*80)

    print("TẤT CẢ XONG! BẠN ĐÃ CÓ KẾT QUẢ CÔNG BẰNG NHẤT!")
