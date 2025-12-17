# test4.py – PHIÊN BẢN CHUẨN CHO BI_KP (TƯƠNG THÍCH LLM4AD 2025)
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
    "logs/mpage/tsp_semo/nhv_runtime/v3/samples/samples_1~300.json",
    "logs/mpage/tsp_semo/nhv_runtime/v3/samples/samples_301~600.json",
]

PROBLEM = "bi_tsp"  # chỉ hỗ trợ bi_kp trong file này

# ===================================================================
# CẤU HÌNH TỰ ĐỘNG
# ===================================================================
CONFIG = {
    "bi_tsp":  {"eval": "llm4ad/task/optimization/bi_tsp_semo/evaluation.py",
                "inst": "llm4ad/task/optimization/bi_tsp_semo/get_instance.py",
                "sizes": [200], "n_inst": 4,  "ref": [1.1, 1.1]},
    "bi_kp":   {"eval": "llm4ad/task/optimization/bi_kp/evaluation.py",
                "inst": "llm4ad/task/optimization/bi_kp/get_instance.py",
                "sizes": [100], "n_inst": 10, "ref": [1.1, 1.1]},
    "bi_cvrp": {"eval": "llm4ad/task/optimization/bi_cvrp/evaluation.py",
                "inst": "llm4ad/task/optimization/bi_cvrp/get_instance.py",
                "sizes": [100], "n_inst": 5,  "ref": [1.1, 1.1]},
    "tri_tsp": {"eval": "llm4ad/task/optimization/tri_tsp_semo/evaluation.py",
                "inst": "llm4ad/task/optimization/tri_tsp_semo/get_instance.py",
                "sizes": [100], "n_inst": 5,  "ref": [1.1, 1.1, 1.1]},
}

cfg = CONFIG[PROBLEM]
print(f"ĐÁNH GIÁ CHUẨN LLM4AD: {PROBLEM.upper()}")
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
        print(f"Đã thêm vào PYTHONPATH: {root}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod

eval_mod = import_mod(cfg["eval"])
inst_mod = import_mod(cfg["inst"])
print("Import evaluation.py và get_instance.py thành công!\n")

# ===================================================================
# WORKER – ĐÃ SỬA CHUẨN THEO evaluation.py CỦA BI_KP
# ===================================================================
def worker(code_str: str, instance_data, problem_size: int, ref_point, queue):
    try:
        import random
        local_ns = {}
        exec(code_str, {"np": np, "random": random}, local_ns)
        func = local_ns.get("select_neighbor")
        if func is None:
            raise NameError("Không tìm thấy hàm select_neighbor trong code")

        # Gọi đúng hàm evaluate từ evaluation.py
        neg_mean_hv, mean_time = eval_mod.evaluate(
            instance_data=instance_data,           # list of (w, v1, v2)
            n_instance=len(instance_data),
            problem_size=problem_size,
            ref_point=np.array(ref_point),
            eva=func
        )

        actual_hv = -neg_mean_hv  # Vì trong evaluation.py lưu -HV
        queue.put([float(actual_hv), float(mean_time)])

    except Exception as e:
        import traceback
        msg = f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n[ERROR] {msg}")
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
            print(f"Không tìm thấy file: {path} → bỏ qua")
            continue

        print(f"\nĐANG XỬ LÝ: {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = {}

        for ps in cfg["sizes"]:
            print(f"\n--- Problem size: {ps} ---")

            # Tạo instances cố định (cùng seed → công bằng)
            getter = inst_mod.GetData(cfg["n_inst"], ps)
            instance_data = getter.generate_instances()  # trả về list (w, v1, v2, capacity?) nhưng thực tế là list (w,v1,v2)

            print(f"Đã tạo {cfg['n_inst']} instance cố định (seed cố định)")

            for idx, entry in enumerate(data):
                hid = idx + 1
                if "program" not in entry or not entry["program"].strip():
                    print(f"Heuristic {hid:3d}: [SKIP] Không có code")
                    continue

                code = entry["program"]
                algorithm_name = entry.get("algorithm", "No name")
                desc = str(algorithm_name).strip()[:90]
                if not desc:
                    desc = "No name"

                print(f"Heuristic {hid:3d}: {desc}... ", end="", flush=True)

                q = multiprocessing.Queue()
                p = multiprocessing.Process(
                    target=worker,
                    args=(code, instance_data, ps, cfg["ref"], q)
                )
                p.start()
                p.join(timeout=3600)  # 1 giờ timeout

                if p.is_alive():
                    p.terminate()
                    p.join()
                    score = "TIMEOUT"
                    print("TIMEOUT")
                else:
                    res = q.get()
                    if isinstance(res, str):  # lỗi
                        score = res.split("\n")[0]
                        print("ERROR")
                    else:
                        hv, t = res
                        score = [round(float(hv), 6), round(float(t), 1)]
                        print(f"HV = {hv:.6f} | Time = {t:.1f}s")

                results.setdefault(hid, {})[ps] = score

        # LƯU KẾT QUẢ THEO CẤU TRÚC BẠN MUỐN: list các {"score": [hv, time]}
        output_list = []
        current_size = cfg["sizes"][0]  # 100 (int, giữ nguyên int)

        for idx, entry in enumerate(data):
            hid = idx + 1
            if "program" not in entry or not entry["program"].strip():
                # Nếu heuristic bị skip (không có program), vẫn thêm placeholder để giữ thứ tự
                output_list.append({"score": None})
                continue

            # Lấy score đúng với key là int (current_size)
            score_for_size = results.get(hid, {}).get(current_size, None)

            if isinstance(score_for_size, list) and len(score_for_size) == 2:
                hv, t = score_for_size
                output_list.append({"score": [float(hv), float(t)]})
            else:
                # Bao gồm cả TIMEOUT, ERROR, hoặc không chạy
                output_list.append({"score": None})

        # Lưu file
        out = path.stem + f"_FAIR_{PROBLEM.upper()}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)
        print(f"\nHOÀN TẤT → Đã lưu: {out} (danh sách {len(output_list)} heuristic)\n" + "-"*80)
        

    print("TẤT CẢ XONG! BẠN ĐÃ CÓ KẾT QUẢ CÔNG BẰNG NHẤT!")























