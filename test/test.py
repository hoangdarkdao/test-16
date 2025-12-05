import json
import multiprocessing
import warnings
import numpy as np
import time
import os
from typing import Any, List, Dict, Tuple 
from pymoo.indicators.hv import Hypervolume 

# ====================================================================
# --- TÍCH HỢP NỘI DUNG TỪ get_instance.py ---
# ====================================================================
class GetData():
    """Lớp tạo instance bài toán Bi-objective Knapsack Problem (BI-KP)."""
    def __init__(self, n_instance: int, n_items: int):
        self.n_instance = n_instance
        self.n_items = n_items

    def generate_instances(self):
        np.random.seed(2025)
        instance_data = []
        for _ in range(self.n_instance):
            weights = np.random.rand(self.n_items)
            values_obj1 = np.random.rand(self.n_items)
            values_obj2 = np.random.rand(self.n_items)
            
            # Thiết lập Capacity dựa trên kích thước bài toán
            if 50 <= self.n_items < 100:
                capacity = 12.5
            elif 100 <= self.n_items <= 200:
                capacity = 25
            else:
                raise ValueError("Number of items must be between 50 and 200.")

            instance_data.append((weights, values_obj1, values_obj2))
        return instance_data, capacity

# Hàm phụ trợ knapsack_value
def knapsack_value(solution: np.ndarray, weight_lst: np.ndarray, value1_lst: np.ndarray, value2_lst: np.ndarray, capacity: float):
    """Tính toán giá trị mục tiêu (value1, value2) và kiểm tra tính khả thi."""
    
    # Kiểm tra tính khả thi
    if np.sum(solution * weight_lst) > capacity:
        return -1e10, -1e10  # Phạt nặng giải pháp không khả thi
    
    # Kiểm tra tính hợp lệ của mảng giải pháp
    if not np.all(np.isin(solution, [0, 1])) or len(solution) != len(weight_lst):
        return -1e10, -1e10
        
    total_val1 = np.sum(solution * value1_lst)
    total_val2 = np.sum(solution * value2_lst)
    return total_val1, total_val2

# ====================================================================
# --- TÍCH HỢP NỘI DUNG TỪ evaluation.py (BIKPEvaluation) ---
# ====================================================================
class BIKPEvaluation():
    """Lớp đánh giá thuật toán BI-KP."""
    def __init__(self, weight_lst: np.ndarray, value1_lst: np.ndarray, value2_lst: np.ndarray, capacity: float):
        self.weight_lst = weight_lst
        self.value1_lst = value1_lst
        self.value2_lst = value2_lst
        self.capacity = capacity
        
        self.ref_point = np.array([1.0, 1.0]) 

    def _random_solution(self):
        """Tạo một giải pháp ngẫu nhiên khả thi."""
        solution = np.zeros(len(self.weight_lst), dtype=int)
        current_weight = 0
        indices = np.random.permutation(len(self.weight_lst))
        for i in indices:
            if current_weight + self.weight_lst[i] <= self.capacity:
                solution[i] = 1
                current_weight += self.weight_lst[i]
        return solution

    def _dominates(self, a, b):
        """True nếu a thống trị b (maximization)."""
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    def evaluate_program(self, instance_name: str, select_neighbor_func: Any) -> tuple[np.ndarray, float]:
        
        archive = []
        # Tạo 10 giải pháp ngẫu nhiên ban đầu
        for _ in range(10):
            solution = self._random_solution()
            obj = knapsack_value(solution, self.weight_lst, self.value1_lst, self.value2_lst, self.capacity)
            if obj[0] > -1e9: 
                archive.append((solution, obj))

        start_time = time.time()
        
        # Thực hiện Local Search (100 lần lặp)
        max_iterations = 100
        for _ in range(max_iterations):
            try:
                neighbor_solution = select_neighbor_func(archive, self.weight_lst, self.value1_lst, self.value2_lst, self.capacity)
            except Exception as e:
                print(f"Algorithm execution failed during neighbor selection: {e}")
                break 

            neighbor_obj = knapsack_value(neighbor_solution, self.weight_lst, self.value1_lst, self.value2_lst, self.capacity)
            
            if neighbor_obj[0] > -1e9: 
                new_archive = []
                is_dominated = False
                for sol, obj in archive:
                    if self._dominates(obj, neighbor_obj):
                        is_dominated = True
                        break
                    if not self._dominates(neighbor_obj, obj):
                        new_archive.append((sol, obj))
                
                if not is_dominated:
                    new_archive.append((neighbor_solution, neighbor_obj))
                    archive = new_archive
        
        end_time = time.time()
        
        # Tính toán kết quả
        final_objs = np.array([obj for _, obj in archive])
        
        # SỬA LỖI HYPERVOLUME: Đảo dấu mục tiêu cho pymoo
        negated_objs = -final_objs
        
        hv = 0.0
        if len(negated_objs) > 0:
            if np.isnan(negated_objs).any() or np.isinf(negated_objs).any():
                return np.array([-1.0, len(archive)]), end_time - start_time
            
            try:
                ind = Hypervolume(ref_point=self.ref_point)
                hv = ind.do(negated_objs) 
            except Exception as e:
                print(f"Hypervolume calculation failed: {e}")
                hv = -1.0 

        # Trả về: [Hypervolume, Kích thước Archive], Thời gian
        return np.array([hv, len(archive)]), end_time - start_time

# ====================================================================
# --- HÀM CHÍNH ĐỂ CHẠY ĐÁNH GIÁ ---
# ====================================================================

# --- Hàm chạy đánh giá trong tiến trình riêng ---
def run_exec_and_eval(code_str: str, instance_params: dict, result_queue: multiprocessing.Queue):
    """Thực thi mã thuật toán và đánh giá nó."""
    try:
        local_vars = {}
        exec(code_str, globals(), local_vars) 
        select_neighbor_func = local_vars["select_neighbor"]
        
        tsp = BIKPEvaluation(
            weight_lst=instance_params['weights'],
            value1_lst=instance_params['values1'],
            value2_lst=instance_params['values2'],
            capacity=instance_params['capacity']
        )
        
        cst, tme = tsp.evaluate_program('_', select_neighbor_func) 
        result_queue.put([cst, tme])
    except Exception as e:
        result_queue.put(f"Error: {e}")

# --- Hàm chính để xử lý và đánh giá file JSON (ĐÃ SỬA) ---
def evaluate_algorithms_from_json(json_file_path: str, instance_sizes: list):
    """
    Trích xuất thuật toán từ file JSON, đánh giá, và cập nhật cấu trúc dữ liệu.
    :return: Danh sách các thuật toán đã được cập nhật scores.
    """
    warnings.filterwarnings("ignore")

    # 1. Đọc file JSON và lưu vào biến 'data'
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f) # data là một list các dictionary
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Lỗi đọc file JSON: {e}")
        return []

    # 2. Tạo tất cả các instance cần thiết trước
    all_instances = {}
    for size in instance_sizes:
        data_generator = GetData(n_instance=1, n_items=size)
        instance_data, capacity = data_generator.generate_instances()
        
        all_instances[size] = {
            'weights': instance_data[0][0],
            'values1': instance_data[0][1],
            'values2': instance_data[0][2],
            'capacity': capacity
        }
        print(f"Đã tạo instance cho kích thước {size} items (Capacity: {capacity}).")

    # 3. Lặp qua và đánh giá từng thuật toán
    for k in range(len(data)):
        algorithm_data = data[k] # Tham chiếu trực tiếp đến dictionary
        select_neighbor_code = algorithm_data.get("function")
        
        if not select_neighbor_code:
            print(f"Cảnh báo: Cá thể {k+1} không có trường 'function', bỏ qua.")
            continue

        algorithm_id = f"Algo_{k+1}"
        # Khởi tạo scores dictionary trong algorithm_data
        algorithm_data["scores"] = {} 

        for size in instance_sizes:
            print(f"Đánh giá {algorithm_id} trên kích thước {size} items...")
            instance_params = all_instances[size]

            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=run_exec_and_eval, 
                args=(select_neighbor_code, instance_params, result_queue)
            )
            p.start()
            p.join(timeout=300) 

            if p.is_alive():
                print(f"Thời gian chờ {algorithm_id} ({size} items) đã hết (5 phút).")
                p.terminate()
                p.join()
                score = "Timeout"
            else:
                try:
                    result = result_queue.get(timeout=5)
                except:
                    result = "Error: Process failed to return result via queue."

                if isinstance(result, str) and result.startswith("Error"):
                    print(f"Lỗi khi thực thi {algorithm_id} ({size} items): {result}")
                    score = result
                else:
                    cost, time_taken = result[0], result[1]
                    
                    if cost[0] == -1.0:
                        score = "Error: Hypervolume calculation failed."
                        print(f"Lỗi khi thực thi {algorithm_id} ({size} items): Hypervolume calculation failed.")
                    else:
                        score = {"hypervolume": cost[0], "archive_size": cost[1], "time": time_taken}

            # LƯU KẾT QUẢ TRỰC TIẾP VÀO dictionary của thuật toán
            algorithm_data["scores"][f"{size}_items"] = score

    return data # Trả về toàn bộ danh sách đã cập nhật

# --- CHẠY CHƯƠNG TRÌNH (ĐÃ SỬA) ---
if __name__ == '__main__':
    # Đặt đường dẫn đến file JSON đầu vào
    input_json_path = "test/samples_301~600.json" 
    
    # Đặt tên cho file JSON đầu ra mới
    output_json_path = "test/evaluation_results_with_algorithms.json" 
    
    # Định nghĩa các kích thước bài toán
    problem_sizes = [200] 

    try:
        print(f"Bắt đầu đánh giá các thuật toán từ {input_json_path}...")
        
        # Thực hiện đánh giá và nhận lại toàn bộ danh sách đã cập nhật
        updated_algorithms_data = evaluate_algorithms_from_json(input_json_path, problem_sizes)

        if updated_algorithms_data:
            # Lưu kết quả vào file JSON mới
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(updated_algorithms_data, f, indent=4, ensure_ascii=False)
            
            print("\n" + "="*50)
            print(f"✅ Đánh giá hoàn tất. Kết quả đã được lưu vào file:")
            print(f"   {output_json_path}")
            print("="*50)
            
            # In một phần kết quả để kiểm tra
            print("\n## Kiểm tra cấu trúc dữ liệu của Algo_1 ##")
            print(json.dumps(updated_algorithms_data[0], indent=4))
        else:
            print("\n⚠️ Không có dữ liệu để lưu do lỗi khi đọc file JSON.")

    except Exception as e:
        print(f"\nLỗi chính trong quá trình chạy chương trình: {e}")