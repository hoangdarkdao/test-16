from typing import List
from ...base import *

def dominates(objective_a: List[float], objective_b: List[float]) -> bool:
    '''
    Args: 
        minimization problme
    '''
    is_strictly_better = False
    for i in range(len(objective_a)):
        if objective_a[i] > objective_b[i]:
            return False
        if objective_a[i] < objective_b[i]:
            is_strictly_better = True
    return is_strictly_better

def hypervolume_contribution(objective: List[float], pareto_front: List[Function]) -> float:
    if not pareto_front:
        return 0.0
    distances = [
        sum([(objective[i] - other.score[i])**2 for i in range(len(objective))])**0.5
        for other in pareto_front
    ] # tinh khoang cach cua objective hien tai voi cac diem trong pareto_front, chon thang be nhat
    return min(distances)

