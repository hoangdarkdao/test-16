from __future__ import annotations
import random
import copy
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import math
from copy import deepcopy
import numpy as np
from typing import List, Tuple, Any, Optional  # Ensure Optional is imported

class MCTSNode:
    def __init__(self, algorithm, code, obj: List[float], individual=None, depth=0, is_root=False, parent=None, visit=0, raw_info=None):
        self.algorithm = algorithm
        self.code: str = code
        self.parent: MCTSNode = parent
        self.individual = individual
        self.depth: int = depth
        self.rewards_collected: List[List[float]] = []
        self.children: List[MCTSNode] = []  # list of MCTSNode class
        self.children_info: List[dict] = []  # Raw info dictionaries of children, often used for prompting LLMs
        self.visits: int = visit
        self.subtree: List[MCTSNode] = []
        self.raw_info: List[MCTSNode] = raw_info
        self.reward_vector: List[float] = np.array(obj)  

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.algorithm}, visits={self.visits})"


class MCTS:
    def __init__(self, root_answer: Any, num_objectives: int, exploration_constant_0: float = 0.1, alpha: float = 0.5):
        self.exploration_constant_0 = exploration_constant_0  # Parameter for UCB
        self.num_objectives = num_objectives
        self.max_depth = 10
        self.epsilon = 1e-10
        self.alpha = alpha # used for progressive widenning
        self.root = MCTSNode(algorithm=root_answer, code=root_answer, obj=[0.0] * num_objectives,
                             is_root=True)
        self.global_pareto_front: List[List[float, float]] = []
        self.rewards = []
        self.selected_nodes: List[MCTSNode] = []
        self.rank_list = []
        
    @staticmethod
    def dominates(reward_a: List[float], reward_b: List[float]) -> bool: 
        '''
        Args:
            Minimization problem
        '''
        if reward_a is None or reward_b is None:
            return False
        
        is_strictly_better_on_at_least_one = False
        for i in range(len(reward_a)):
            if reward_a[i] > reward_b[i]:  
                return False 
            if reward_a[i] < reward_b[i]:  
                is_strictly_better_on_at_least_one = True
        return is_strictly_better_on_at_least_one

    @staticmethod
    def is_non_dominated(rewards: List[List[float]], new_reward: List[float]) -> bool:
       
        for r in rewards:
            if MCTS.dominates(r, new_reward):
                return False
        return True
    
    def update_pareto_front(self, new_reward: List[float]) -> List[List[float]]:
        """
        Updates the global Pareto front with a new reward vector,
        maintaining only non-dominated solutions.
        """
        # If new_reward is dominated by the current front, return unchanged
        if not self.is_non_dominated(self.global_pareto_front, new_reward):
            print(f"Dominated solution, pareto front keep the same, pareto front is: {self.global_pareto_front}")
            return self.global_pareto_front

        # Otherwise, add new_reward and prune dominated ones
        updated_front = [r for r in self.global_pareto_front if not self.dominates(new_reward, r)]
        updated_front.append(new_reward)

        # Update the global archive in place
        self.global_pareto_front = updated_front
        
        print(f"Updated pareto front: {self.global_pareto_front}")
        return self.global_pareto_front

    def backpropagate(self, node: MCTSNode, reward_vector: List[float]):
        
        current_node = node
        while current_node:
            current_node.visits += 1
            current_node.rewards_collected.append(reward_vector)
            current_node = current_node.parent

    def _calculate_hypervolume(self, front: List[List[float, float]]) -> float: 
        
        front_array = np.array(front) # [NHV, runtime]
        print(f"Current Pareto Front to calculate HV is: {front_array}")        
        if not front:
            return 0.0
        
        z_ideal = np.array([-1.5, 0]) # lower bound of [NHV, runtime]
        z_nadir = np.array([0, 20]) # upper bound of [NHV, runtime]
        
        print(f"Z_ideal: {z_ideal}, Z_nadir: {z_nadir}")
                
        metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                        norm_ref_point=False,
                        zero_to_one=True, # tell to normalize all points to [0, 1]
                        ideal=z_ideal,
                        nadir=z_nadir)
        
        hv = metric(front_array)
        print(f"Final HV indicator for current front: {hv}")
        return hv

    def _calculate_penalty(self, reward_vector: List[float],
                                        pareto_front: List[List[float]],
                                        reference_point: List[float]) -> Tuple[List[float], float]: # do not pass anything here
        '''
        Args:
            High level idea: calculate the distance from a dominated solution (reward_vector) to pareto front
        '''
        for p in pareto_front:
            if np.array_equal(p, reward_vector):
                return 0.0

        sorted_front = sorted(pareto_front, key=lambda x: x[0], reverse=True)
        
        r = np.array(reward_vector)
        z = np.array(reference_point)
        line_dir = r - z
        
        # 4. Find the intersection of this line with the Pareto front's envelope
        for i in range(len(sorted_front) - 1):
            p1 = np.array(sorted_front[i])
            p2 = np.array(sorted_front[i+1])
            
            # Define the line segment between two consecutive Pareto points
            front_dir = p2 - p1
            
            denom = line_dir[0] * front_dir[1] - line_dir[1] * front_dir[0] # if denom = 0, that mean 2 lines are parallel
        
            if abs(denom) > 1e-9: # Avoid division by zero
                t = ((p1[0] - z[0]) * front_dir[1] - (p1[1] - z[1]) * front_dir[0]) / denom
                u = -((p1[0] - z[0]) * line_dir[1] - (p1[1] - z[1]) * line_dir[0]) / denom
                
                if 0 <= t and 0 <= u <= 1:
                    projection_point = p1 + u * front_dir
                    penalty = np.linalg.norm(r - projection_point)
                    print(f"penalty score is: {penalty}")
                    return float(penalty)
        return 0.0 # Return default values

    def _calculate_multi_objective_ucb(self, child: MCTSNode, parent_visits: int) -> List[float, float]:
        
        avg_reward = []
        for i in range(self.num_objectives):
            avg = sum(r[i] for r in child.rewards_collected) / child.visits if child.visits > 0 else 0.0
            avg_reward.append(avg)
            
        print(f"Avg_reward for dim before normalization: {avg_reward}")
        
        exploration_term = self.exploration_constant_0 * math.sqrt(
            math.log(parent_visits + 1) / (child.visits + self.epsilon)
        )
        print(f"Exploration term: {exploration_term}")
        ucb_vector = [obj - exploration_term for obj in avg_reward]
        
        print(f"Final UCB Vector: {ucb_vector}")
        return ucb_vector 
    
    
    def best_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        if not node.children:
            return None

        best_child = None
        best_w_score = -float('inf')

        print(f"\nEvaluating {len(node.children)} children for node with {node.visits} visits")
        
        old_hv = self._calculate_hypervolume(self.global_pareto_front)
        dominated_child = []
        exist_non_dominated_child = 0
        
        for i, child in enumerate(node.children):
            print(f"\n--- Child {i+1} ---")
            print(f"Visits: {child.visits}")
            if child.visits == 0:
                return child
            
            r_sa = self._calculate_multi_objective_ucb(child, node.visits) # from a node and its parent visit, calculate the ucb vector
            print(f"UCB vector (r_sa): {r_sa}")
            # if any p dominates a -> so a is dominated
            is_dominated = any(self.dominates(p, r_sa) for p in self.global_pareto_front)
            print(f"Dominated by current Pareto front? {is_dominated}")
            
            if not is_dominated:
                # Case 1: Non-dominated solution
                exist_non_dominated_child = 1
                new_hv = self._calculate_hypervolume(self.global_pareto_front + [r_sa])
                w_score = new_hv - old_hv
                print(f"Non-dominated → HV improved from {old_hv:.4f} to {new_hv:.4f} (Δ={w_score:.4f})")
                
            else:
                dominated_child.append({
                    "index": i,
                    "child": child,
                    "r_sa": r_sa
                })
                w_score = -1000
                # penalty = self._calculate_penalty(r_sa, deepcopy(self.global_pareto_front), reference_point=[1.1] * self.num_objectives)
                # w_score = -penalty
                # print(f"Dominated → penalty = {penalty:.4f}, w_score = {w_score:.4f}")
            # 4. Select the child with the highest W(s,a) score
            print(f"Current w score of child: {w_score}")
            print(f"Child {i+1} final w_score: {w_score:.4f}")
            
            if w_score > best_w_score and not is_dominated:
                best_r_sa = r_sa
                best_w_score = w_score
                best_child = child
                print(f"→ Child {i+1} is the new best candidate with score {best_w_score:.4f}")
        
        if exist_non_dominated_child == 0:
            F = np.array([c["r_sa"] for c in dominated_child])

            # Perform non-dominated sorting (assuming minimization)
            nds = NonDominatedSorting()
            fronts = nds.do(F)

            # Reorder dominated_child based on fronts
            sorted_children = []
            for front in fronts:
                sorted_children.extend([dominated_child[i] for i in front])
                
            print(f"Dominated_child_non_dominated_sort: {sorted_children}")
            best_child = sorted_children[0]["child"]
            return best_child
        
        self.update_pareto_front(best_r_sa)
        
        print(f"\nFinal selection → best child with w_score = {best_w_score:.4f}")
        return best_child
    