from __future__ import annotations
import math
from threading import Lock
from typing import List
import numpy as np
import traceback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.randomized_argsort import randomized_argsort
from .extension import dominates, hypervolume_contribution
from ...base import *


class Population:
    def __init__(self, init_pop_size, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population
        self._init_pop_size = init_pop_size
        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._elitist = []
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def elitist(self) -> List:
        return self._elitist

    @property
    def generation(self):
        return self._generation

    def register_function(self, func: Function):
        # we only accept valid functions
        print("Inside population.py, register_function")
        if func.score is None:
            return
        try:
            self._lock.acquire()
            # register to next_gen
            if not self.has_duplicate_function(func):
                self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size:
                self.survival()  # population management
        except Exception as e:
            traceback.print_exc()
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        if func.score is None:
            return True

        for i in range(len(self._population)):
            f = self._population[i]
            if str(f) == str(func):
                if func.score[0] > f.score[0]:
                    self._population[i] = func
                    return True
                if func.score[0] == f.score[0] and func.score[1] > f.score[1]:
                    self._population[i] = func
                    return True

        for i in range(len(self._next_gen_pop)):
            f = self._next_gen_pop[i]
            if str(f) == str(func):
                if func.score[0] > f.score[0]:
                    self._next_gen_pop[i] = func
                    return True
                if func.score[0] == f.score[0] and func.score[1] > f.score[1]:
                    self._next_gen_pop[i] = func
                    return True
        return False

    def selection(self, pop: List[Function] = None) -> Function:
        '''
            randomly choose 2 code from the population, get the dominant one
        '''
        if pop is None:
            pop = self._population
        funcs = [f for f in pop if not np.isinf(np.array(f.score)).any()]
        if len(funcs) > 1:
            a, b = np.random.choice(funcs, size=2, replace=False)
            if dominates(a.score, b.score):
                return a
            elif dominates(b.score, a.score):
                return b
            return a if np.random.rand() < 0.5 else b
        return funcs[0]

    def selection_e1(self, pop: List[Function] = None) -> Function:
        if pop is None:
            pop = self._population
        funcs = [f for f in pop if not np.isinf(np.array(f.score)).any()]
        return np.random.choice(funcs)

    def survival(self, pop_size: int = None):
        '''
        Args:
            Update self.population, keep the length to max is pop_size
        '''
        print("Inside population.py, def survival")
        if pop_size is None:
            pop_size = self._pop_size 
        pop = [ind for ind in self._population +
               self._next_gen_pop if ind.score is not None]
        if pop_size > len(pop):
            pop_size = len(pop)

        unique_pop, seen_scores = [], set()
        for ind in pop:
            key = tuple(ind.score)
            if key not in seen_scores:
                unique_pop.append(ind)
                seen_scores.add(key)

        possitive_scores = []

        for indi in unique_pop:
            obj1, obj2 = indi.score
            possitive_scores.append([obj1, obj2])
        print(f"Score array to perforrm nondomiated sort: {possitive_scores}")
        possitive_scores = np.array(possitive_scores)
        non_dot = NonDominatedSorting()
        fronts = non_dot.do(possitive_scores, return_rank=True)
        '''
        Suppose we have score array:
        [[1 2]
        [3 1]
        [4 5]]
        ([array([0, 1]), array([2])], array([0, 0, 1]))
        That means: front 1 has index 0 and 1, front 2 has index 2
        array 0, 0, 1: index 0 and 1 is rank 0, index 2 rank 1
        '''
        survivors = []  # list of individual
        print(f"Full fronts: {fronts}")
        for front_indices in fronts[0]:
            print(f"Front indices: {front_indices}")
            front_individuals = [unique_pop[i] for i in front_indices]
            if len(survivors) + len(front_individuals) <= pop_size:
                survivors.extend(front_individuals)
            else:
                remaining_slots = pop_size - len(survivors)
                if remaining_slots > 0:
                    front_individuals_scores = np.array(
                        # List[List[float]]
                        [ind.score for ind in front_individuals])
                    crowding_distances = get_crowding_function("cd").do(
                        front_individuals_scores, front_indices)

                    # Sort by crowding distance (higher is better for diversity)
                    sorted_indices = randomized_argsort(
                        crowding_distances, order="descending")
                    front_individuals = [front_individuals[i]
                                         for i in sorted_indices]

                    survivors.extend(front_individuals[:remaining_slots])
                break

        self._population = survivors
        self._next_gen_pop = []
        self._generation += 1
