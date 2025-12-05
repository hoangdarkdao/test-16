from __future__ import annotations
import logging
import numpy as np
import concurrent.futures
import copy
import time
import random
import traceback
from threading import Thread, Lock
from typing import Optional, Literal
from .extension import dominates, hypervolume_contribution
from .population import Population
from .mo_mcts import MCTS, MCTSNode
from .profiler import MOMCTSProfiler
from .prompt import MOMCTSPrompt
from .sampler import MOMCTSSampler
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function

from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase

class MOMCTS_AHD:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_sample_nums: Optional[int] = 100,
                 init_size: Optional[float] = 5,
                 pop_size: Optional[int] = 10,
                 selection_num: int = 2,
                 num_samplers: int = 4,  # the number of threads to sample in parallel
                 num_evaluators: int = 4,
                 alpha: float = 0.4,
                 lambda_0: float = 0.01,
                 review = True,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread',
                                                       'process'] = 'thread',
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            init_size       : population size, if set to 'None', EoH will automatically adjust this parameter.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            alpha           : a parameter for the UCT formula, which is used to balance exploration and exploitation.
            lambda_0        : a parameter for the UCT formula, which is used to balance exploration and exploitation.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self._init_pop_size = init_size
        self._pop_size = pop_size
        self._selection_num = selection_num
        self.review = review
        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(
            self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(
            self._template_program_str)

        # population, sampler, and evaluator
        self._population = Population(
            init_pop_size=init_size, pop_size=self._pop_size)
        self._sampler = MOMCTSSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(
            evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0  # the current function that is evaluated

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = min(
            self._max_sample_nums,
            10 * self._init_pop_size
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )
        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(
                llm, evaluation, self)  # ZL: necessary

        # Flash Reflection variables for long-term
        self.good_reflections = []  # Store phase 2 outputs where new heuristic is discovered (better score)
        self.bad_reflections = []   # Store phase 2 outputs where no improvement
        self.long_term_hints = ""   # Aggregate good reflections for long-term influence on code generation


    def _sample_evaluate_register(self, prompt, op: str, func_only=False, max_retries=3):
        for attempt in range(max_retries):
            sample_start = time.time()
            thought, func = self._sampler.get_thought_and_function(
                self._task_description_str, prompt
            )
            sample_time = time.time() - sample_start

            if func is None:
                print("New return in _sample_evaluate_register, momcts_ahd.py")
                return False

            program = TextFunctionProgramConverter.function_to_program(
                func, self._template_program
            )

            score, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time, program
            ).result()  # score is 2D or None

            if score is not None:
                break
            else:
                print(f"⚠️ Score is None (attempt {attempt+1}/{max_retries}), retrying...")

        if score is None:
            print("❌ Failed to get a valid score after retries.")
            return False

        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        print(f"✅ Success getting func information")

        if self._profiler is not None:
            print(f"gonna register!")
            self._profiler.register_function(prompt, func, program=str(program), op=op)
            self._tot_sample_nums += 1

        if func_only:
            return func

        if func.score is None:
            return False

        self._population.register_function(func)
        return True

    def _continue_loop(self) -> bool:
        if self._max_sample_nums is None:
            return True
        else:
            # if the current evaluation func number < max number, still do this
            return self._tot_sample_nums < self._max_sample_nums

    def check_duplicate(self, population, code: str):
        for ind in population:
            if code == ind.code:
                return True
        return False

    def check_duplicate_obj(self, population, score: list[float, float]):
        for ind in population:
            if np.array_equal(score, ind.individual.score):
                return True
        return False

    # add tree path for reasoning s1
    def population_management_s1(self, pop_input: list[Function], size=None):
        """
        Args:
            pop_input: list of individuals (each with .score attribute)
            size: optional, desired population size (ignored if larger than nondominated set)

        Returns:
            pop_new: list of non-dominated individuals only
        """
        # Filter valid individuals
        pop = [ind for ind in pop_input if ind.score is not None]
        if not pop:
            return []

        # Deduplicate by score
        unique_pop, seen_scores = [], set()
        for ind in pop:
            key = tuple(ind.score)
            if key not in seen_scores:
                unique_pop.append(ind)
                seen_scores.add(key)

        if not unique_pop:
            return []

        # Score matrix
        scores = np.array([ind.score for ind in unique_pop])

        # Perform non-dominated sorting
        nds = NonDominatedSorting()
        fronts = nds.do(scores)

        # Keep only the first front (nondominated solutions)
        nondominated_indices = fronts[0]
        nondominated = [unique_pop[i] for i in nondominated_indices]

        # Optionally truncate if 'size' is smaller than number of nondominated
        if size is not None and len(nondominated) > size:
            nondominated = nondominated[:size]

        print(f"Output for population management s1: {nondominated}")
        return nondominated


    def expand(self, mcts: MCTS, node_set: list[MCTSNode], cur_node: MCTSNode, option: str):
        print(f"Current node depth: {cur_node.depth}")  
        
        is_valid_func = True
        if option == 's1':
            path_set = []
            now = copy.deepcopy(cur_node)
            while now.algorithm != "Root":
                path_set.append(now.individual)
                now = copy.deepcopy(now.parent)
            # path_set = self.population_management_s1(path_set)
            if len(path_set) == 1:
                return node_set

            i = 0
            while i < 3:
                if self.review and self.long_term_hints is not None:
                    prompt = MOMCTSPrompt.get_prompt_s1(
                    self._task_description_str, path_set, self._function_to_evolve, self.long_term_hints)
                else:
                    prompt = MOMCTSPrompt.get_prompt_s1(
                        self._task_description_str, path_set, self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, option, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(
                    node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'e1':
            indivs = [copy.deepcopy(children.subtree[random.choices(range(len(children.subtree)), k=1)[0]].individual)
                      for
                      # so mcts.root.children becauses we only use e1 in initialization
                      children in mcts.root.children]
            
            prompt = MOMCTSPrompt.get_prompt_e1(
                self._task_description_str, indivs, self._function_to_evolve)
                
            func = self._sample_evaluate_register(prompt, option, func_only=True)
            if func is False:
                is_valid_func = False
            else:
                is_valid_func = (func.score is not None)

        elif option == 'e2':
            i = 0
            while i < 3:
                now_indiv = None
                while True:
                    now_indiv = self._population.selection()
                    if now_indiv != cur_node.individual:
                        break        
                
                indivs = [now_indiv, cur_node.individual] 
                
                if self.review and self.long_term_hints is not None:
                    prompt = MOMCTSPrompt.get_prompt_e2(
                                            self._task_description_str, indivs, self._function_to_evolve, self.long_term_hints)
                else:
                    prompt = MOMCTSPrompt.get_prompt_e2(
                    self._task_description_str, indivs, self._function_to_evolve)
                    
                func = self._sample_evaluate_register(prompt, option, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(
                    node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break
        

        elif option == 'm1':
            i = 0
            while i < 3:
                prompt = MOMCTSPrompt.get_prompt_m1(self._task_description_str, cur_node.individual,
                                                self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, option, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break
                
        elif option == 'm2':
            i = 0
            while i < 3:
                prompt = MOMCTSPrompt.get_prompt_m2(self._task_description_str, cur_node.individual,
                                                self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, option, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break
                
        elif option == 'elitist':
            i = 0
            while i < 3:  # Retries giới hạn để flash
                elites = self.population_management_s1(self._population.population) # survival is enough 
                if len(elites) < 2:
                    print("Not enough elites for elitist action.")
                    return node_set

                # Phase 1: Analyze elites để get flash reflection insights (như Prompt 5)
                prompt_phase1 = MOMCTSPrompt.get_flash_reflection_phase1_prompt(
                    self._task_description_str, elites, self._function_to_evolve
                )
                insights = self._sampler.get_thought(prompt_phase1)  # Chỉ get thought (nhanh), guide từ phase 1

                if not insights:
                    i += 1
                    continue

                # Adjust long-term reflection using guide from phase 1
                good_str = "\n".join(self.good_reflections[-5:]) if self.good_reflections else ""  # Limit to keep flash
                bad_str = "\n".join(self.bad_reflections[-5:]) if self.bad_reflections else ""
                prompt_phase2 = MOMCTSPrompt.get_flash_reflection_phase2_prompt(
                    self._task_description_str, insights, good_str, bad_str
                )
                long_term_guide = self._sampler.get_thought(prompt_phase2)  # Redefine to create adjusted long-term guide

                if not long_term_guide:
                    i += 1
                    continue

                # Use the adjusted long-term guide to generate code
                prompt_generate = MOMCTSPrompt.get_flash_generate_code_prompt(
                    self._task_description_str, elites, self._function_to_evolve, long_term_guide
                )
                func = self._sample_evaluate_register(prompt_generate, option, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(
                    node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    is_dominated = any(dominates(elite.score, func.score) for elite in elites)            
                    if not is_dominated: 
                        self.good_reflections.append(long_term_guide)
                    else:
                        self.bad_reflections.append(long_term_guide)

                    self.long_term_hints = long_term_guide  # Accumulate recent good ones
                    break
                
        elif option == 'r1':
            i = 0
            while i < 3:
                # Chọn heuristic có runtime tốt nhất từ population (ưu tiên low runtime)
                best_runtime_indiv = min(self._population.population, key=lambda ind: ind.score[1] if ind.score is not None else float('inf'))  # score[1] là runtime
                
              
                prompt = MOMCTSPrompt.get_prompt_r1(
                    self._task_description_str, best_runtime_indiv, self._function_to_evolve
                )
                
                func = self._sample_evaluate_register(prompt, option, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break    
            
            if not is_valid_func:
                print(f"Timeout in elitist action with flash reflection.")
                return node_set

            print(f"Action: {option}, Elites Obj: {[e.score for e in elites]}, New Obj: {func.score}, Depth: {cur_node.depth + 1}")

            if is_valid_func and np.any(func.score != float('-inf')):
                self._population.register_function(func)
                print(f"Passed score into MCTSNode in expand: {func.score}")
                # func.score here [NHV, runtime]
                now_node = MCTSNode(func.algorithm, str(func), func.score, individual=func,
                                    parent=cur_node, depth=cur_node.depth + 1, visit=0, raw_info=func)
                if option == 'e1':
                    now_node.subtree.append(now_node)
                cur_node.add_child(now_node)
                mcts.backpropagate(now_node, now_node.reward_vector)
                node_set.append(now_node)
            return node_set

        else:
            assert False, 'Invalid option!'

        if not is_valid_func:
            print(f"Timeout emerge, no expanding with action {option}.")
            return node_set

        if option != 'e1':
            print(
                f"Action: {option}, Father Obj: {cur_node.raw_info.score}, Now Obj: {func.score}, Depth: {cur_node.depth + 1}")
        else:
            if self.check_duplicate_obj(node_set, func.score):
                print(
                    f"Duplicated e1, no action, Father is Root, Abandon Obj: {func.score}")
            else:
                print(
                    f"Action: {option}, Father is Root, Now Obj: {func.score}")

        if is_valid_func and np.any(func.score != float('-inf')):
            self._population.register_function(func)
            print(f"Passed score into MCTSNode in expand: {func.score}")
            # func.score here [NHV, runtime]
            now_node = MCTSNode(func.algorithm, str(func), func.score, individual=func,
                                parent=cur_node, depth=cur_node.depth + 1, visit=0, raw_info=func)
            if option == 'e1':
                now_node.subtree.append(now_node)
            cur_node.add_child(now_node)
            mcts.backpropagate(now_node, now_node.reward_vector)
            node_set.append(now_node)
        return node_set

    def _iteratively_init_population_root(self):
        while len(self._population.population) < self._init_pop_size:
            try:
                prompt = MOMCTSPrompt.get_prompt_i1(self._task_description_str,
                                                self._function_to_evolve)
                self._sample_evaluate_register(prompt, op = 'i1')
                self._population.survival()
                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    print(
                        f'Note: During initialization, EoH gets {len(self._population) + len(self._population._next_gen_pop)} algorithms '
                        f'after {self._initial_sample_nums_max} trails.')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _init_one_solution(self):
        while len(self._population._next_gen_pop) == 0:
            try:
                prompt = MOMCTSPrompt.get_prompt_i1(
                    self._task_description_str, self._function_to_evolve)
                self._sample_evaluate_register(prompt, op = 'i1')
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue
    
    def _multi_threaded_sampling(self, fn: callable, tasks=None, *args, **kwargs):
        sampler_threads = []
        lock = Lock()  # create a lock for thread safety

        def safe_wrapper(*a, **k):
            # critical section guarded by lock
            with lock:
                fn(*a, **k)

        if tasks is None:  # mode 1: same task for all threads
            sampler_threads = [
                Thread(target=safe_wrapper, args=args, kwargs=kwargs)
                for _ in range(self._num_samplers)
            ]
        else:  # mode 2: each task has its own args/kwargs
            for task in tasks:
                if isinstance(task, tuple):
                    task_args, task_kwargs = task if len(task) == 2 else (task, {})
                else:
                    task_args, task_kwargs = (task,), {}
                sampler_threads.append(
                    Thread(target=safe_wrapper, args=task_args, kwargs=task_kwargs)
                )

        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()
                
    def run(self):
        self._init_one_solution()
        self._population.survival()
        
        self.mcts = MCTS('Root', num_objectives=2, alpha=self.alpha,
                         exploration_constant_0=self.lambda_0)
        start_time = time.time()
        self._multi_threaded_sampling(self._iteratively_init_population_root)
        
        print(f"Perform multi thread initialization in: {time.time() - start_time}")
        # 3. update mcts
        for indiv in self._population.population:
            now_node = MCTSNode(indiv.algorithm, str(indiv), indiv.score, individual=indiv,
                                parent=self.mcts.root,
                                depth=1, visit=0, raw_info=indiv)
            self.mcts.root.add_child(now_node)
            self.mcts.backpropagate(now_node, now_node.reward_vector)
            now_node.subtree.append(now_node)
          
        if len(self._population) < self._selection_num:
            print(
                f'The search is terminated since MCTS_AHD unable to obtain {self._selection_num} feasible algorithms during initialization. '
                f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                f'Please also check your evaluation implementation and LLM implementation.')
            return

        # evolutionary search
        n_op = ['e1', 'e2', 'm1', 'm2', 's1', 'elitist', 'r1']  
        op_weights = [0, 1, 1, 1, 0, 1, 2] 
        while self._continue_loop():  # if current evaluation < max evaluation, still evaluate function
            node_set = []
            print(
                f"Current number of MCTS nodes in the subtree of each child of the root: {[len(node.subtree) for node in self.mcts.root.children]}")
            cur_node = self.mcts.root
            while len(cur_node.children) > 0 and cur_node.depth < self.mcts.max_depth:
                # change this to choose the single best child from a parent node
                next_node = self.mcts.best_child(cur_node)
                if next_node is None:
                    break

                # here is progressive something
                if int((cur_node.visits) ** self.mcts.alpha) > len(cur_node.children):
                    if cur_node == self.mcts.root:
                        op = 'e1'
                        print("Perfrom e1 operation")
                        self.expand(
                            self.mcts, self.mcts.root.children, cur_node, op)
                    else:
                        op = 'e2'
                        print("Perfrom 'e2' operation")
                        self.expand(self.mcts, cur_node.children, cur_node, op)

                cur_node = next_node
            

            tasks = []
            for i in range(len(n_op)):
                op = n_op[i]  # get operation
                print(f"Get operation {op}")
                print(
                    f"Iter: {self._tot_sample_nums}/{self._max_sample_nums} OP: {op}", end="|")
                # get the max number of performing that operation
                op_w = op_weights[i]
                
                for j in range(op_w):
                    tasks.append((self.mcts, node_set, cur_node, op))
                
            self._multi_threaded_sampling(self.expand, tasks)
            print(f"Multi sampling successful!")
            self._population.survival() 
            
            if isinstance(self._profiler, MOMCTSProfiler):
                self._profiler.register_population(self._population)
           
        if self._profiler is not None:
            self._profiler.finish()

        self._sampler.llm.close()