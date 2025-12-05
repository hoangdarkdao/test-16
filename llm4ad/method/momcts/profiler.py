from __future__ import annotations

import json
import os
import numpy as np
from abc import ABC, abstractmethod
from threading import Lock # imagine multiple threads updating the same variables at the same time can get conflict, use this to avoid that
from typing import List, Dict, Optional
from .population import Population
from ...base import Function
from ...tools.profiler import ProfilerBase


class MOMCTSProfiler(ProfilerBase):

    def __init__(self,
                 log_dir: Optional[str] = None,
                 num_objs=2,
                 result_folder = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
       
        super().__init__(log_dir=log_dir,
                         num_objs=num_objs,
                         result_folder=result_folder,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         create_random_path=create_random_path,
                         **kwargs)
        self._cur_gen = 0
        self._pop_lock = Lock()
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, 'population')
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def register_population(self, pop: Population):
        try:
            self._pop_lock.acquire()
            if (self._num_samples == 0 or
                    pop.generation == self._cur_gen):
                return
            funcs = pop.population  # type: List[Function]
            funcs_json = []  # type: List[Dict]
            for f in funcs:
                f_score = f.score
                if f.score is not None:
                    if np.isinf(np.array(f.score)).any():
                        f_score = None
                    else:
                        f_score = f_score
                f_json = {
                    'algorithm': f.algorithm,
                    'function': str(f),
                    'score': f_score # [acc, runtime]
                }
                funcs_json.append(f_json)
            path = os.path.join(self._ckpt_dir, f'pop_{pop.generation}.json')
            with open(path, 'w') as json_file:
                json.dump(funcs_json, json_file, indent=4)
            self._cur_gen += 1
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    def _write_json(self, prompt: str, function: Function, program='', *, record_type='history', record_sep=300, **kwargs):
        """Write function data to a JSON file.
        Args:
            function   : The function object containing score and string representation.
            record_type: Type of record, 'history' or 'best'. Defaults to 'history'.
            record_sep : Separator for history records. Defaults to 200.
        """
        assert record_type in ['history', 'best']

        if not self._log_dir:
            return

        sample_order = self._num_samples

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):  # e.g., np.float32
                return obj.item()
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj

        content = {
            'prompt': prompt,
            'sample_order': sample_order,
            'algorithm': function.algorithm,
            'function': str(function),
            'score': convert(function.score),
            'program': program,
        }
        
        if 'op' in kwargs:
            content['operation'] = kwargs['op']

        if record_type == 'history':
            lower_bound = ((sample_order - 1) // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f'samples_{lower_bound + 1}~{upper_bound}.json'
        else:
            filename = 'samples_best.json'

        path = os.path.join(self._samples_json_dir, filename)

        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)

        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


