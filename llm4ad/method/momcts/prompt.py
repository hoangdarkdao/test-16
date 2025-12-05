from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *
from llm4ad.method.momcts.mo_mcts import MCTSNode


class MOMCTSPrompt:
    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @staticmethod
    def _format_score(score):
        """Format score to a comma-separated string of objective values.
        Handles both iterable (multi-objective) and scalar scores.
        We negate values as the original code displayed -score.
        """
        try:
            return ', '.join(str(-v) for v in score)
        except Exception:
            return str(-score)

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function):
        # template
        temp_func = copy.deepcopy(template_function)
        prompt_content = f'''{task_prompt}
1. First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside within boxed {{}}. 
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        
        temp_func = copy.deepcopy(template_function)
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'

        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}

Analyze the logic of all the given code snippets carefully. Then identify the two code snippets whose logic is most different from each other
and create a new algorithm that totally different in logic and form from both of them.
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content

    
    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function, suggestions = None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        if suggestions is not None:
            prompt_content = f'''{task_prompt}
            I have {len(indivs)} existing algorithms with their codes as follows:
            {indivs_prompt}

            Additionally, here is a long-term reflection that provides higher-level guidance for improvement:
            {suggestions}

            Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them and the above long-term reflection.
            1. Firstly, identify the common backbone idea in the provided algorithms.
            2. Secondly, based on both the backbone idea and the long-term reflection, describe your new algorithm in one long, detailed sentence. The description must be enclosed within boxed {{}}.
            3. Thirdly, implement the following Python function:
            {str(temp_func)}

            Check syntax and code carefully before returning the final function. Do not give any additional explanations.
            '''
        else:
            prompt_content = f'''{task_prompt}
            I have {len(indivs)} existing algorithms with their codes as follows:
            {indivs_prompt}
            Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
            1. Firstly, identify the common backbone idea in the provided algorithms. 
            2. Secondly, based on the backbone idea describe your new algorithm in one long, detail sentence. The description must be inside within boxed {{}}.
            3. Thirdly, implement the following Python function:
            {str(temp_func)}
            Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content

    
    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
       
        temp_func = copy.deepcopy(template_function)

        prompt_content = f'''{task_prompt}
        I have one algorithm with its code as follows. Algorithm description:
        {indi.algorithm}
        Code:
        {str(indi)}
        Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.
        1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
        2. Next, implement the idea in the following Python function:
        {str(temp_func)}
        Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content
    
    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')

        temp_func = copy.deepcopy(template_function)
        prompt_content = f'''{task_prompt}
        I have one algorithm with its code as follows. Algorithm description:
        {indi.algorithm}
        Code:
        {str(indi)}
        Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm.
        1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
        2. Next, implement the idea in the following Python function:
        {str(temp_func)}
        Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content
    
    @classmethod
    def get_prompt_s1(cls, task_prompt: str, indivs: List[Function], template_function: Function, suggestion = None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        temp_func = copy.deepcopy(template_function)

        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            
            indi.docstring = ''
            indivs_prompt += (
                f"No. {i + 1} algorithm's description and the corresponding code are:\n"
                f"{indi.algorithm}\n{str(indi)}\n"
            )
        
        if suggestion is not None:
            prompt_content = f'''{task_prompt}
            I have {len(indivs)} existing algorithms with their codes as follows:
            {indivs_prompt}

            Additionally, here is a long-term reflection that provides higher-level guidance for improvement:
            {suggestion}

            Please help me create a new algorithm that is inspired by all the above algorithms and the long-term reflection, aiming to achieve objective values lower than any of them.

            1. Firstly, list some ideas in the provided algorithms and the long-term reflection that are clearly helpful for designing a better algorithm.
            2. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one long, detailed sentence. The description must be enclosed within boxed {{}}.
            3. Thirdly, implement the idea in the following Python function:
            {str(temp_func)}

            Check syntax, code carefully before returning the final function. Do not give any additional explanations.
            '''
        else:   
            prompt_content = f'''{task_prompt}
            I have {len(indivs)} existing algorithms with their codes as follows:
            {indivs_prompt}
            Please help me create a new algorithm that is inspired by all the above algorithms with its objective values lower than any of them.

            1. Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm.
            2. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. The description must be inside within boxed {{}}.
            3. Thirdly, implement the idea in the following Python function:

            {str(temp_func)}
            Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content
    
    
    
    @classmethod
    def get_flash_reflection_phase1_prompt(cls, task_prompt: str, sorted_indivs: List[Function], template_function: Function):
        indivs_prompt = ""
        for i, indi in enumerate(sorted_indivs):
            indi.docstring = ''
            indivs_prompt += f"No. {i+1}: Description: {indi.algorithm}\nCode: {str(indi)}\n"

        prompt_content = f'''{cls.get_system_prompt()}\n{task_prompt}
        ### List heuristics
        Below is a list of design heuristics grouped by dominance relationships (Pareto principle).
        {indivs_prompt}
        ### Guide
        - Keep in mind, heuristics are **grouped by dominance** rather than ranked linearly.
        - The **Nondominated** group represents the best trade-offs among objectives.
        - The **Dominated** group contains heuristics that are outperformed on at least one objective.
        - The response in Markdown style and nothing else has the following structure:
        '**Analysis:**\n**Experience:**'
        In there:
        + Meticulously analyze **comments, docstrings, and source code** of several pairs or groups of heuristics across and within these groups to fill **Analysis:**.
        Example: “Comparing nondominated vs dominated heuristics, we see ...; Within nondominated ones, comparing ...; Overall: ...”
        + Self-reflect to extract useful experience for designing better heuristics and fill **Experience:** (< 60 words).
        I’m going to tip $999K for a better nondominated heuristic design! Let’s think step by step.'''
        return prompt_content

    @classmethod
    def get_flash_reflection_phase2_prompt(cls, task_prompt: str, current_short_term: str, good_reflection: str, bad_reflection: str):
        prompt_content = f'''{task_prompt}
        Your task is to redefine ‘Current self-reflection’ paying attention to avoid all things in ‘Ineffective self-reflection’ in order to come up with ideas to design better heuristics.
        ### Current self-reflection
        {current_short_term}
        {good_reflection}
        ### Ineffective self-reflection
        {bad_reflection}
        Response (<100 words) should have 4 bullet points: Keywords, Advice, Avoid, Explanation.
        I’m going to tip $999K for a better heuristics! Let’s think step by step.'''
        return prompt_content

    @classmethod
    def get_flash_generate_code_prompt(cls, task_prompt: str, indivs: List[Function], template_function: Function, long_term_guide: str):
        temp_func = copy.deepcopy(template_function)
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += (
                f"No. {i + 1} algorithm's description and the corresponding code are:\n"
                f"{indi.algorithm}\n{str(indi)}\n"
            )
            
        prompt_content = f'''{task_prompt}
        I have {len(indivs)} existing algorithms with their codes as follows:
        {indivs_prompt}
        Adjusted long-term guide: {long_term_guide}
        Please create a new algorithm inspired by above with better objectives, using the long-term guide.
        1. Describe new algorithm in one sentence. Boxed {{}}.
        2. Implement:
        {str(temp_func)}
        Check syntax. No extra explanations.'''
        return prompt_content
    
    @classmethod
    def get_prompt_r1(cls, task_prompt: str, indi: Function, template_function: Function, suggestions=None):
        assert hasattr(indi, 'algorithm')
        
        temp_func = copy.deepcopy(template_function)
        
        prompt_content = f'''{task_prompt}
        I have one algorithm with its code as follows. Algorithm description:
        {indi.algorithm}
        Code:
        {str(indi)}
        
        Your **sole objective** is to reduce the runtime of the algorithm as much as possible.
        Ignore any performance metrics other than runtime.
        
        Please create a new algorithm that is a modified, faster version of the provided one.
        Focus on optimizations such as reducing algorithmic complexity, improving loops,
        using efficient data structures, removing redundant computations, and simplifying logic.
        
        1. First, identify parts of the code that most likely cause high runtime and suggest optimizations.
        2. Next, describe your new optimized algorithm and its main steps in one detailed sentence. The description must be inside boxed {{}}.
        3. Finally, implement the optimized algorithm in the following Python function:
        {str(temp_func)}
        
        Check syntax and code carefully before returning the final function. Do not give additional explanations.'''
        
        # if suggestions is not None:
        #     prompt_content += f"\nAdditionally, incorporate this long-term reflection for higher-level guidance:\n{suggestions}"
        
        return prompt_content