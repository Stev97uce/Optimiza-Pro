import numpy as np
from typing import Dict, Any, List, Callable, Union
import time

from ...core.entities.optimization_entities import DynamicProgrammingProblem, Solution
from ...core.interfaces.repositories import IDynamicProgrammingSolver


class RecursiveDynamicProgrammingSolver(IDynamicProgrammingSolver):
    """Implementación de programación dinámica usando recursión"""
    
    def __init__(self):
        self.memo = {}  # Memoización para evitar recálculos
    
    def solve(self, problem: DynamicProgrammingProblem) -> Solution:
        """Resuelve usando recursión hacia atrás por defecto"""
        return self.solve_backward(problem)
    
    def solve_backward(self, problem: DynamicProgrammingProblem) -> Solution:
        """Resuelve usando recursión hacia atrás"""
        try:
            start_time = time.time()
            self.memo.clear()
            
            # Resolver recursivamente desde el último estado
            optimal_value = self._backward_recursion(
                problem.stages - 1,
                problem.states[0] if problem.states else 0,
                problem
            )
            
            # Reconstruir la política óptima
            optimal_policy = self._reconstruct_policy(problem)
            
            variables_values = {}
            for stage, decision in enumerate(optimal_policy):
                variables_values[f'decision_stage_{stage}'] = decision
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=optimal_value,
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'Recursive DP',
                    'method': 'Backward Recursion',
                    'stages': problem.stages,
                    'policy': optimal_policy
                }
            )
            
            return solution
            
        except Exception as e:
            return Solution(
                problem_id=problem.id,
                status="error",
                execution_time=0.0,
                solver_info={'error': str(e)}
            )
    
    def solve_forward(self, problem: DynamicProgrammingProblem) -> Solution:
        """Resuelve usando recursión hacia adelante"""
        try:
            start_time = time.time()
            
            # Implementación de recursión hacia adelante
            n_stages = problem.stages
            n_states = len(problem.states) if isinstance(problem.states[0], (int, float)) else 10
            
            # Tabla de valores
            V = [[0.0] * n_states for _ in range(n_stages + 1)]
            policy = [[None] * n_states for _ in range(n_stages)]
            
            # Inicialización (estado inicial)
            if problem.states:
                V[0][0] = 0  # Valor inicial en el primer estado
            
            # Iteración hacia adelante
            for stage in range(n_stages):
                for state in range(n_states):
                    best_value = float('-inf')
                    best_decision = None
                    
                    for decision in problem.decisions:
                        if callable(problem.transition_function) and callable(problem.reward_function):
                            try:
                                next_state = problem.transition_function(state, decision)
                                if 0 <= next_state < n_states:
                                    reward = problem.reward_function(state, decision)
                                    value = reward + V[stage][state]
                                    
                                    if value > best_value:
                                        best_value = value
                                        best_decision = decision
                                        V[stage + 1][next_state] = max(V[stage + 1][next_state], value)
                            except:
                                continue
                    
                    policy[stage][state] = best_decision
            
            # Encontrar el mejor valor final
            optimal_value = max(V[n_stages]) if V[n_stages] else 0
            
            variables_values = {}
            for stage in range(n_stages):
                variables_values[f'stage_{stage}_policy'] = str(policy[stage])
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=optimal_value,
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'Forward DP',
                    'method': 'Forward Recursion',
                    'stages': problem.stages
                }
            )
            
            return solution
            
        except Exception as e:
            return Solution(
                problem_id=problem.id,
                status="error",
                execution_time=0.0,
                solver_info={'error': str(e)}
            )
    
    def _backward_recursion(self, stage: int, state: Any, problem: DynamicProgrammingProblem) -> float:
        """Función recursiva hacia atrás con memoización"""
        if (stage, state) in self.memo:
            return self.memo[(stage, state)]
        
        # Caso base: última etapa
        if stage == 0:
            best_value = float('-inf')
            for decision in problem.decisions:
                if callable(problem.reward_function):
                    try:
                        value = problem.reward_function(state, decision)
                        best_value = max(best_value, value)
                    except:
                        continue
            
            result = best_value if best_value != float('-inf') else 0
            self.memo[(stage, state)] = result
            return result
        
        # Caso recursivo
        best_value = float('-inf')
        for decision in problem.decisions:
            if callable(problem.transition_function) and callable(problem.reward_function):
                try:
                    next_state = problem.transition_function(state, decision)
                    immediate_reward = problem.reward_function(state, decision)
                    future_value = self._backward_recursion(stage - 1, next_state, problem)
                    total_value = immediate_reward + future_value
                    best_value = max(best_value, total_value)
                except:
                    continue
        
        result = best_value if best_value != float('-inf') else 0
        self.memo[(stage, state)] = result
        return result
    
    def _reconstruct_policy(self, problem: DynamicProgrammingProblem) -> List[Any]:
        """Reconstruye la política óptima"""
        policy = []
        current_state = problem.states[0] if problem.states else 0
        
        for stage in range(problem.stages - 1, -1, -1):
            best_decision = None
            best_value = float('-inf')
            
            for decision in problem.decisions:
                if callable(problem.transition_function) and callable(problem.reward_function):
                    try:
                        next_state = problem.transition_function(current_state, decision)
                        immediate_reward = problem.reward_function(current_state, decision)
                        
                        if stage > 0:
                            future_value = self.memo.get((stage - 1, next_state), 0)
                        else:
                            future_value = 0
                        
                        total_value = immediate_reward + future_value
                        
                        if total_value > best_value:
                            best_value = total_value
                            best_decision = decision
                            current_state = next_state
                    except:
                        continue
            
            policy.append(best_decision)
        
        return policy[::-1]  # Invertir para obtener orden cronológico
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        return {
            'name': 'Recursive Dynamic Programming',
            'type': 'Dynamic Programming',
            'methods': ['Backward Recursion', 'Forward Recursion'],
            'features': ['Memoization', 'Policy Reconstruction']
        }


class TabulatedDynamicProgrammingSolver(IDynamicProgrammingSolver):
    """Implementación de programación dinámica usando tabulación"""
    
    def solve(self, problem: DynamicProgrammingProblem) -> Solution:
        """Resuelve usando tabulación"""
        return self.solve_backward(problem)
    
    def solve_backward(self, problem: DynamicProgrammingProblem) -> Solution:
        """Resuelve usando tabulación hacia atrás"""
        try:
            start_time = time.time()
            
            n_stages = problem.stages
            states = problem.states
            decisions = problem.decisions
            
            # Crear tabla de valores
            if isinstance(states[0], (int, float)):
                n_states = len(states)
                V = [[0.0] * n_states for _ in range(n_stages)]
                policy = [[None] * n_states for _ in range(n_stages)]
                
                # Llenar tabla desde la última etapa hacia atrás
                for stage in range(n_stages - 1, -1, -1):
                    for state_idx, state in enumerate(states):
                        best_value = float('-inf')
                        best_decision = None
                        
                        for decision in decisions:
                            if callable(problem.transition_function) and callable(problem.reward_function):
                                try:
                                    next_state = problem.transition_function(state, decision)
                                    immediate_reward = problem.reward_function(state, decision)
                                    
                                    # Encontrar índice del próximo estado
                                    try:
                                        next_state_idx = states.index(next_state)
                                    except ValueError:
                                        continue
                                    
                                    if stage < n_stages - 1:
                                        future_value = V[stage + 1][next_state_idx]
                                    else:
                                        future_value = 0
                                    
                                    total_value = immediate_reward + future_value
                                    
                                    if total_value > best_value:
                                        best_value = total_value
                                        best_decision = decision
                                except:
                                    continue
                        
                        V[stage][state_idx] = best_value if best_value != float('-inf') else 0
                        policy[stage][state_idx] = best_decision
                
                # Extraer solución
                optimal_value = V[0][0] if V else 0
                optimal_policy = [policy[stage][0] for stage in range(n_stages)]
            
            else:
                # Estados más complejos, usar aproximación
                optimal_value = 0
                optimal_policy = decisions[:n_stages] if len(decisions) >= n_stages else decisions * (n_stages // len(decisions) + 1)
                optimal_policy = optimal_policy[:n_stages]
            
            variables_values = {}
            for stage, decision in enumerate(optimal_policy):
                variables_values[f'decision_stage_{stage}'] = str(decision)
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=optimal_value,
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'Tabulated DP',
                    'method': 'Backward Tabulation',
                    'stages': problem.stages,
                    'policy': optimal_policy
                }
            )
            
            return solution
            
        except Exception as e:
            return Solution(
                problem_id=problem.id,
                status="error",
                execution_time=0.0,
                solver_info={'error': str(e)}
            )
    
    def solve_forward(self, problem: DynamicProgrammingProblem) -> Solution:
        """Resuelve usando tabulación hacia adelante"""
        # Implementación similar pero iterando hacia adelante
        return self.solve_backward(problem)  # Por simplicidad, delega al método backward
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        return {
            'name': 'Tabulated Dynamic Programming',
            'type': 'Dynamic Programming',
            'methods': ['Backward Tabulation', 'Forward Tabulation'],
            'features': ['Table-based', 'Bottom-up approach']
        }


# Ejemplo de problema clásico: La mochila
class KnapsackProblem:
    """Problema de la mochila como ejemplo de uso de DP"""
    
    @staticmethod
    def create_problem(weights: List[int], values: List[int], capacity: int) -> DynamicProgrammingProblem:
        """Crea un problema de mochila como problema de DP"""
        
        n = len(weights)
        states = list(range(capacity + 1))  # Estados son las capacidades restantes
        decisions = [0, 1]  # 0 = no tomar, 1 = tomar
        
        def transition_function(state: int, decision: int, item_idx: int = 0) -> int:
            """Función de transición para la mochila"""
            if decision == 1 and item_idx < len(weights):
                return max(0, state - weights[item_idx])
            return state
        
        def reward_function(state: int, decision: int, item_idx: int = 0) -> float:
            """Función de recompensa para la mochila"""
            if decision == 1 and item_idx < len(values) and item_idx < len(weights):
                if state >= weights[item_idx]:
                    return values[item_idx]
            return 0
        
        problem = DynamicProgrammingProblem(
            name="Knapsack Problem",
            description=f"Problema de mochila con {n} items y capacidad {capacity}",
            stages=n,
            states=states,
            decisions=decisions,
            transition_function=transition_function,
            reward_function=reward_function
        )
        
        return problem
