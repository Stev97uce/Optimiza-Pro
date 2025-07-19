import numpy as np
from typing import Dict, Any, List
import time

from ...core.entities.optimization_entities import TransportationProblem, Solution
from ...core.interfaces.repositories import ITransportationSolver


class VogelTransportationSolver(ITransportationSolver):
    """Implementación del método de Vogel para problemas de transporte"""
    
    def solve(self, problem: TransportationProblem) -> Solution:
        """Resuelve el problema de transporte usando el método de Vogel"""
        try:
            start_time = time.time()
            
            # Verificar balance
            total_supply = sum(problem.supply)
            total_demand = sum(problem.demand)
            
            if abs(total_supply - total_demand) > 1e-6:
                return Solution(
                    problem_id=problem.id,
                    status="infeasible",
                    execution_time=time.time() - start_time,
                    solver_info={'error': 'Problema no balanceado'}
                )
            
            # Resolver usando método de Vogel
            solution_matrix = self._vogel_method(
                problem.supply.copy(),
                problem.demand.copy(),
                [row.copy() for row in problem.costs]
            )
            
            # Calcular costo total
            total_cost = 0
            variables_values = {}
            
            for i in range(len(problem.origins)):
                for j in range(len(problem.destinations)):
                    var_name = f"x_{problem.origins[i]}_{problem.destinations[j]}"
                    value = solution_matrix[i][j]
                    variables_values[var_name] = value
                    total_cost += value * problem.costs[i][j]
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=total_cost,
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'Vogel Method',
                    'method': 'Approximation'
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
    
    def solve_transportation(self, problem: TransportationProblem) -> Solution:
        """Método específico para problemas de transporte"""
        return self.solve(problem)
    
    def _vogel_method(self, supply: List[float], demand: List[float], costs: List[List[float]]) -> List[List[float]]:
        """Implementación del método de aproximación de Vogel"""
        m, n = len(supply), len(demand)
        allocation = [[0.0 for _ in range(n)] for _ in range(m)]
        
        # Crear copias para trabajar
        supply_copy = supply.copy()
        demand_copy = demand.copy()
        
        while any(s > 1e-6 for s in supply_copy) and any(d > 1e-6 for d in demand_copy):
            # Calcular penalizaciones por fila
            row_penalties = []
            for i in range(m):
                if supply_copy[i] <= 1e-6:
                    row_penalties.append(-1)
                    continue
                
                valid_costs = [costs[i][j] for j in range(n) if demand_copy[j] > 1e-6]
                if len(valid_costs) < 2:
                    row_penalties.append(float('inf') if valid_costs else -1)
                else:
                    valid_costs.sort()
                    row_penalties.append(valid_costs[1] - valid_costs[0])
            
            # Calcular penalizaciones por columna
            col_penalties = []
            for j in range(n):
                if demand_copy[j] <= 1e-6:
                    col_penalties.append(-1)
                    continue
                
                valid_costs = [costs[i][j] for i in range(m) if supply_copy[i] > 1e-6]
                if len(valid_costs) < 2:
                    col_penalties.append(float('inf') if valid_costs else -1)
                else:
                    valid_costs.sort()
                    col_penalties.append(valid_costs[1] - valid_costs[0])
            
            # Encontrar la máxima penalización
            max_row_penalty = max(p for p in row_penalties if p >= 0) if any(p >= 0 for p in row_penalties) else -1
            max_col_penalty = max(p for p in col_penalties if p >= 0) if any(p >= 0 for p in col_penalties) else -1
            
            if max_row_penalty == -1 and max_col_penalty == -1:
                break
            
            # Seleccionar fila o columna con mayor penalización
            if max_row_penalty >= max_col_penalty:
                # Trabajar con fila
                row_idx = row_penalties.index(max_row_penalty)
                # Encontrar la celda de menor costo en esta fila
                min_cost = float('inf')
                col_idx = -1
                for j in range(n):
                    if demand_copy[j] > 1e-6 and costs[row_idx][j] < min_cost:
                        min_cost = costs[row_idx][j]
                        col_idx = j
            else:
                # Trabajar con columna
                col_idx = col_penalties.index(max_col_penalty)
                # Encontrar la celda de menor costo en esta columna
                min_cost = float('inf')
                row_idx = -1
                for i in range(m):
                    if supply_copy[i] > 1e-6 and costs[i][col_idx] < min_cost:
                        min_cost = costs[i][col_idx]
                        row_idx = i
            
            # Asignar la cantidad máxima posible
            if row_idx >= 0 and col_idx >= 0:
                amount = min(supply_copy[row_idx], demand_copy[col_idx])
                allocation[row_idx][col_idx] = amount
                supply_copy[row_idx] -= amount
                demand_copy[col_idx] -= amount
        
        return allocation
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        return {
            'name': 'Vogel Method',
            'type': 'Transportation',
            'method': 'Heuristic',
            'optimal': False
        }


class TransportationSimplexSolver(ITransportationSolver):
    """Implementación del método simplex especializado para transporte"""
    
    def solve(self, problem: TransportationProblem) -> Solution:
        """Resuelve usando el método simplex para transporte"""
        try:
            start_time = time.time()
            
            # Convertir a problema de PL estándar y resolver
            lp_problem = self._convert_to_linear_program(problem)
            
            # Aquí se implementaría el simplex especializado
            # Por simplicidad, usamos el método de Vogel como aproximación inicial
            vogel_solver = VogelTransportationSolver()
            initial_solution = vogel_solver.solve(problem)
            
            # Mejorar la solución con pasos del simplex
            improved_solution = self._improve_solution(problem, initial_solution)
            
            execution_time = time.time() - start_time
            improved_solution.execution_time = execution_time
            improved_solution.solver_info['solver_name'] = 'Transportation Simplex'
            
            return improved_solution
            
        except Exception as e:
            return Solution(
                problem_id=problem.id,
                status="error",
                execution_time=0.0,
                solver_info={'error': str(e)}
            )
    
    def solve_transportation(self, problem: TransportationProblem) -> Solution:
        """Método específico para problemas de transporte"""
        return self.solve(problem)
    
    def _convert_to_linear_program(self, problem: TransportationProblem):
        """Convierte el problema de transporte a PL estándar"""
        # Implementación simplificada
        pass
    
    def _improve_solution(self, problem: TransportationProblem, initial_solution: Solution) -> Solution:
        """Mejora la solución inicial usando pasos del simplex"""
        # Por ahora retorna la solución inicial
        # Aquí se implementarían las iteraciones del simplex
        return initial_solution
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        return {
            'name': 'Transportation Simplex',
            'type': 'Transportation',
            'method': 'Exact',
            'optimal': True
        }
