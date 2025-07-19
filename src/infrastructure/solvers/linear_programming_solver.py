import pulp
import numpy as np
from typing import Dict, Any
import time

from ...core.entities.optimization_entities import LinearProgrammingProblem, Solution
from ...core.interfaces.repositories import ILinearProgrammingSolver


class PulpLinearProgrammingSolver(ILinearProgrammingSolver):
    """Implementación de solucionador de PL usando PuLP"""
    
    def __init__(self, solver_name='PULP_CBC_CMD'):
        self.solver_name = solver_name
        self.solver = pulp.getSolver(solver_name)
    
    def solve(self, problem: LinearProgrammingProblem) -> Solution:
        """Resuelve el problema de programación lineal"""
        try:
            # Crear el problema en PuLP
            if problem.objective.sense == "minimize":
                lp_problem = pulp.LpProblem(problem.name, pulp.LpMinimize)
            else:
                lp_problem = pulp.LpProblem(problem.name, pulp.LpMaximize)
            
            # Crear variables
            variables = {}
            for var in problem.variables:
                if var.var_type == "binary":
                    variables[var.name] = pulp.LpVariable(
                        var.name, 
                        lowBound=var.lower_bound,
                        upBound=var.upper_bound,
                        cat='Binary'
                    )
                elif var.var_type == "integer":
                    variables[var.name] = pulp.LpVariable(
                        var.name,
                        lowBound=var.lower_bound,
                        upBound=var.upper_bound,
                        cat='Integer'
                    )
                else:
                    variables[var.name] = pulp.LpVariable(
                        var.name,
                        lowBound=var.lower_bound,
                        upBound=var.upper_bound,
                        cat='Continuous'
                    )
            
            # Función objetivo
            objective_expr = pulp.lpSum([
                coef * variables[var_name] 
                for var_name, coef in problem.objective.variables.items()
                if var_name in variables
            ])
            lp_problem += objective_expr
            
            # Restricciones
            for constraint in problem.constraints:
                constraint_expr = pulp.lpSum([
                    coef * variables[var_name]
                    for var_name, coef in constraint.variables.items()
                    if var_name in variables
                ])
                
                if constraint.operator == "<=":
                    lp_problem += constraint_expr <= constraint.rhs
                elif constraint.operator == ">=":
                    lp_problem += constraint_expr >= constraint.rhs
                elif constraint.operator == "=":
                    lp_problem += constraint_expr == constraint.rhs
            
            # Resolver
            start_time = time.time()
            lp_problem.solve(self.solver)
            execution_time = time.time() - start_time
            
            # Crear solución
            status_map = {
                pulp.LpStatusOptimal: "optimal",
                pulp.LpStatusInfeasible: "infeasible",
                pulp.LpStatusUnbounded: "unbounded",
                pulp.LpStatusNotSolved: "not_solved",
                pulp.LpStatusUndefined: "undefined"
            }
            
            status = status_map.get(lp_problem.status, "unknown")
            
            variables_values = {}
            objective_value = None
            
            if status == "optimal":
                for var_name, var in variables.items():
                    variables_values[var_name] = var.varValue
                objective_value = pulp.value(lp_problem.objective)
            
            solution = Solution(
                problem_id=problem.id,
                status=status,
                objective_value=objective_value,
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': self.solver_name,
                    'solver_status': lp_problem.status
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
    
    def solve_with_sensitivity(self, problem: LinearProgrammingProblem) -> Solution:
        """Resuelve con análisis de sensibilidad"""
        solution = self.solve(problem)
        
        if solution.status == "optimal":
            # Análisis de sensibilidad básico
            sensitivity_analysis = self._perform_sensitivity_analysis(problem, solution)
            solution.sensitivity_analysis = sensitivity_analysis
        
        return solution
    
    def _perform_sensitivity_analysis(self, problem: LinearProgrammingProblem, solution: Solution) -> Dict[str, Any]:
        """Realiza análisis de sensibilidad básico"""
        analysis = {
            'shadow_prices': {},
            'reduced_costs': {},
            'ranges': {}
        }
        
        # Aquí se implementaría un análisis de sensibilidad más detallado
        # Por ahora, retornamos estructura básica
        
        return analysis
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        return {
            'name': self.solver_name,
            'type': 'Linear Programming',
            'library': 'PuLP',
            'supports_integer': True,
            'supports_binary': True
        }


class ScipyLinearProgrammingSolver(ILinearProgrammingSolver):
    """Implementación alternativa usando SciPy"""
    
    def __init__(self):
        pass
    
    def solve(self, problem: LinearProgrammingProblem) -> Solution:
        """Resuelve usando scipy.optimize.linprog"""
        try:
            from scipy.optimize import linprog
            
            # Convertir problema a formato estándar de scipy
            # scipy.linprog minimiza c^T * x sujeto a A_ub * x <= b_ub, A_eq * x = b_eq
            
            # Construir vectores y matrices
            var_names = [var.name for var in problem.variables]
            n_vars = len(var_names)
            
            # Función objetivo
            if problem.objective.sense == "maximize":
                c = [-problem.objective.variables.get(name, 0) for name in var_names]
            else:
                c = [problem.objective.variables.get(name, 0) for name in var_names]
            
            # Separar restricciones de igualdad y desigualdad
            A_ub, b_ub = [], []
            A_eq, b_eq = [], []
            
            for constraint in problem.constraints:
                row = [constraint.variables.get(name, 0) for name in var_names]
                
                if constraint.operator == "<=":
                    A_ub.append(row)
                    b_ub.append(constraint.rhs)
                elif constraint.operator == ">=":
                    A_ub.append([-x for x in row])
                    b_ub.append(-constraint.rhs)
                elif constraint.operator == "=":
                    A_eq.append(row)
                    b_eq.append(constraint.rhs)
            
            # Límites de variables
            bounds = []
            for var in problem.variables:
                bounds.append((var.lower_bound, var.upper_bound))
            
            # Resolver
            start_time = time.time()
            
            result = linprog(
                c=c,
                A_ub=A_ub if A_ub else None,
                b_ub=b_ub if b_ub else None,
                A_eq=A_eq if A_eq else None,
                b_eq=b_eq if b_eq else None,
                bounds=bounds,
                method='highs'
            )
            
            execution_time = time.time() - start_time
            
            # Convertir resultado
            status_map = {
                0: "optimal",
                1: "iteration_limit",
                2: "infeasible",
                3: "unbounded",
                4: "numerical_error"
            }
            
            status = status_map.get(result.status, "unknown")
            
            variables_values = {}
            objective_value = None
            
            if status == "optimal":
                for i, name in enumerate(var_names):
                    variables_values[name] = result.x[i]
                
                if problem.objective.sense == "maximize":
                    objective_value = -result.fun
                else:
                    objective_value = result.fun
            
            solution = Solution(
                problem_id=problem.id,
                status=status,
                objective_value=objective_value,
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'SciPy',
                    'method': 'HiGHS',
                    'iterations': result.nit if hasattr(result, 'nit') else 0
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
    
    def solve_with_sensitivity(self, problem: LinearProgrammingProblem) -> Solution:
        """Resuelve con análisis de sensibilidad"""
        return self.solve(problem)  # SciPy no incluye análisis de sensibilidad directo
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        return {
            'name': 'SciPy',
            'type': 'Linear Programming',
            'library': 'SciPy',
            'supports_integer': False,
            'supports_binary': False
        }
