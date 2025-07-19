import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import pickle

from ...core.entities.optimization_entities import OptimizationProblem, Solution
from ...core.interfaces.repositories import IProblemRepository, ISolutionRepository


class FileBasedProblemRepository(IProblemRepository):
    """Repositorio de problemas basado en archivos"""
    
    def __init__(self, storage_path: str = "data/problems"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save(self, problem: OptimizationProblem) -> str:
        """Guarda un problema y retorna su ID"""
        try:
            problem_data = {
                'id': problem.id,
                'name': problem.name,
                'description': problem.description,
                'type': type(problem).__name__,
                'created_at': datetime.now().isoformat(),
                'data': self._serialize_problem(problem)
            }
            
            file_path = os.path.join(self.storage_path, f"{problem.id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(problem_data, f, indent=2, ensure_ascii=False)
            
            # También guardar el objeto completo para preservar métodos
            pickle_path = os.path.join(self.storage_path, f"{problem.id}.pickle")
            with open(pickle_path, 'wb') as f:
                pickle.dump(problem, f)
            
            return problem.id
            
        except Exception as e:
            raise Exception(f"Error al guardar problema: {str(e)}")
    
    def get_by_id(self, problem_id: str) -> Optional[OptimizationProblem]:
        """Obtiene un problema por su ID"""
        try:
            # Intentar cargar desde pickle primero
            pickle_path = os.path.join(self.storage_path, f"{problem_id}.pickle")
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            
            # Fallback a JSON
            file_path = os.path.join(self.storage_path, f"{problem_id}.json")
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                problem_data = json.load(f)
            
            return self._deserialize_problem(problem_data)
            
        except Exception as e:
            print(f"Error al cargar problema {problem_id}: {str(e)}")
            return None
    
    def get_all(self) -> List[OptimizationProblem]:
        """Obtiene todos los problemas"""
        problems = []
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    problem_id = filename[:-5]  # Remover .json
                    problem = self.get_by_id(problem_id)
                    if problem:
                        problems.append(problem)
        except Exception as e:
            print(f"Error al cargar problemas: {str(e)}")
        
        return problems
    
    def delete(self, problem_id: str) -> bool:
        """Elimina un problema"""
        try:
            file_path = os.path.join(self.storage_path, f"{problem_id}.json")
            pickle_path = os.path.join(self.storage_path, f"{problem_id}.pickle")
            
            deleted = False
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted = True
            if os.path.exists(pickle_path):
                os.remove(pickle_path)
                deleted = True
            
            return deleted
            
        except Exception as e:
            print(f"Error al eliminar problema {problem_id}: {str(e)}")
            return False
    
    def _serialize_problem(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """Serializa un problema a diccionario"""
        data = {
            'variables': [self._serialize_variable(var) for var in problem.variables],
            'constraints': [self._serialize_constraint(const) for const in problem.constraints],
            'objective': self._serialize_objective(problem.objective) if problem.objective else None
        }
        
        # Agregar campos específicos según el tipo
        if hasattr(problem, 'supply'):
            data['supply'] = problem.supply
        if hasattr(problem, 'demand'):
            data['demand'] = problem.demand
        if hasattr(problem, 'costs'):
            data['costs'] = problem.costs
        if hasattr(problem, 'origins'):
            data['origins'] = problem.origins
        if hasattr(problem, 'destinations'):
            data['destinations'] = problem.destinations
        if hasattr(problem, 'nodes'):
            data['nodes'] = problem.nodes
        if hasattr(problem, 'edges'):
            data['edges'] = problem.edges
        if hasattr(problem, 'node_demands'):
            data['node_demands'] = problem.node_demands
        if hasattr(problem, 'periods'):
            data['periods'] = problem.periods
        if hasattr(problem, 'holding_cost'):
            data['holding_cost'] = problem.holding_cost
        if hasattr(problem, 'ordering_cost'):
            data['ordering_cost'] = problem.ordering_cost
        if hasattr(problem, 'unit_cost'):
            data['unit_cost'] = problem.unit_cost
        if hasattr(problem, 'initial_inventory'):
            data['initial_inventory'] = problem.initial_inventory
        if hasattr(problem, 'capacity'):
            data['capacity'] = problem.capacity
        if hasattr(problem, 'stages'):
            data['stages'] = problem.stages
        if hasattr(problem, 'states'):
            data['states'] = problem.states
        if hasattr(problem, 'decisions'):
            data['decisions'] = problem.decisions
        
        return data
    
    def _serialize_variable(self, variable) -> Dict[str, Any]:
        """Serializa una variable"""
        return {
            'name': variable.name,
            'lower_bound': variable.lower_bound,
            'upper_bound': variable.upper_bound,
            'var_type': variable.var_type,
            'coefficient': variable.coefficient
        }
    
    def _serialize_constraint(self, constraint) -> Dict[str, Any]:
        """Serializa una restricción"""
        return {
            'name': constraint.name,
            'variables': constraint.variables,
            'operator': constraint.operator,
            'rhs': constraint.rhs
        }
    
    def _serialize_objective(self, objective) -> Dict[str, Any]:
        """Serializa función objetivo"""
        return {
            'variables': objective.variables,
            'sense': objective.sense,
            'name': objective.name
        }
    
    def _deserialize_problem(self, problem_data: Dict[str, Any]) -> OptimizationProblem:
        """Deserializa un problema desde diccionario"""
        # Esta es una implementación simplificada
        # En un caso real, necesitaríamos recrear el objeto específico
        from ...core.entities.optimization_entities import LinearProgrammingProblem
        
        problem = LinearProgrammingProblem(
            name=problem_data['name'],
            description=problem_data['description']
        )
        problem.id = problem_data['id']
        
        return problem


class FileBasedSolutionRepository(ISolutionRepository):
    """Repositorio de soluciones basado en archivos"""
    
    def __init__(self, storage_path: str = "data/solutions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save(self, solution: Solution) -> str:
        """Guarda una solución y retorna su ID"""
        try:
            solution_data = {
                'id': solution.id,
                'problem_id': solution.problem_id,
                'status': solution.status,
                'objective_value': solution.objective_value,
                'variables_values': solution.variables_values,
                'execution_time': solution.execution_time,
                'solver_info': solution.solver_info,
                'sensitivity_analysis': solution.sensitivity_analysis,
                'created_at': datetime.now().isoformat()
            }
            
            file_path = os.path.join(self.storage_path, f"{solution.id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(solution_data, f, indent=2, ensure_ascii=False)
            
            return solution.id
            
        except Exception as e:
            raise Exception(f"Error al guardar solución: {str(e)}")
    
    def get_by_id(self, solution_id: str) -> Optional[Solution]:
        """Obtiene una solución por su ID"""
        try:
            file_path = os.path.join(self.storage_path, f"{solution_id}.json")
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                solution_data = json.load(f)
            
            solution = Solution(
                problem_id=solution_data['problem_id'],
                status=solution_data['status'],
                objective_value=solution_data.get('objective_value'),
                variables_values=solution_data.get('variables_values', {}),
                execution_time=solution_data.get('execution_time', 0.0),
                solver_info=solution_data.get('solver_info', {}),
                sensitivity_analysis=solution_data.get('sensitivity_analysis', {})
            )
            solution.id = solution_data['id']
            
            return solution
            
        except Exception as e:
            print(f"Error al cargar solución {solution_id}: {str(e)}")
            return None
    
    def get_by_problem_id(self, problem_id: str) -> List[Solution]:
        """Obtiene todas las soluciones de un problema"""
        solutions = []
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    solution_id = filename[:-5]
                    solution = self.get_by_id(solution_id)
                    if solution and solution.problem_id == problem_id:
                        solutions.append(solution)
        except Exception as e:
            print(f"Error al cargar soluciones del problema {problem_id}: {str(e)}")
        
        return solutions
    
    def get_all(self) -> List[Solution]:
        """Obtiene todas las soluciones"""
        solutions = []
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    solution_id = filename[:-5]
                    solution = self.get_by_id(solution_id)
                    if solution:
                        solutions.append(solution)
        except Exception as e:
            print(f"Error al cargar soluciones: {str(e)}")
        
        return solutions
    
    def delete(self, solution_id: str) -> bool:
        """Elimina una solución"""
        try:
            file_path = os.path.join(self.storage_path, f"{solution_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
            
        except Exception as e:
            print(f"Error al eliminar solución {solution_id}: {str(e)}")
            return False


class InMemoryProblemRepository(IProblemRepository):
    """Repositorio de problemas en memoria para testing"""
    
    def __init__(self):
        self.problems: Dict[str, OptimizationProblem] = {}
    
    def save(self, problem: OptimizationProblem) -> str:
        """Guarda un problema y retorna su ID"""
        self.problems[problem.id] = problem
        return problem.id
    
    def get_by_id(self, problem_id: str) -> Optional[OptimizationProblem]:
        """Obtiene un problema por su ID"""
        return self.problems.get(problem_id)
    
    def get_all(self) -> List[OptimizationProblem]:
        """Obtiene todos los problemas"""
        return list(self.problems.values())
    
    def delete(self, problem_id: str) -> bool:
        """Elimina un problema"""
        if problem_id in self.problems:
            del self.problems[problem_id]
            return True
        return False


class InMemorySolutionRepository(ISolutionRepository):
    """Repositorio de soluciones en memoria para testing"""
    
    def __init__(self):
        self.solutions: Dict[str, Solution] = {}
    
    def save(self, solution: Solution) -> str:
        """Guarda una solución y retorna su ID"""
        self.solutions[solution.id] = solution
        return solution.id
    
    def get_by_id(self, solution_id: str) -> Optional[Solution]:
        """Obtiene una solución por su ID"""
        return self.solutions.get(solution_id)
    
    def get_by_problem_id(self, problem_id: str) -> List[Solution]:
        """Obtiene todas las soluciones de un problema"""
        return [sol for sol in self.solutions.values() if sol.problem_id == problem_id]
    
    def get_all(self) -> List[Solution]:
        """Obtiene todas las soluciones"""
        return list(self.solutions.values())
    
    def delete(self, solution_id: str) -> bool:
        """Elimina una solución"""
        if solution_id in self.solutions:
            del self.solutions[solution_id]
            return True
        return False
