import math
import numpy as np
from typing import Dict, Any, List
import time

from ...core.entities.optimization_entities import InventoryProblem, Solution
from ...core.interfaces.repositories import IInventorySolver


class ClassicInventorySolver(IInventorySolver):
    """Implementación de modelos clásicos de inventario"""
    
    def solve(self, problem: InventoryProblem) -> Solution:
        """Resuelve según el tipo de problema de inventario"""
        # Por defecto, resuelve EOQ
        return self.solve_eoq(problem)
    
    def solve_eoq(self, problem: InventoryProblem) -> Solution:
        """Resuelve modelo EOQ (Economic Order Quantity)"""
        try:
            start_time = time.time()
            
            # Calcular demanda anual total
            annual_demand = sum(problem.demand)
            
            if annual_demand <= 0:
                return Solution(
                    problem_id=problem.id,
                    status="infeasible",
                    execution_time=time.time() - start_time,
                    solver_info={'error': 'Demanda debe ser positiva'}
                )
            
            # Fórmula EOQ: Q* = sqrt(2 * D * S / H)
            # D = demanda anual, S = costo de pedido, H = costo de mantenimiento
            eoq = math.sqrt(2 * annual_demand * problem.ordering_cost / problem.holding_cost)
            
            # Calcular métricas asociadas
            number_of_orders = annual_demand / eoq
            cycle_time = eoq / annual_demand * 365  # días
            total_ordering_cost = number_of_orders * problem.ordering_cost
            total_holding_cost = (eoq / 2) * problem.holding_cost
            total_cost = total_ordering_cost + total_holding_cost
            
            variables_values = {
                'optimal_order_quantity': eoq,
                'number_of_orders_per_year': number_of_orders,
                'cycle_time_days': cycle_time,
                'total_ordering_cost': total_ordering_cost,
                'total_holding_cost': total_holding_cost,
                'average_inventory': eoq / 2
            }
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=total_cost,
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'EOQ Model',
                    'model': 'Economic Order Quantity',
                    'assumptions': [
                        'Demanda constante',
                        'Lead time cero',
                        'No hay descuentos por cantidad',
                        'No hay faltantes'
                    ]
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
    
    def solve_dynamic_lot_sizing(self, problem: InventoryProblem) -> Solution:
        """Resuelve problema de lotificación dinámica usando programación dinámica"""
        try:
            start_time = time.time()
            
            n = problem.periods
            demand = problem.demand
            
            if len(demand) != n:
                return Solution(
                    problem_id=problem.id,
                    status="infeasible",
                    execution_time=time.time() - start_time,
                    solver_info={'error': 'Número de períodos no coincide con demanda'}
                )
            
            # Programación dinámica para lot sizing
            # f(t) = costo mínimo desde período t hasta n
            f = [float('inf')] * (n + 1)
            f[n] = 0  # Costo final es 0
            
            # Decisiones óptimas
            decisions = [0] * n
            
            # Hacia atrás desde período n-1 hasta 0
            for t in range(n - 1, -1, -1):
                for k in range(t, n):  # k es hasta qué período satisfacer demanda
                    # Demanda acumulada desde t hasta k
                    total_demand = sum(demand[t:k+1])
                    
                    # Costo de producir total_demand en período t
                    if total_demand > 0:
                        production_cost = problem.ordering_cost + total_demand * problem.unit_cost
                    else:
                        production_cost = 0
                    
                    # Costo de mantener inventario
                    holding_cost = 0
                    cumulative_demand = 0
                    for j in range(t, k + 1):
                        cumulative_demand += demand[j]
                        remaining_inventory = total_demand - cumulative_demand
                        if remaining_inventory > 0:
                            holding_cost += remaining_inventory * problem.holding_cost
                    
                    total_cost = production_cost + holding_cost + f[k + 1]
                    
                    if total_cost < f[t]:
                        f[t] = total_cost
                        decisions[t] = k
            
            # Reconstruir solución
            production_schedule = [0] * n
            inventory_levels = [0] * (n + 1)
            inventory_levels[0] = problem.initial_inventory
            
            t = 0
            while t < n:
                k = decisions[t]
                total_production = sum(demand[t:k+1])
                production_schedule[t] = total_production
                
                # Actualizar niveles de inventario
                for j in range(t, k + 1):
                    if j == t:
                        inventory_levels[j + 1] = inventory_levels[j] + production_schedule[j] - demand[j]
                    else:
                        inventory_levels[j + 1] = inventory_levels[j] - demand[j]
                
                t = k + 1
            
            variables_values = {}
            for i in range(n):
                variables_values[f'production_period_{i+1}'] = production_schedule[i]
                variables_values[f'inventory_end_period_{i+1}'] = inventory_levels[i + 1]
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=f[0],
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'Dynamic Lot Sizing',
                    'model': 'Wagner-Whitin',
                    'method': 'Dynamic Programming'
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
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        return {
            'name': 'Classic Inventory Solver',
            'type': 'Inventory Management',
            'models': ['EOQ', 'Dynamic Lot Sizing'],
            'methods': ['Analytical', 'Dynamic Programming']
        }


class StochasticInventorySolver(IInventorySolver):
    """Implementación de modelos de inventario estocásticos"""
    
    def solve(self, problem: InventoryProblem) -> Solution:
        """Resuelve modelos estocásticos de inventario"""
        return self.solve_safety_stock_model(problem)
    
    def solve_eoq(self, problem: InventoryProblem) -> Solution:
        """EOQ con demanda estocástica"""
        # Delega al solucionador clásico y agrega stock de seguridad
        classic_solver = ClassicInventorySolver()
        base_solution = classic_solver.solve_eoq(problem)
        
        if base_solution.status != "optimal":
            return base_solution
        
        # Agregar análisis de stock de seguridad
        # Asumimos desviación estándar del 20% de la demanda promedio
        annual_demand = sum(problem.demand)
        demand_std = annual_demand * 0.2
        
        # Stock de seguridad para nivel de servicio del 95% (z = 1.645)
        z_score = 1.645
        safety_stock = z_score * math.sqrt(annual_demand / 365) * demand_std / math.sqrt(annual_demand)
        
        # Actualizar variables
        base_solution.variables_values['safety_stock'] = safety_stock
        base_solution.variables_values['reorder_point'] = safety_stock
        base_solution.variables_values['average_inventory'] += safety_stock
        
        # Actualizar costo total
        additional_holding_cost = safety_stock * problem.holding_cost
        base_solution.objective_value += additional_holding_cost
        base_solution.variables_values['total_holding_cost'] += additional_holding_cost
        
        base_solution.solver_info['model'] = 'EOQ with Safety Stock'
        base_solution.solver_info['service_level'] = 0.95
        
        return base_solution
    
    def solve_safety_stock_model(self, problem: InventoryProblem) -> Solution:
        """Modelo específico para cálculo de stock de seguridad"""
        try:
            start_time = time.time()
            
            annual_demand = sum(problem.demand)
            
            # Diferentes niveles de servicio
            service_levels = [0.90, 0.95, 0.99]
            z_scores = [1.282, 1.645, 2.326]
            
            # Estimación de variabilidad de demanda
            demand_mean = annual_demand / len(problem.demand)
            demand_variance = sum((d - demand_mean) ** 2 for d in problem.demand) / len(problem.demand)
            demand_std = math.sqrt(demand_variance)
            
            # Lead time asumido (días)
            lead_time = 7
            
            variables_values = {}
            total_costs = {}
            
            for i, (service_level, z_score) in enumerate(zip(service_levels, z_scores)):
                safety_stock = z_score * demand_std * math.sqrt(lead_time / 365)
                holding_cost = safety_stock * problem.holding_cost
                
                variables_values[f'safety_stock_service_{int(service_level*100)}'] = safety_stock
                variables_values[f'holding_cost_service_{int(service_level*100)}'] = holding_cost
                total_costs[service_level] = holding_cost
            
            # Elegir el nivel de servicio más económico considerando costos de faltante
            # Por simplicidad, asumimos 95% como óptimo
            optimal_service_level = 0.95
            optimal_safety_stock = variables_values['safety_stock_service_95']
            
            variables_values['optimal_safety_stock'] = optimal_safety_stock
            variables_values['optimal_service_level'] = optimal_service_level
            variables_values['reorder_point'] = optimal_safety_stock + demand_mean * (lead_time / 365)
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=total_costs[optimal_service_level],
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'Stochastic Inventory Model',
                    'model': 'Safety Stock',
                    'lead_time_days': lead_time,
                    'demand_std': demand_std
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
    
    def solve_dynamic_lot_sizing(self, problem: InventoryProblem) -> Solution:
        """Delega al solucionador clásico"""
        classic_solver = ClassicInventorySolver()
        return classic_solver.solve_dynamic_lot_sizing(problem)
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        return {
            'name': 'Stochastic Inventory Solver',
            'type': 'Inventory Management',
            'models': ['Safety Stock', 'Stochastic EOQ'],
            'features': ['Uncertainty handling', 'Service levels']
        }
