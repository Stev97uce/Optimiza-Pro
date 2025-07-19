from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import uuid


@dataclass
class Variable:
    """Representa una variable de decisión en problemas de optimización"""
    name: str
    lower_bound: float = 0.0
    upper_bound: Optional[float] = None
    var_type: str = "continuous"  # continuous, integer, binary
    coefficient: float = 0.0
    
    def __post_init__(self):
        if not hasattr(self, 'id'):
            self.id = str(uuid.uuid4())


@dataclass 
class Constraint:
    """Representa una restricción en problemas de optimización"""
    name: str
    variables: Dict[str, float]  # variable_name: coefficient
    operator: str  # <=, >=, =
    rhs: float
    
    def __post_init__(self):
        if not hasattr(self, 'id'):
            self.id = str(uuid.uuid4())


@dataclass
class ObjectiveFunction:
    """Representa la función objetivo"""
    variables: Dict[str, float]  # variable_name: coefficient
    sense: str = "minimize"  # minimize, maximize
    name: str = "objective"


@dataclass
class OptimizationProblem(ABC):
    """Clase base para todos los problemas de optimización"""
    name: str
    description: str
    variables: List[Variable] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    objective: Optional[ObjectiveFunction] = None
    
    def __post_init__(self):
        if not hasattr(self, 'id'):
            self.id = str(uuid.uuid4())
    
    @abstractmethod
    def validate(self) -> bool:
        """Valida que el problema esté bien definido"""
        pass


@dataclass
class Solution:
    """Representa la solución de un problema de optimización"""
    problem_id: str
    status: str  # optimal, infeasible, unbounded, etc.
    objective_value: Optional[float] = None
    variables_values: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    solver_info: Dict[str, Any] = field(default_factory=dict)
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, 'id'):
            self.id = str(uuid.uuid4())


@dataclass
class LinearProgrammingProblem(OptimizationProblem):
    """Problema de Programación Lineal específico"""
    
    def validate(self) -> bool:
        """Valida que sea un problema de PL válido"""
        if not self.objective:
            return False
        if not self.variables:
            return False
        if not self.constraints:
            return False
        
        # Verificar que todas las variables en restricciones existan
        var_names = {var.name for var in self.variables}
        for constraint in self.constraints:
            if not all(var_name in var_names for var_name in constraint.variables.keys()):
                return False
        
        return True


@dataclass
class TransportationProblem(OptimizationProblem):
    """Problema de Transporte específico"""
    supply: List[float] = field(default_factory=list)
    demand: List[float] = field(default_factory=list)
    costs: List[List[float]] = field(default_factory=list)
    origins: List[str] = field(default_factory=list)
    destinations: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Valida que sea un problema de transporte válido"""
        if len(self.supply) != len(self.origins):
            return False
        if len(self.demand) != len(self.destinations):
            return False
        if len(self.costs) != len(self.origins):
            return False
        for cost_row in self.costs:
            if len(cost_row) != len(self.destinations):
                return False
        
        # Verificar balance de oferta y demanda
        total_supply = sum(self.supply)
        total_demand = sum(self.demand)
        
        return abs(total_supply - total_demand) < 1e-6


@dataclass
class NetworkProblem(OptimizationProblem):
    """Problema de Redes específico"""
    nodes: List[str] = field(default_factory=list)
    edges: List[tuple] = field(default_factory=list)  # (from, to, capacity, cost)
    node_demands: Dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Valida que sea un problema de redes válido"""
        if not self.nodes:
            return False
        
        # Verificar que todos los nodos en edges existan
        for edge in self.edges:
            if edge[0] not in self.nodes or edge[1] not in self.nodes:
                return False
        
        # Verificar balance de flujo
        total_demand = sum(self.node_demands.values())
        return abs(total_demand) < 1e-6


@dataclass
class InventoryProblem(OptimizationProblem):
    """Problema de Inventario específico"""
    periods: int = 12
    demand: List[float] = field(default_factory=list)
    holding_cost: float = 0.0
    ordering_cost: float = 0.0
    unit_cost: float = 0.0
    initial_inventory: float = 0.0
    capacity: float = float('inf')
    
    def validate(self) -> bool:
        """Valida que sea un problema de inventario válido"""
        if self.periods <= 0:
            return False
        if len(self.demand) != self.periods:
            return False
        if any(d < 0 for d in self.demand):
            return False
        if self.holding_cost < 0 or self.ordering_cost < 0:
            return False
        
        return True


@dataclass
class DynamicProgrammingProblem(OptimizationProblem):
    """Problema de Programación Dinámica específico"""
    stages: int = 0
    states: List[Any] = field(default_factory=list)
    decisions: List[Any] = field(default_factory=list)
    transition_function: Optional[callable] = None
    reward_function: Optional[callable] = None
    
    def validate(self) -> bool:
        """Valida que sea un problema de PD válido"""
        if self.stages <= 0:
            return False
        if not self.states:
            return False
        if not self.decisions:
            return False
        if not self.transition_function or not self.reward_function:
            return False
        
        return True
