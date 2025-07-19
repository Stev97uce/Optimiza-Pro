from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..entities.optimization_entities import (
    OptimizationProblem, 
    Solution, 
    LinearProgrammingProblem,
    TransportationProblem,
    NetworkProblem,
    InventoryProblem,
    DynamicProgrammingProblem
)


class ISolver(ABC):
    """Interfaz para solucionadores de problemas de optimización"""
    
    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> Solution:
        """Resuelve un problema de optimización"""
        pass
    
    @abstractmethod
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        pass


class ILinearProgrammingSolver(ISolver):
    """Interfaz específica para solucionadores de programación lineal"""
    
    @abstractmethod
    def solve_with_sensitivity(self, problem: LinearProgrammingProblem) -> Solution:
        """Resuelve con análisis de sensibilidad incluido"""
        pass


class ITransportationSolver(ISolver):
    """Interfaz específica para solucionadores de problemas de transporte"""
    
    @abstractmethod
    def solve_transportation(self, problem: TransportationProblem) -> Solution:
        """Resuelve problema de transporte específico"""
        pass


class INetworkSolver(ISolver):
    """Interfaz específica para solucionadores de problemas de redes"""
    
    @abstractmethod
    def solve_max_flow(self, problem: NetworkProblem) -> Solution:
        """Resuelve problema de flujo máximo"""
        pass
    
    @abstractmethod
    def solve_min_cost_flow(self, problem: NetworkProblem) -> Solution:
        """Resuelve problema de flujo de costo mínimo"""
        pass
    
    @abstractmethod
    def solve_shortest_path(self, problem: NetworkProblem, source: str, target: str) -> Solution:
        """Resuelve problema de camino más corto"""
        pass


class IInventorySolver(ISolver):
    """Interfaz específica para solucionadores de problemas de inventario"""
    
    @abstractmethod
    def solve_eoq(self, problem: InventoryProblem) -> Solution:
        """Resuelve modelo EOQ (Economic Order Quantity)"""
        pass
    
    @abstractmethod
    def solve_dynamic_lot_sizing(self, problem: InventoryProblem) -> Solution:
        """Resuelve problema de lotificación dinámica"""
        pass


class IDynamicProgrammingSolver(ISolver):
    """Interfaz específica para solucionadores de programación dinámica"""
    
    @abstractmethod
    def solve_backward(self, problem: DynamicProgrammingProblem) -> Solution:
        """Resuelve usando recursión hacia atrás"""
        pass
    
    @abstractmethod
    def solve_forward(self, problem: DynamicProgrammingProblem) -> Solution:
        """Resuelve usando recursión hacia adelante"""
        pass


class IProblemRepository(ABC):
    """Interfaz para repositorio de problemas"""
    
    @abstractmethod
    def save(self, problem: OptimizationProblem) -> str:
        """Guarda un problema y retorna su ID"""
        pass
    
    @abstractmethod
    def get_by_id(self, problem_id: str) -> Optional[OptimizationProblem]:
        """Obtiene un problema por su ID"""
        pass
    
    @abstractmethod
    def get_all(self) -> List[OptimizationProblem]:
        """Obtiene todos los problemas"""
        pass
    
    @abstractmethod
    def delete(self, problem_id: str) -> bool:
        """Elimina un problema"""
        pass


class ISolutionRepository(ABC):
    """Interfaz para repositorio de soluciones"""
    
    @abstractmethod
    def save(self, solution: Solution) -> str:
        """Guarda una solución y retorna su ID"""
        pass
    
    @abstractmethod
    def get_by_id(self, solution_id: str) -> Optional[Solution]:
        """Obtiene una solución por su ID"""
        pass
    
    @abstractmethod
    def get_by_problem_id(self, problem_id: str) -> List[Solution]:
        """Obtiene todas las soluciones de un problema"""
        pass
    
    @abstractmethod
    def get_all(self) -> List[Solution]:
        """Obtiene todas las soluciones"""
        pass
    
    @abstractmethod
    def delete(self, solution_id: str) -> bool:
        """Elimina una solución"""
        pass


class ISensitivityAnalyzer(ABC):
    """Interfaz para análisis de sensibilidad"""
    
    @abstractmethod
    def analyze_objective_coefficients(self, problem: OptimizationProblem, solution: Solution) -> Dict[str, Any]:
        """Analiza sensibilidad de coeficientes de función objetivo"""
        pass
    
    @abstractmethod
    def analyze_rhs_values(self, problem: OptimizationProblem, solution: Solution) -> Dict[str, Any]:
        """Analiza sensibilidad de valores del lado derecho"""
        pass
    
    @abstractmethod
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        pass


class IVisualizationService(ABC):
    """Interfaz para servicios de visualización"""
    
    @abstractmethod
    def plot_solution(self, problem: OptimizationProblem, solution: Solution) -> Dict[str, Any]:
        """Genera gráficos de la solución"""
        pass
    
    @abstractmethod
    def plot_sensitivity_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera gráficos del análisis de sensibilidad"""
        pass
    
    @abstractmethod
    def generate_report(self, problem: OptimizationProblem, solution: Solution) -> str:
        """Genera reporte completo"""
        pass


class IAIAnalysisService(ABC):
    """Interfaz para servicios de análisis con IA"""
    
    @abstractmethod
    def predict_optimal_parameters(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predice parámetros óptimos usando IA"""
        pass
    
    @abstractmethod
    def detect_patterns(self, solutions: List[Solution]) -> Dict[str, Any]:
        """Detecta patrones en las soluciones"""
        pass
    
    @abstractmethod
    def recommend_improvements(self, problem: OptimizationProblem, solution: Solution) -> List[str]:
        """Recomienda mejoras usando técnicas de IA"""
        pass


class INotificationService(ABC):
    """Interfaz para servicios de notificación"""
    
    @abstractmethod
    def send_solution_notification(self, solution: Solution) -> bool:
        """Envía notificación cuando se completa una solución"""
        pass
    
    @abstractmethod
    def send_error_notification(self, error: str) -> bool:
        """Envía notificación de error"""
        pass
