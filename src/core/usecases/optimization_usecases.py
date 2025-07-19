from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from ..entities.optimization_entities import (
    OptimizationProblem, 
    Solution, 
    LinearProgrammingProblem,
    TransportationProblem,
    NetworkProblem,
    InventoryProblem,
    DynamicProgrammingProblem
)
from ..interfaces.repositories import (
    ISolver,
    IProblemRepository,
    ISolutionRepository,
    ISensitivityAnalyzer,
    IAIAnalysisService,
    IVisualizationService,
    INotificationService
)


@dataclass
class SolveOptimizationProblemRequest:
    problem: OptimizationProblem
    include_sensitivity: bool = True
    include_ai_analysis: bool = True


@dataclass
class SolveOptimizationProblemResponse:
    solution: Solution
    success: bool
    message: str
    visualization_data: Optional[Dict[str, Any]] = None
    ai_recommendations: Optional[List[str]] = None


class SolveOptimizationProblemUseCase:
    """Caso de uso principal para resolver problemas de optimización"""
    
    def __init__(
        self,
        solver: ISolver,
        problem_repository: IProblemRepository,
        solution_repository: ISolutionRepository,
        sensitivity_analyzer: Optional[ISensitivityAnalyzer] = None,
        ai_service: Optional[IAIAnalysisService] = None,
        visualization_service: Optional[IVisualizationService] = None,
        notification_service: Optional[INotificationService] = None
    ):
        self.solver = solver
        self.problem_repository = problem_repository
        self.solution_repository = solution_repository
        self.sensitivity_analyzer = sensitivity_analyzer
        self.ai_service = ai_service
        self.visualization_service = visualization_service
        self.notification_service = notification_service
    
    def execute(self, request: SolveOptimizationProblemRequest) -> SolveOptimizationProblemResponse:
        try:
            # Validar el problema
            if not request.problem.validate():
                return SolveOptimizationProblemResponse(
                    solution=None,
                    success=False,
                    message="El problema no es válido"
                )
            
            # Guardar el problema
            problem_id = self.problem_repository.save(request.problem)
            
            # Resolver el problema
            start_time = time.time()
            solution = self.solver.solve(request.problem)
            solution.execution_time = time.time() - start_time
            
            # Análisis de sensibilidad
            if request.include_sensitivity and self.sensitivity_analyzer:
                obj_analysis = self.sensitivity_analyzer.analyze_objective_coefficients(
                    request.problem, solution
                )
                rhs_analysis = self.sensitivity_analyzer.analyze_rhs_values(
                    request.problem, solution
                )
                solution.sensitivity_analysis = {
                    'objective_coefficients': obj_analysis,
                    'rhs_values': rhs_analysis
                }
            
            # Análisis con IA
            ai_recommendations = []
            if request.include_ai_analysis and self.ai_service:
                ai_recommendations = self.ai_service.recommend_improvements(
                    request.problem, solution
                )
            
            # Generar visualizaciones
            visualization_data = None
            if self.visualization_service:
                visualization_data = self.visualization_service.plot_solution(
                    request.problem, solution
                )
            
            # Guardar la solución
            self.solution_repository.save(solution)
            
            # Enviar notificación
            if self.notification_service:
                self.notification_service.send_solution_notification(solution)
            
            return SolveOptimizationProblemResponse(
                solution=solution,
                success=True,
                message="Problema resuelto exitosamente",
                visualization_data=visualization_data,
                ai_recommendations=ai_recommendations
            )
            
        except Exception as e:
            error_msg = f"Error al resolver el problema: {str(e)}"
            if self.notification_service:
                self.notification_service.send_error_notification(error_msg)
            
            return SolveOptimizationProblemResponse(
                solution=None,
                success=False,
                message=error_msg
            )


@dataclass
class GetProblemHistoryRequest:
    problem_type: Optional[str] = None
    limit: Optional[int] = None


@dataclass
class GetProblemHistoryResponse:
    problems: List[OptimizationProblem]
    solutions: Dict[str, List[Solution]]
    success: bool
    message: str


class GetProblemHistoryUseCase:
    """Caso de uso para obtener el historial de problemas"""
    
    def __init__(
        self,
        problem_repository: IProblemRepository,
        solution_repository: ISolutionRepository
    ):
        self.problem_repository = problem_repository
        self.solution_repository = solution_repository
    
    def execute(self, request: GetProblemHistoryRequest) -> GetProblemHistoryResponse:
        try:
            problems = self.problem_repository.get_all()
            
            # Filtrar por tipo si se especifica
            if request.problem_type:
                problems = [p for p in problems if type(p).__name__ == request.problem_type]
            
            # Aplicar límite si se especifica
            if request.limit:
                problems = problems[:request.limit]
            
            # Obtener soluciones para cada problema
            solutions = {}
            for problem in problems:
                solutions[problem.id] = self.solution_repository.get_by_problem_id(problem.id)
            
            return GetProblemHistoryResponse(
                problems=problems,
                solutions=solutions,
                success=True,
                message="Historial obtenido exitosamente"
            )
            
        except Exception as e:
            return GetProblemHistoryResponse(
                problems=[],
                solutions={},
                success=False,
                message=f"Error al obtener historial: {str(e)}"
            )


@dataclass
class GenerateReportRequest:
    problem_id: str
    include_sensitivity: bool = True
    include_visualizations: bool = True


@dataclass
class GenerateReportResponse:
    report_content: str
    report_format: str
    success: bool
    message: str


class GenerateReportUseCase:
    """Caso de uso para generar reportes de análisis"""
    
    def __init__(
        self,
        problem_repository: IProblemRepository,
        solution_repository: ISolutionRepository,
        visualization_service: Optional[IVisualizationService] = None
    ):
        self.problem_repository = problem_repository
        self.solution_repository = solution_repository
        self.visualization_service = visualization_service
    
    def execute(self, request: GenerateReportRequest) -> GenerateReportResponse:
        try:
            problem = self.problem_repository.get_by_id(request.problem_id)
            if not problem:
                return GenerateReportResponse(
                    report_content="",
                    report_format="",
                    success=False,
                    message="Problema no encontrado"
                )
            
            solutions = self.solution_repository.get_by_problem_id(request.problem_id)
            if not solutions:
                return GenerateReportResponse(
                    report_content="",
                    report_format="",
                    success=False,
                    message="No se encontraron soluciones para este problema"
                )
            
            # Usar la solución más reciente
            latest_solution = max(solutions, key=lambda s: s.execution_time)
            
            # Generar reporte
            if self.visualization_service:
                report_content = self.visualization_service.generate_report(
                    problem, latest_solution
                )
            else:
                # Reporte básico si no hay servicio de visualización
                report_content = self._generate_basic_report(problem, latest_solution)
            
            return GenerateReportResponse(
                report_content=report_content,
                report_format="html",
                success=True,
                message="Reporte generado exitosamente"
            )
            
        except Exception as e:
            return GenerateReportResponse(
                report_content="",
                report_format="",
                success=False,
                message=f"Error al generar reporte: {str(e)}"
            )
    
    def _generate_basic_report(self, problem: OptimizationProblem, solution: Solution) -> str:
        """Genera un reporte básico en formato HTML"""
        return f"""
        <h1>Reporte de Optimización</h1>
        <h2>Problema: {problem.name}</h2>
        <p><strong>Descripción:</strong> {problem.description}</p>
        <p><strong>Tipo:</strong> {type(problem).__name__}</p>
        
        <h2>Solución</h2>
        <p><strong>Estado:</strong> {solution.status}</p>
        <p><strong>Valor Objetivo:</strong> {solution.objective_value}</p>
        <p><strong>Tiempo de Ejecución:</strong> {solution.execution_time:.4f} segundos</p>
        
        <h3>Variables de Decisión</h3>
        <ul>
        {''.join([f'<li>{var}: {value}</li>' for var, value in solution.variables_values.items()])}
        </ul>
        """


@dataclass
class CompareAlgorithmsRequest:
    problem: OptimizationProblem
    solvers: List[ISolver]


@dataclass
class CompareAlgorithmsResponse:
    comparisons: Dict[str, Solution]
    best_solver: str
    comparison_metrics: Dict[str, Any]
    success: bool
    message: str


class CompareAlgorithmsUseCase:
    """Caso de uso para comparar diferentes algoritmos de solución"""
    
    def __init__(
        self,
        problem_repository: IProblemRepository,
        solution_repository: ISolutionRepository
    ):
        self.problem_repository = problem_repository
        self.solution_repository = solution_repository
    
    def execute(self, request: CompareAlgorithmsRequest) -> CompareAlgorithmsResponse:
        try:
            if not request.problem.validate():
                return CompareAlgorithmsResponse(
                    comparisons={},
                    best_solver="",
                    comparison_metrics={},
                    success=False,
                    message="El problema no es válido"
                )
            
            comparisons = {}
            metrics = {}
            
            for solver in request.solvers:
                solver_name = solver.__class__.__name__
                
                start_time = time.time()
                solution = solver.solve(request.problem)
                solution.execution_time = time.time() - start_time
                
                comparisons[solver_name] = solution
                
                # Guardar solución
                self.solution_repository.save(solution)
                
                # Métricas de comparación
                metrics[solver_name] = {
                    'execution_time': solution.execution_time,
                    'objective_value': solution.objective_value,
                    'status': solution.status
                }
            
            # Determinar el mejor solucionador
            valid_solutions = {name: sol for name, sol in comparisons.items() 
                             if sol.status == 'optimal'}
            
            if valid_solutions:
                # El mejor es el que tiene menor tiempo de ejecución entre soluciones óptimas
                best_solver = min(valid_solutions.keys(), 
                                key=lambda x: comparisons[x].execution_time)
            else:
                best_solver = ""
            
            return CompareAlgorithmsResponse(
                comparisons=comparisons,
                best_solver=best_solver,
                comparison_metrics=metrics,
                success=True,
                message="Comparación completada exitosamente"
            )
            
        except Exception as e:
            return CompareAlgorithmsResponse(
                comparisons={},
                best_solver="",
                comparison_metrics={},
                success=False,
                message=f"Error en la comparación: {str(e)}"
            )
