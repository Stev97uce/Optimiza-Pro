from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import sys
import os
import json
import traceback

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.entities.optimization_entities import (
    LinearProgrammingProblem, TransportationProblem, NetworkProblem, 
    InventoryProblem, DynamicProgrammingProblem,
    Variable, Constraint, ObjectiveFunction, Solution
)
from src.core.usecases.optimization_usecases import (
    SolveOptimizationProblemUseCase, SolveOptimizationProblemRequest,
    GetProblemHistoryUseCase, GetProblemHistoryRequest,
    GenerateReportUseCase, GenerateReportRequest,
    CompareAlgorithmsUseCase, CompareAlgorithmsRequest
)
from src.infrastructure.repositories.storage_repositories import (
    InMemoryProblemRepository, InMemorySolutionRepository
)
from src.infrastructure.services.ai_service import (
    BasicSensitivityAnalyzer, MachineLearningAIService
)
from src.infrastructure.services.visualization_service import (
    PlotlyVisualizationService
)
from src.infrastructure.services.ai_problem_generator import AIProblemGenerator

# Importar solucionadores (con manejo de errores para dependencias)
try:
    from src.infrastructure.solvers.linear_programming_solver import PulpLinearProgrammingSolver, ScipyLinearProgrammingSolver
except ImportError:
    print("Warning: No se pudieron cargar los solucionadores de PL. Instale las dependencias.")
    PulpLinearProgrammingSolver = None
    ScipyLinearProgrammingSolver = None

try:
    from src.infrastructure.solvers.transportation_solver import VogelTransportationSolver, TransportationSimplexSolver
except ImportError:
    print("Warning: No se pudieron cargar los solucionadores de transporte.")
    VogelTransportationSolver = None
    TransportationSimplexSolver = None

try:
    from src.infrastructure.solvers.network_solver import NetworkXSolver, CustomNetworkSolver
except ImportError:
    print("Warning: No se pudieron cargar los solucionadores de redes.")
    NetworkXSolver = None
    CustomNetworkSolver = None

try:
    from src.infrastructure.solvers.inventory_solver import ClassicInventorySolver, StochasticInventorySolver
except ImportError:
    print("Warning: No se pudieron cargar los solucionadores de inventario.")
    ClassicInventorySolver = None
    StochasticInventorySolver = None

try:
    from src.infrastructure.solvers.dynamic_programming_solver import RecursiveDynamicProgrammingSolver, TabulatedDynamicProgrammingSolver
except ImportError:
    print("Warning: No se pudieron cargar los solucionadores de programación dinámica.")
    RecursiveDynamicProgrammingSolver = None
    TabulatedDynamicProgrammingSolver = None

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Cambiar en producción

# Configuración global
app.config['DEBUG'] = True

# Inicializar servicios globales
problem_repository = InMemoryProblemRepository()
solution_repository = InMemorySolutionRepository()
sensitivity_analyzer = BasicSensitivityAnalyzer()
ai_service = MachineLearningAIService()
visualization_service = PlotlyVisualizationService()
ai_problem_generator = AIProblemGenerator()

# Casos de uso globales
get_history_usecase = GetProblemHistoryUseCase(problem_repository, solution_repository)
generate_report_usecase = GenerateReportUseCase(problem_repository, solution_repository, visualization_service)


@app.route('/')
def index():
    """Página principal"""
    try:
        # Obtener estadísticas
        history_request = GetProblemHistoryRequest(limit=5)
        history_response = get_history_usecase.execute(history_request)
        
        problem_count = len(history_response.problems)
        
        all_solutions = []
        for solutions in history_response.solutions.values():
            all_solutions.extend(solutions)
        
        solution_count = len(all_solutions)
        
        avg_time = 0.0
        if all_solutions:
            avg_time = sum(s.execution_time for s in all_solutions) / len(all_solutions)
        
        recent_solutions = all_solutions[:5] if all_solutions else []
        
        return render_template('index.html',
                             problem_count=problem_count,
                             solution_count=solution_count,
                             avg_time=f"{avg_time:.3f}",
                             recent_solutions=recent_solutions)
    except Exception as e:
        flash(f'Error al cargar la página principal: {str(e)}', 'error')
        return render_template('index.html',
                             problem_count=0,
                             solution_count=0,
                             avg_time="0.000",
                             recent_solutions=[])


@app.route('/linear_programming', methods=['GET', 'POST'])
def linear_programming():
    """Página de programación lineal"""
    if request.method == 'POST':
        try:
            # Procesar formulario de PL
            problem_name = request.form.get('problem_name', 'Problema PL')
            description = request.form.get('description', '')
            objective_sense = request.form.get('objective_sense', 'minimize')
            
            # Procesar variables
            var_names = request.form.getlist('var_name[]')
            var_coeffs = request.form.getlist('var_coeff[]')
            var_lowers = request.form.getlist('var_lower[]')
            var_uppers = request.form.getlist('var_upper[]')
            var_types = request.form.getlist('var_type[]')
            
            print(f"Variables recibidas: {var_names}")
            print(f"Coeficientes recibidos: {var_coeffs}")
            
            variables = []
            objective_vars = {}
            
            for i, name in enumerate(var_names):
                if name:
                    coeff = float(var_coeffs[i]) if var_coeffs[i] else 0.0
                    lower = float(var_lowers[i]) if var_lowers[i] else 0.0
                    upper = float(var_uppers[i]) if var_uppers[i] else None
                    var_type = var_types[i] if i < len(var_types) else 'continuous'
                    
                    print(f"Variable {name}: coeff={coeff}, lower={lower}, upper={upper}")
                    
                    variables.append(Variable(
                        name=name,
                        lower_bound=lower,
                        upper_bound=upper,
                        var_type=var_type,
                        coefficient=coeff
                    ))
                    
                    objective_vars[name] = coeff
            
            # Procesar restricciones
            const_names = request.form.getlist('const_name[]')
            const_coeffs = request.form.getlist('const_coeffs[]')
            const_operators = request.form.getlist('const_operator[]')
            const_rhs = request.form.getlist('const_rhs[]')
            
            print(f"Restricciones recibidas: {const_names}")
            print(f"Coeficientes restricciones: {const_coeffs}")
            print(f"Operadores: {const_operators}")
            print(f"RHS: {const_rhs}")
            
            constraints = []
            
            for i, name in enumerate(const_names):
                if name and i < len(const_coeffs):
                    # Parsear coeficientes
                    coeffs_str = const_coeffs[i].replace(' ', '')
                    coeffs_list = [float(x) for x in coeffs_str.split(',') if x]
                    
                    print(f"Restricción {name}: coeffs_list={coeffs_list}")
                    
                    # Crear diccionario de variables
                    var_dict = {}
                    for j, var_name in enumerate(var_names[:len(coeffs_list)]):
                        if var_name:
                            var_dict[var_name] = coeffs_list[j]
                    
                    operator = const_operators[i] if i < len(const_operators) else '<='
                    rhs = float(const_rhs[i]) if i < len(const_rhs) and const_rhs[i] else 0.0
                    
                    print(f"Restricción {name}: var_dict={var_dict}, operator={operator}, rhs={rhs}")
                    
                    constraints.append(Constraint(
                        name=name,
                        variables=var_dict,
                        operator=operator,
                        rhs=rhs
                    ))
            
            # Crear función objetivo
            objective = ObjectiveFunction(
                variables=objective_vars,
                sense=objective_sense,
                name='objetivo'
            )
            
            print(f"Función objetivo: sense={objective_sense}, variables={objective_vars}")
            
            # Crear problema
            problem = LinearProgrammingProblem(
                name=problem_name,
                description=description,
                variables=variables,
                constraints=constraints,
                objective=objective
            )
            
            # Resolver problema
            solver_type = request.form.get('solver', 'scipy')
            include_sensitivity = 'include_sensitivity' in request.form
            include_ai = 'include_ai' in request.form
            
            # Seleccionar solucionador
            print(f"Solver type requested: {solver_type}")
            print(f"PulpLinearProgrammingSolver available: {PulpLinearProgrammingSolver is not None}")
            
            if PulpLinearProgrammingSolver:
                solver = PulpLinearProgrammingSolver()
                print("Using PulpLinearProgrammingSolver")
            else:
                solver = MockLinearProgrammingSolver()  # Fallback a mock
                print("Using MockLinearProgrammingSolver")
            
            # Ejecutar caso de uso
            solve_usecase = SolveOptimizationProblemUseCase(
                solver=solver,
                problem_repository=problem_repository,
                solution_repository=solution_repository,
                sensitivity_analyzer=sensitivity_analyzer if include_sensitivity else None,
                ai_service=ai_service if include_ai else None,
                visualization_service=visualization_service
            )
            
            solve_request = SolveOptimizationProblemRequest(
                problem=problem,
                include_sensitivity=include_sensitivity,
                include_ai_analysis=include_ai
            )
            
            result = solve_usecase.execute(solve_request)
            
            if result.success:
                return jsonify({
                    'success': True,
                    'solution': {
                        'status': result.solution.status,
                        'objective_value': result.solution.objective_value,
                        'variables_values': result.solution.variables_values,
                        'execution_time': result.solution.execution_time,
                        'solver_info': result.solution.solver_info
                    },
                    'visualization_data': result.visualization_data,
                    'ai_recommendations': result.ai_recommendations
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error al procesar el problema: {str(e)}'
            })
    
    return render_template('linear_programming.html')


@app.route('/transportation', methods=['GET', 'POST'])
def transportation():
    """Página de problemas de transporte"""
    if request.method == 'POST':
        try:
            # Procesar formulario de transporte
            problem_name = request.form.get('problem_name', 'Problema de Transporte')
            description = request.form.get('description', '')
            
            # Procesar datos de oferta y demanda
            supplies = [float(x) for x in request.form.get('supplies', '').split(',') if x.strip()]
            demands = [float(x) for x in request.form.get('demands', '').split(',') if x.strip()]
            
            # Procesar matriz de costos
            costs_str = request.form.get('costs_matrix', '')
            costs = []
            for row in costs_str.strip().split('\n'):
                if row.strip():
                    cost_row = [float(x) for x in row.split(',') if x.strip()]
                    costs.append(cost_row)
            
            # Crear problema
            problem = TransportationProblem(
                name=problem_name,
                description=description,
                supplies=supplies,
                demands=demands,
                costs=costs
            )
            
            # Resolver problema
            algorithm = request.form.get('algorithm', 'vogel')
            
            if algorithm == 'vogel' and VogelTransportationSolver:
                solver = VogelTransportationSolver()
            elif algorithm == 'simplex' and TransportationSimplexSolver:
                solver = TransportationSimplexSolver()
            else:
                solver = MockTransportationSolver()
            
            solve_usecase = SolveOptimizationProblemUseCase(
                solver=solver,
                problem_repository=problem_repository,
                solution_repository=solution_repository,
                sensitivity_analyzer=sensitivity_analyzer,
                ai_service=ai_service,
                visualization_service=visualization_service
            )
            
            solve_request = SolveOptimizationProblemRequest(
                problem=problem,
                include_sensitivity=True,
                include_ai_analysis=True
            )
            
            result = solve_usecase.execute(solve_request)
            
            if result.success:
                return jsonify({
                    'success': True,
                    'solution': {
                        'status': result.solution.status,
                        'objective_value': result.solution.objective_value,
                        'allocation_matrix': result.solution.variables_values,
                        'execution_time': result.solution.execution_time,
                        'solver_info': result.solution.solver_info
                    },
                    'visualization_data': result.visualization_data,
                    'ai_recommendations': result.ai_recommendations
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error al procesar el problema: {str(e)}'
            })
    
    return render_template('transportation.html')


@app.route('/networks', methods=['GET', 'POST'])
def networks():
    """Página de problemas de redes"""
    if request.method == 'POST':
        try:
            # Procesar formulario de redes
            problem_name = request.form.get('problem_name', 'Problema de Redes')
            problem_type = request.form.get('problem_type')
            algorithm = request.form.get('algorithm', 'auto')
            
            # Procesar nodos y arcos
            nodes = [x.strip() for x in request.form.get('nodes', '').split(',') if x.strip()]
            
            edges = []
            form_data = request.form
            edge_count = 0
            
            # Detectar número de arcos
            while f'edge_from_{edge_count}' in form_data:
                from_node = form_data.get(f'edge_from_{edge_count}')
                to_node = form_data.get(f'edge_to_{edge_count}')
                weight = form_data.get(f'edge_weight_{edge_count}')
                capacity = form_data.get(f'edge_capacity_{edge_count}')
                
                if from_node and to_node and weight:
                    edge = {
                        'from': from_node,
                        'to': to_node,
                        'weight': float(weight)
                    }
                    if capacity:
                        edge['capacity'] = float(capacity)
                    edges.append(edge)
                
                edge_count += 1
            
            # Crear problema
            problem = NetworkProblem(
                name=problem_name,
                problem_type=problem_type,
                nodes=nodes,
                edges=edges,
                source_node=request.form.get('source_node'),
                target_node=request.form.get('target_node'),
                algorithm=algorithm
            )
            
            # Resolver problema
            if NetworkXSolver:
                solver = NetworkXSolver()
            else:
                solver = MockNetworkSolver()
            
            solve_usecase = SolveOptimizationProblemUseCase(
                solver=solver,
                problem_repository=problem_repository,
                solution_repository=solution_repository,
                sensitivity_analyzer=sensitivity_analyzer,
                ai_service=ai_service,
                visualization_service=visualization_service
            )
            
            solve_request = SolveOptimizationProblemRequest(
                problem=problem,
                include_sensitivity=True,
                include_ai_analysis=True
            )
            
            result = solve_usecase.execute(solve_request)
            
            if result.success:
                return jsonify({
                    'success': True,
                    'solution': result.solution.__dict__,
                    'visualization_data': result.visualization_data,
                    'ai_recommendations': result.ai_recommendations
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error al procesar el problema: {str(e)}'
            })
    
    return render_template('networks.html')


@app.route('/inventory', methods=['GET', 'POST'])
def inventory():
    """Página de problemas de inventario"""
    if request.method == 'POST':
        try:
            # Procesar formulario de inventario
            problem_name = request.form.get('problem_name', 'Problema de Inventario')
            model_type = request.form.get('model_type')
            
            # Parámetros básicos
            demand = float(request.form.get('demand', 0))
            ordering_cost = float(request.form.get('ordering_cost', 0))
            holding_cost = float(request.form.get('holding_cost', 0))
            
            # Parámetros adicionales
            shortage_cost = float(request.form.get('shortage_cost', 0)) if request.form.get('shortage_cost') else None
            production_rate = float(request.form.get('production_rate', 0)) if request.form.get('production_rate') else None
            lead_time = float(request.form.get('lead_time', 0)) if request.form.get('lead_time') else None
            
            # Parámetros estocásticos
            demand_std = float(request.form.get('demand_std', 0)) if request.form.get('demand_std') else None
            service_level = float(request.form.get('service_level', 95)) if request.form.get('service_level') else 95
            
            # Crear problema
            problem = InventoryProblem(
                name=problem_name,
                model_type=model_type,
                demand=demand,
                ordering_cost=ordering_cost,
                holding_cost=holding_cost,
                shortage_cost=shortage_cost,
                production_rate=production_rate,
                lead_time=lead_time,
                demand_std=demand_std,
                service_level=service_level
            )
            
            # Resolver problema
            if model_type.startswith('stochastic') and StochasticInventorySolver:
                solver = StochasticInventorySolver()
            elif ClassicInventorySolver:
                solver = ClassicInventorySolver()
            else:
                solver = MockInventorySolver()
            
            solve_usecase = SolveOptimizationProblemUseCase(
                solver=solver,
                problem_repository=problem_repository,
                solution_repository=solution_repository,
                sensitivity_analyzer=sensitivity_analyzer,
                ai_service=ai_service,
                visualization_service=visualization_service
            )
            
            solve_request = SolveOptimizationProblemRequest(
                problem=problem,
                include_sensitivity=True,
                include_ai_analysis=True
            )
            
            result = solve_usecase.execute(solve_request)
            
            if result.success:
                return jsonify({
                    'success': True,
                    'solution': result.solution.__dict__,
                    'visualization_data': result.visualization_data,
                    'ai_recommendations': result.ai_recommendations
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error al procesar el problema: {str(e)}'
            })
    
    return render_template('inventory.html')


@app.route('/dynamic_programming', methods=['GET', 'POST'])
def dynamic_programming():
    """Página de programación dinámica"""
    if request.method == 'POST':
        try:
            # Procesar formulario de programación dinámica
            problem_name = request.form.get('problem_name', 'Problema de Programación Dinámica')
            problem_type = request.form.get('problem_type')
            optimization_type = request.form.get('optimization_type', 'maximize')
            
            problem_data = {
                'problem_type': problem_type,
                'optimization_type': optimization_type
            }
            
            # Procesar datos específicos según el tipo de problema
            if problem_type in ['knapsack_01', 'knapsack_unbounded']:
                problem_data['capacity'] = int(request.form.get('capacity', 0))
                problem_data['items'] = []
                
                # Procesar elementos
                item_count = int(request.form.get('item_count', 0))
                for i in range(item_count):
                    weight = request.form.get(f'weight_{i}')
                    value = request.form.get(f'value_{i}')
                    if weight and value:
                        problem_data['items'].append({
                            'weight': int(weight),
                            'value': int(value)
                        })
            
            elif problem_type in ['lcs', 'edit_distance']:
                problem_data['string1'] = request.form.get('string1', '')
                problem_data['string2'] = request.form.get('string2', '')
            
            elif problem_type == 'coin_change':
                problem_data['target_amount'] = int(request.form.get('target_amount', 0))
                denominations = request.form.get('coin_denominations', '')
                problem_data['denominations'] = [int(x.strip()) for x in denominations.split(',') if x.strip()]
            
            # Crear problema
            problem = DynamicProgrammingProblem(
                name=problem_name,
                problem_type=problem_type,
                optimization_type=optimization_type,
                parameters=problem_data
            )
            
            # Resolver problema
            if TabulatedDynamicProgrammingSolver:
                solver = TabulatedDynamicProgrammingSolver()
            else:
                solver = MockDynamicProgrammingSolver()
            
            solve_usecase = SolveOptimizationProblemUseCase(
                solver=solver,
                problem_repository=problem_repository,
                solution_repository=solution_repository,
                sensitivity_analyzer=sensitivity_analyzer,
                ai_service=ai_service,
                visualization_service=visualization_service
            )
            
            solve_request = SolveOptimizationProblemRequest(
                problem=problem,
                include_sensitivity=True,
                include_ai_analysis=True
            )
            
            result = solve_usecase.execute(solve_request)
            
            if result.success:
                return jsonify({
                    'success': True,
                    'solution': result.solution.__dict__,
                    'visualization_data': result.visualization_data,
                    'ai_recommendations': result.ai_recommendations
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.message
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error al procesar el problema: {str(e)}'
            })
    
    return render_template('dynamic_programming.html')


@app.route('/network')
def network():
    """Página de problemas de redes - nueva plantilla mejorada"""
    return render_template('network.html')


@app.route('/inventory_management')
def inventory_management():
    """Página de gestión de inventarios - nueva plantilla mejorada"""
    return render_template('inventory_management.html')


@app.route('/history')
def history():
    """Página de historial de problemas"""
    try:
        history_request = GetProblemHistoryRequest()
        history_response = get_history_usecase.execute(history_request)
        
        return render_template('history.html',
                             problems=history_response.problems,
                             solutions=history_response.solutions)
    except Exception as e:
        flash(f'Error al cargar historial: {str(e)}', 'error')
        return render_template('history.html', problems=[], solutions={})


@app.route('/analytics')
def analytics():
    """Página de análisis y estadísticas"""
    return render_template('analytics.html')


@app.route('/solution/<solution_id>')
def view_solution(solution_id):
    """Ver detalles de una solución específica"""
    try:
        solution = solution_repository.get_by_id(solution_id)
        if not solution:
            flash('Solución no encontrada', 'error')
            return redirect(url_for('history'))
        
        problem = problem_repository.get_by_id(solution.problem_id)
        
        return render_template('solution_detail.html',
                             solution=solution,
                             problem=problem)
    except Exception as e:
        flash(f'Error al cargar la solución: {str(e)}', 'error')
        return redirect(url_for('history'))


@app.route('/api/problem/<problem_id>/report')
def generate_problem_report(problem_id):
    """API para generar reporte de un problema"""
    try:
        report_request = GenerateReportRequest(
            problem_id=problem_id,
            include_sensitivity=True,
            include_visualizations=True
        )
        
        result = generate_report_usecase.execute(report_request)
        
        if result.success:
            return result.report_content, 200, {'Content-Type': 'text/html'}
        else:
            return jsonify({'error': result.message}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ai_analyze', methods=['POST'])
def ai_analyze():
    """Endpoint para análisis automático de problemas con IA"""
    try:
        data = request.get_json()
        description = data.get('description', '')
        problem_type = data.get('problem_type', None)
        
        if not description.strip():
            return jsonify({
                'success': False,
                'message': 'La descripción no puede estar vacía'
            })
        
        # Analizar la descripción con IA
        analysis = ai_problem_generator.analyze_description(description, problem_type)
        
        return jsonify({
            'success': True,
            'analysis': {
                'problem_type': analysis.problem_type,
                'variables': analysis.variables,
                'constraints': analysis.constraints,
                'objective': analysis.objective,
                'parameters': analysis.parameters,
                'confidence': analysis.confidence,
                'suggestions': analysis.suggestions
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error en análisis IA: {str(e)}',
            'error': traceback.format_exc()
        })


@app.route('/ai_generate_problem', methods=['POST'])
def ai_generate_problem():
    """Endpoint para generar y resolver problema completo desde descripción"""
    try:
        data = request.get_json()
        description = data.get('description', '')
        problem_type = data.get('problem_type', None)
        auto_solve = data.get('auto_solve', False)
        
        if not description.strip():
            return jsonify({
                'success': False,
                'message': 'La descripción no puede estar vacía'
            })
        
        # Analizar la descripción
        analysis = ai_problem_generator.analyze_description(description, problem_type)
        
        # Generar objeto del problema
        problem = ai_problem_generator.generate_problem_object(analysis)
        problem.description = description
        
        response_data = {
            'success': True,
            'analysis': {
                'problem_type': analysis.problem_type,
                'variables': analysis.variables,
                'constraints': analysis.constraints,
                'objective': analysis.objective,
                'parameters': analysis.parameters,
                'confidence': analysis.confidence,
                'suggestions': analysis.suggestions
            },
            'problem_generated': True
        }
        
        # Si se solicita resolución automática
        if auto_solve:
            # Seleccionar solucionador apropiado
            solver = _get_solver_for_problem(problem, analysis.problem_type)
            
            if solver:
                # Ejecutar caso de uso de solución
                solve_usecase = SolveOptimizationProblemUseCase(
                    solver=solver,
                    problem_repository=problem_repository,
                    solution_repository=solution_repository,
                    sensitivity_analyzer=sensitivity_analyzer,
                    ai_service=ai_service,
                    visualization_service=visualization_service
                )
                
                solve_request = SolveOptimizationProblemRequest(
                    problem=problem,
                    include_sensitivity=True,
                    include_ai_analysis=True
                )
                
                result = solve_usecase.execute(solve_request)
                
                if result.success:
                    response_data.update({
                        'solution': {
                            'status': result.solution.status,
                            'objective_value': result.solution.objective_value,
                            'variables_values': result.solution.variables_values,
                            'execution_time': result.solution.execution_time,
                            'solver_info': result.solution.solver_info,
                            'sensitivity_analysis': result.solution.sensitivity_analysis
                        },
                        'visualization_data': result.visualization_data,
                        'ai_recommendations': result.ai_recommendations,
                        'problem_solved': True
                    })
                else:
                    response_data.update({
                        'solution_error': result.message,
                        'problem_solved': False
                    })
            else:
                response_data.update({
                    'solution_error': 'No hay solucionador disponible para este tipo de problema',
                    'problem_solved': False
                })
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generando problema: {str(e)}',
            'error': traceback.format_exc()
        })


def _get_solver_for_problem(problem, problem_type):
    """Selecciona el solucionador apropiado para el tipo de problema"""
    try:
        if problem_type == "linear_programming":
            if PulpLinearProgrammingSolver:
                return PulpLinearProgrammingSolver()
            elif ScipyLinearProgrammingSolver:
                return ScipyLinearProgrammingSolver()
            else:
                return MockLinearProgrammingSolver()
        
        elif problem_type == "transportation":
            if VogelTransportationSolver:
                return VogelTransportationSolver()
            elif TransportationSimplexSolver:
                return TransportationSimplexSolver()
            else:
                return MockTransportationSolver()
        
        elif problem_type == "network":
            if NetworkXSolver:
                return NetworkXSolver()
            elif CustomNetworkSolver:
                return CustomNetworkSolver()
            else:
                return MockNetworkSolver()
        
        elif problem_type == "inventory":
            if ClassicInventorySolver:
                return ClassicInventorySolver()
            else:
                return MockInventorySolver()
        
        elif problem_type == "dynamic_programming":
            if RecursiveDynamicProgrammingSolver:
                return RecursiveDynamicProgrammingSolver()
            elif TabulatedDynamicProgrammingSolver:
                return TabulatedDynamicProgrammingSolver()
            else:
                return MockDynamicProgrammingSolver()
        
        else:
            return MockLinearProgrammingSolver()  # Por defecto
    
    except Exception as e:
        print(f"Error seleccionando solucionador: {str(e)}")
        return MockLinearProgrammingSolver()


# Mock solucionadores para cuando las dependencias no estén disponibles
class MockLinearProgrammingSolver:
    """Solucionador mock para demostración"""
    
    def solve(self, problem):
        """Resuelve el problema con valores simulados"""
        import time
        import random
        
        start_time = time.time()
        
        # Simular tiempo de procesamiento
        time.sleep(0.1)
        
        # Generar solución mock
        variables_values = {}
        for var in problem.variables:
            variables_values[var.name] = round(random.uniform(0, 10), 2)
        
        objective_value = sum(
            problem.objective.variables.get(var_name, 0) * value
            for var_name, value in variables_values.items()
        )
        
        return Solution(
            problem_id=problem.id,
            status="optimal",
            objective_value=objective_value,
            variables_values=variables_values,
            execution_time=time.time() - start_time,
            solver_info={
                'solver_name': 'Mock Solver',
                'note': 'Solución simulada para demostración'
            }
        )
    
    def get_solver_info(self):
        return {
            'name': 'Mock Solver',
            'type': 'Simulation',
            'note': 'Para propósitos de demostración'
        }


class MockTransportationSolver:
    """Solucionador mock para problemas de transporte"""
    
    def solve(self, problem):
        import time
        import random
        
        start_time = time.time()
        time.sleep(0.1)
        
        # Generar asignación mock
        allocation_matrix = {}
        total_cost = 0
        
        for i, supply in enumerate(problem.supplies):
            for j, demand in enumerate(problem.demands):
                if i < len(problem.costs) and j < len(problem.costs[i]):
                    allocation = round(random.uniform(0, min(supply, demand)), 2)
                    allocation_matrix[f'x_{i}_{j}'] = allocation
                    total_cost += allocation * problem.costs[i][j]
        
        return Solution(
            problem_id=problem.id,
            status="optimal",
            objective_value=total_cost,
            variables_values=allocation_matrix,
            execution_time=time.time() - start_time,
            solver_info={'solver_name': 'Mock Transportation Solver'}
        )


class MockNetworkSolver:
    """Solucionador mock para problemas de redes"""
    
    def solve(self, problem):
        import time
        import random
        
        start_time = time.time()
        time.sleep(0.1)
        
        result_data = {}
        
        if problem.problem_type == 'shortest_path':
            result_data = {
                'path': problem.nodes[:3],  # Simulación simple
                'distance': round(random.uniform(10, 50), 2)
            }
        elif problem.problem_type == 'max_flow':
            result_data = {
                'max_flow': round(random.uniform(10, 100), 2),
                'flow_edges': {f"{edge['from']}-{edge['to']}": round(random.uniform(0, 20), 2) for edge in problem.edges}
            }
        
        return Solution(
            problem_id=problem.id,
            status="optimal",
            objective_value=result_data.get('distance', result_data.get('max_flow', 0)),
            variables_values=result_data,
            execution_time=time.time() - start_time,
            solver_info={'solver_name': 'Mock Network Solver'}
        )


class MockInventorySolver:
    """Solucionador mock para problemas de inventario"""
    
    def solve(self, problem):
        import time
        import math
        
        start_time = time.time()
        time.sleep(0.1)
        
        # Calcular EOQ básico
        if problem.demand and problem.ordering_cost and problem.holding_cost:
            eoq = math.sqrt((2 * problem.demand * problem.ordering_cost) / problem.holding_cost)
            total_cost = (problem.demand * problem.ordering_cost / eoq) + (eoq * problem.holding_cost / 2)
        else:
            eoq = 100
            total_cost = 1000
        
        result_data = {
            'optimal_quantity': round(eoq, 0),
            'total_cost': round(total_cost, 2),
            'cycle_time': round(eoq / problem.demand, 3) if problem.demand else 0.1,
            'orders_per_year': round(problem.demand / eoq, 0) if eoq else 10
        }
        
        return Solution(
            problem_id=problem.id,
            status="optimal",
            objective_value=total_cost,
            variables_values=result_data,
            execution_time=time.time() - start_time,
            solver_info={'solver_name': 'Mock Inventory Solver'}
        )


class MockDynamicProgrammingSolver:
    """Solucionador mock para problemas de programación dinámica"""
    
    def solve(self, problem):
        import time
        import random
        
        start_time = time.time()
        time.sleep(0.1)
        
        result_data = {}
        
        if problem.problem_type == 'knapsack_01':
            items = problem.parameters.get('items', [])
            capacity = problem.parameters.get('capacity', 0)
            
            # Simulación simple
            selected_items = random.sample(range(len(items)), min(3, len(items)))
            total_value = sum(items[i]['value'] for i in selected_items if i < len(items))
            total_weight = sum(items[i]['weight'] for i in selected_items if i < len(items))
            
            result_data = {
                'optimal_value': total_value,
                'selected_items': selected_items,
                'total_weight': total_weight
            }
            
        elif problem.problem_type == 'lcs':
            string1 = problem.parameters.get('string1', '')
            string2 = problem.parameters.get('string2', '')
            
            # Simulación muy básica
            common_chars = set(string1) & set(string2)
            lcs_length = len(common_chars)
            
            result_data = {
                'lcs_length': lcs_length,
                'lcs_string': ''.join(sorted(common_chars)[:lcs_length])
            }
            
        elif problem.problem_type == 'coin_change':
            target = problem.parameters.get('target_amount', 0)
            denominations = problem.parameters.get('denominations', [1])
            
            # Algoritmo greedy simple
            coins_used = []
            remaining = target
            for coin in sorted(denominations, reverse=True):
                while remaining >= coin:
                    coins_used.append(coin)
                    remaining -= coin
            
            result_data = {
                'min_coins': len(coins_used),
                'coin_combination': coins_used
            }
        
        return Solution(
            problem_id=problem.id,
            status="optimal",
            objective_value=result_data.get('optimal_value', result_data.get('lcs_length', result_data.get('min_coins', 0))),
            variables_values=result_data,
            execution_time=time.time() - start_time,
            solver_info={'solver_name': 'Mock Dynamic Programming Solver'}
        )


# Ruta de prueba para el problema de la fábrica textil
@app.route('/test_linear_programming')
def test_linear_programming():
    """Ruta de prueba que resuelve el problema de la fábrica textil directamente"""
    try:
        print("=== INICIANDO PRUEBA DIRECTA ===")
        
        # Crear variables
        x1 = Variable(name="x1", lower_bound=0, upper_bound=None, var_type="continuous", coefficient=40)
        x2 = Variable(name="x2", lower_bound=0, upper_bound=None, var_type="continuous", coefficient=30)
        variables = [x1, x2]
        
        # Crear función objetivo (Maximizar 40x1 + 30x2)
        objective = ObjectiveFunction(
            variables={"x1": 40, "x2": 30},
            sense="maximize",
            name="ganancia_maxima"
        )
        
        # Crear restricciones
        # 1. Tela: 2x1 + 3x2 <= 120
        restriccion_tela = Constraint(
            name="limitacion_tela",
            variables={"x1": 2, "x2": 3},
            operator="<=",
            rhs=120
        )
        
        # 2. Tiempo: x1 + 2x2 <= 80
        restriccion_tiempo = Constraint(
            name="limitacion_tiempo",
            variables={"x1": 1, "x2": 2},
            operator="<=",
            rhs=80
        )
        
        # 3. Demanda mínima: x1 >= 10
        restriccion_demanda = Constraint(
            name="demanda_minima",
            variables={"x1": 1, "x2": 0},
            operator=">=",
            rhs=10
        )
        
        constraints = [restriccion_tela, restriccion_tiempo, restriccion_demanda]
        
        # Crear problema
        problem = LinearProgrammingProblem(
            name="Optimización de Producción Textil - PRUEBA",
            description="Problema de prueba: fábrica textil que produce camisas y pantalones",
            variables=variables,
            constraints=constraints,
            objective=objective
        )
        
        print(f"Problema creado: {problem.name}")
        print(f"Variables: {[v.name for v in problem.variables]}")
        print(f"Función objetivo: {problem.objective.variables} ({problem.objective.sense})")
        print(f"Restricciones: {len(problem.constraints)}")
        
        # Usar solver real
        if PulpLinearProgrammingSolver:
            solver = PulpLinearProgrammingSolver()
            print("Usando PulpLinearProgrammingSolver")
        else:
            print("ERROR: PulpLinearProgrammingSolver no disponible")
            return jsonify({"error": "Solver no disponible"})
        
        # Resolver
        solution = solver.solve(problem)
        
        print(f"Solución: {solution.status}")
        print(f"Valor objetivo: {solution.objective_value}")
        print(f"Variables: {solution.variables_values}")
        
        return jsonify({
            "success": True,
            "problem": {
                "name": problem.name,
                "description": problem.description,
                "variables": [{"name": v.name, "coefficient": v.coefficient} for v in variables],
                "objective": {"sense": objective.sense, "variables": objective.variables},
                "constraints": [
                    {
                        "name": c.name,
                        "variables": c.variables,
                        "operator": c.operator,
                        "rhs": c.rhs
                    } for c in constraints
                ]
            },
            "solution": {
                "status": solution.status,
                "objective_value": solution.objective_value,
                "variables_values": solution.variables_values,
                "execution_time": solution.execution_time,
                "solver_info": solution.solver_info
            }
        })
        
    except Exception as e:
        print(f"ERROR en prueba: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Crear directorios de datos si no existen
    os.makedirs('data/problems', exist_ok=True)
    os.makedirs('data/solutions', exist_ok=True)
    
    print("🚀 Iniciando OptimizaPro - Sistema de Optimización Empresarial")
    print("📊 Arquitectura limpia implementada")
    print("🤖 Análisis con IA habilitado")
    print("🎨 Interfaz responsive con Bootstrap")
    print("🌐 Disponible en: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
