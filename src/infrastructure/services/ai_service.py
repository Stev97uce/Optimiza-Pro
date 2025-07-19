import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict, Any, List
import time

from ...core.entities.optimization_entities import OptimizationProblem, Solution
from ...core.interfaces.repositories import ISensitivityAnalyzer, IAIAnalysisService


class BasicSensitivityAnalyzer(ISensitivityAnalyzer):
    """Implementación básica de análisis de sensibilidad"""
    
    def analyze_objective_coefficients(self, problem: OptimizationProblem, solution: Solution) -> Dict[str, Any]:
        """Analiza sensibilidad de coeficientes de función objetivo"""
        analysis = {
            'variables': {},
            'ranges': {},
            'recommendations': []
        }
        
        try:
            if hasattr(problem, 'objective') and problem.objective:
                for var_name, coefficient in problem.objective.variables.items():
                    # Análisis básico de sensibilidad
                    current_value = solution.variables_values.get(var_name, 0)
                    
                    # Rangos estimados (implementación simplificada)
                    lower_range = coefficient * 0.8
                    upper_range = coefficient * 1.2
                    
                    analysis['variables'][var_name] = {
                        'current_coefficient': coefficient,
                        'current_value': current_value,
                        'sensitivity_range': [lower_range, upper_range],
                        'impact_score': abs(coefficient * current_value)
                    }
                    
                    # Recomendaciones
                    if abs(coefficient * current_value) > 1000:  # Alto impacto
                        analysis['recommendations'].append(
                            f"Variable {var_name} tiene alto impacto en el objetivo"
                        )
        
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def analyze_rhs_values(self, problem: OptimizationProblem, solution: Solution) -> Dict[str, Any]:
        """Analiza sensibilidad de valores del lado derecho"""
        analysis = {
            'constraints': {},
            'shadow_prices': {},
            'recommendations': []
        }
        
        try:
            if hasattr(problem, 'constraints'):
                for i, constraint in enumerate(problem.constraints):
                    constraint_name = constraint.name if hasattr(constraint, 'name') else f"constraint_{i}"
                    
                    # Análisis básico
                    rhs_value = constraint.rhs
                    
                    # Estimación de precio sombra (simplificado)
                    shadow_price = 0.1 * rhs_value  # Implementación simplificada
                    
                    analysis['constraints'][constraint_name] = {
                        'rhs_value': rhs_value,
                        'estimated_shadow_price': shadow_price,
                        'operator': constraint.operator
                    }
                    
                    if abs(shadow_price) > 10:
                        analysis['recommendations'].append(
                            f"Restricción {constraint_name} tiene precio sombra significativo"
                        )
        
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Recomendaciones de coeficientes objetivos
        if 'variables' in analysis:
            high_impact_vars = [
                var for var, data in analysis['variables'].items()
                if data.get('impact_score', 0) > 1000
            ]
            
            if high_impact_vars:
                recommendations.append(
                    f"Considere revisar los coeficientes de: {', '.join(high_impact_vars)}"
                )
        
        # Recomendaciones de restricciones
        if 'constraints' in analysis:
            for constraint, data in analysis['constraints'].items():
                if abs(data.get('estimated_shadow_price', 0)) > 10:
                    recommendations.append(
                        f"La restricción {constraint} podría beneficiarse de relajación"
                    )
        
        return recommendations


class MachineLearningAIService(IAIAnalysisService):
    """Servicio de análisis con IA usando técnicas de machine learning"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.parameter_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.pattern_detector = KMeans(n_clusters=3, random_state=42)
        self.is_trained = False
    
    def predict_optimal_parameters(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predice parámetros óptimos usando modelos de ML"""
        predictions = {
            'predicted_parameters': {},
            'confidence_scores': {},
            'recommendations': []
        }
        
        try:
            # Si hay datos históricos, entrenar modelo
            if 'solutions' in historical_data and len(historical_data['solutions']) > 5:
                self._train_models(historical_data)
            
            if self.is_trained and 'current_problem' in historical_data:
                # Extraer características del problema actual
                features = self._extract_features(historical_data['current_problem'])
                
                if features is not None:
                    # Predecir valor objetivo óptimo
                    predicted_objective = self.parameter_predictor.predict([features])[0]
                    
                    predictions['predicted_parameters']['objective_value'] = predicted_objective
                    predictions['confidence_scores']['objective_confidence'] = 0.85  # Simplificado
                    
                    predictions['recommendations'].append(
                        f"Valor objetivo esperado: {predicted_objective:.2f}"
                    )
        
        except Exception as e:
            predictions['error'] = str(e)
        
        return predictions
    
    def detect_patterns(self, solutions: List[Solution]) -> Dict[str, Any]:
        """Detecta patrones en las soluciones usando clustering"""
        patterns = {
            'clusters': {},
            'common_patterns': [],
            'anomalies': []
        }
        
        try:
            if len(solutions) < 3:
                patterns['message'] = "Insuficientes datos para detectar patrones"
                return patterns
            
            # Extraer características de las soluciones
            features_matrix = []
            solution_info = []
            
            for solution in solutions:
                features = self._extract_solution_features(solution)
                if features is not None:
                    features_matrix.append(features)
                    solution_info.append({
                        'id': solution.id,
                        'objective_value': solution.objective_value,
                        'execution_time': solution.execution_time,
                        'status': solution.status
                    })
            
            if len(features_matrix) >= 3:
                # Aplicar clustering
                features_scaled = self.scaler.fit_transform(features_matrix)
                clusters = self.pattern_detector.fit_predict(features_scaled)
                
                # Agrupar soluciones por cluster
                for i, cluster_id in enumerate(clusters):
                    if cluster_id not in patterns['clusters']:
                        patterns['clusters'][cluster_id] = []
                    patterns['clusters'][cluster_id].append(solution_info[i])
                
                # Identificar patrones comunes
                for cluster_id, cluster_solutions in patterns['clusters'].items():
                    if len(cluster_solutions) > 1:
                        avg_objective = np.mean([s['objective_value'] for s in cluster_solutions 
                                               if s['objective_value'] is not None])
                        avg_time = np.mean([s['execution_time'] for s in cluster_solutions])
                        
                        patterns['common_patterns'].append({
                            'cluster': cluster_id,
                            'solutions_count': len(cluster_solutions),
                            'avg_objective': avg_objective,
                            'avg_execution_time': avg_time
                        })
        
        except Exception as e:
            patterns['error'] = str(e)
        
        return patterns
    
    def recommend_improvements(self, problem: OptimizationProblem, solution: Solution) -> List[str]:
        """Recomienda mejoras usando técnicas de IA"""
        recommendations = []
        
        try:
            # Análisis basado en el tiempo de ejecución
            if solution.execution_time > 10:
                recommendations.append(
                    "Considere usar un solucionador más eficiente para problemas grandes"
                )
            
            # Análisis basado en el estado de la solución
            if solution.status != "optimal":
                if solution.status == "infeasible":
                    recommendations.append(
                        "El problema es infactible. Revise las restricciones."
                    )
                elif solution.status == "unbounded":
                    recommendations.append(
                        "El problema no está acotado. Agregue restricciones adicionales."
                    )
            
            # Análisis basado en las variables
            if solution.variables_values:
                # Detectar variables con valores muy altos o bajos
                values = [v for v in solution.variables_values.values() 
                         if isinstance(v, (int, float))]
                
                if values:
                    max_value = max(values)
                    min_value = min(values)
                    
                    if max_value > 1000:
                        recommendations.append(
                            "Algunas variables tienen valores muy altos. Considere escalamiento."
                        )
                    
                    if min_value < 0.01 and min_value > 0:
                        recommendations.append(
                            "Algunas variables tienen valores muy pequeños. Revise precisión."
                        )
            
            # Recomendaciones basadas en el tipo de problema
            problem_type = type(problem).__name__
            if problem_type == "LinearProgrammingProblem":
                recommendations.append(
                    "Para problemas de PL, considere análisis de sensibilidad post-optimal."
                )
            elif problem_type == "TransportationProblem":
                recommendations.append(
                    "Para problemas de transporte, evalúe rutas alternativas."
                )
        
        except Exception as e:
            recommendations.append(f"Error en análisis: {str(e)}")
        
        return recommendations
    
    def _train_models(self, historical_data: Dict[str, Any]):
        """Entrena los modelos de ML con datos históricos"""
        try:
            solutions = historical_data['solutions']
            
            # Preparar datos de entrenamiento
            X = []  # características
            y = []  # valores objetivo
            
            for solution in solutions:
                features = self._extract_solution_features(solution)
                if features is not None and solution.objective_value is not None:
                    X.append(features)
                    y.append(solution.objective_value)
            
            if len(X) >= 3:
                X_scaled = self.scaler.fit_transform(X)
                self.parameter_predictor.fit(X_scaled, y)
                self.is_trained = True
        
        except Exception as e:
            print(f"Error entrenando modelos: {str(e)}")
    
    def _extract_features(self, problem: OptimizationProblem) -> List[float]:
        """Extrae características numéricas de un problema"""
        try:
            features = []
            
            # Número de variables
            features.append(len(problem.variables) if hasattr(problem, 'variables') else 0)
            
            # Número de restricciones
            features.append(len(problem.constraints) if hasattr(problem, 'constraints') else 0)
            
            # Características específicas del tipo de problema
            if hasattr(problem, 'supply'):
                features.extend([sum(problem.supply), len(problem.supply)])
            else:
                features.extend([0, 0])
            
            if hasattr(problem, 'demand'):
                features.extend([sum(problem.demand), len(problem.demand)])
            else:
                features.extend([0, 0])
            
            return features
        
        except Exception:
            return None
    
    def _extract_solution_features(self, solution: Solution) -> List[float]:
        """Extrae características numéricas de una solución"""
        try:
            features = []
            
            # Tiempo de ejecución
            features.append(solution.execution_time)
            
            # Número de variables con valor no cero
            non_zero_vars = sum(1 for v in solution.variables_values.values() 
                              if isinstance(v, (int, float)) and abs(v) > 1e-6)
            features.append(non_zero_vars)
            
            # Total de variables
            features.append(len(solution.variables_values))
            
            # Estado codificado
            status_code = {'optimal': 1, 'infeasible': 0, 'unbounded': -1}.get(solution.status, 0.5)
            features.append(status_code)
            
            return features
        
        except Exception:
            return None
