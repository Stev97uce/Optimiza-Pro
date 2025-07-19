import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ...core.entities.optimization_entities import (
    Variable, Constraint, ObjectiveFunction,
    LinearProgrammingProblem, TransportationProblem,
    NetworkProblem, InventoryProblem, DynamicProgrammingProblem
)


@dataclass
class ProblemAnalysis:
    """Resultado del análisis de IA de un problema"""
    problem_type: str
    variables: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    objective: Dict[str, Any]
    parameters: Dict[str, Any]
    confidence: float
    suggestions: List[str]


class AIProblemGenerator:
    """Servicio de IA para generar problemas de optimización automáticamente"""
    
    def __init__(self):
        self.keywords = self._initialize_keywords()
        self.patterns = self._initialize_patterns()
    
    def analyze_description(self, description: str, problem_type: str = None) -> ProblemAnalysis:
        """Analiza la descripción y genera automáticamente el problema"""
        
        # Limpiar y preparar el texto
        text = self._preprocess_text(description)
        
        # Detectar tipo de problema si no se especifica
        if not problem_type:
            problem_type = self._detect_problem_type(text)
        
        # Extraer entidades numéricas
        numbers = self._extract_numbers(text)
        
        # Generar análisis específico por tipo
        if problem_type == "linear_programming":
            return self._analyze_linear_programming(text, numbers)
        elif problem_type == "transportation":
            return self._analyze_transportation(text, numbers)
        elif problem_type == "network":
            return self._analyze_network(text, numbers)
        elif problem_type == "inventory":
            return self._analyze_inventory(text, numbers)
        elif problem_type == "dynamic_programming":
            return self._analyze_dynamic_programming(text, numbers)
        else:
            return self._analyze_generic(text, numbers, problem_type)
    
    def _initialize_keywords(self) -> Dict[str, List[str]]:
        """Inicializa diccionario de palabras clave por tipo de problema"""
        return {
            "linear_programming": [
                "maximizar", "minimizar", "optimizar", "recursos", "producir", "fabricar",
                "ganancia", "beneficio", "costo", "utilidad", "capacidad", "disponible",
                "limitado", "restricción", "sujeto a", "no exceder", "al menos", "máximo", "mínimo"
            ],
            "transportation": [
                "transportar", "enviar", "distribuir", "almacén", "fábrica", "planta",
                "destino", "origen", "costo de transporte", "demanda", "oferta", "suministro",
                "centro de distribución", "ruta", "flota", "vehículo"
            ],
            "network": [
                "red", "flujo", "nodo", "arista", "camino", "ruta", "conexión",
                "capacidad", "flujo máximo", "costo mínimo", "nodos", "arcos",
                "fuente", "sumidero", "grafo", "conectar"
            ],
            "inventory": [
                "inventario", "stock", "almacenar", "pedir", "orden", "EOQ",
                "costo de mantener", "costo de ordenar", "demanda", "periodo",
                "almacenamiento", "existencias", "reposición", "lote"
            ],
            "dynamic_programming": [
                "etapas", "estados", "decisiones", "recursivo", "secuencial",
                "política", "valor óptimo", "programación dinámica", "mochila",
                "camino", "asignación", "planificación"
            ]
        }
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Inicializa patrones de expresiones regulares"""
        return {
            "variables": [
                r"(\w+)\s+(producir|fabricar|hacer)",
                r"cantidad\s+de\s+(\w+)",
                r"número\s+de\s+(\w+)",
                r"(\w+)\s+unidades",
                r"(\w+)\s+productos?"
            ],
            "constraints": [
                r"no\s+más\s+de\s+(\d+(?:\.\d+)?)",
                r"al\s+menos\s+(\d+(?:\.\d+)?)",
                r"máximo\s+(\d+(?:\.\d+)?)",
                r"mínimo\s+(\d+(?:\.\d+)?)",
                r"capacidad\s+de\s+(\d+(?:\.\d+)?)",
                r"disponible\s+(\d+(?:\.\d+)?)"
            ],
            "objective": [
                r"maximizar\s+(\w+)",
                r"minimizar\s+(\w+)",
                r"optimizar\s+(\w+)",
                r"mayor\s+(\w+)",
                r"menor\s+(\w+)"
            ]
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocesa el texto para análisis"""
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover caracteres especiales pero mantener números y puntos
        text = re.sub(r'[^\w\s\.\,\;\:\(\)]', '', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _detect_problem_type(self, text: str) -> str:
        """Detecta automáticamente el tipo de problema"""
        scores = {}
        
        for problem_type, keywords in self.keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            scores[problem_type] = score
        
        # Retorna el tipo con mayor score
        if scores:
            return max(scores, key=scores.get)
        else:
            return "linear_programming"  # Por defecto
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extrae todos los números del texto"""
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return [float(n) for n in numbers]
    
    def _analyze_linear_programming(self, text: str, numbers: List[float]) -> ProblemAnalysis:
        """Análisis específico para problemas de programación lineal"""
        
        # Detectar variables de decisión
        variables = []
        var_patterns = [
            r"producir\s+(\w+)",
            r"fabricar\s+(\w+)", 
            r"cantidad\s+de\s+(\w+)",
            r"(\w+)\s+productos?",
            r"(\w+)\s+unidades"
        ]
        
        found_vars = set()
        for pattern in var_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2 and match not in ['de', 'la', 'el', 'un', 'una']:
                    found_vars.add(match)
        
        # Si no se encuentran variables específicas, crear genéricas
        if not found_vars:
            found_vars = {'producto_1', 'producto_2'}
        
        # Crear variables
        for i, var_name in enumerate(found_vars):
            variables.append({
                'name': f'x_{var_name}',
                'description': f'Cantidad de {var_name} a producir',
                'lower_bound': 0,
                'upper_bound': None,
                'var_type': 'continuous'
            })
        
        # Detectar función objetivo
        objective_sense = "maximize"
        objective_terms = {}
        
        if any(word in text for word in ["minimizar", "menor", "reducir", "costo"]):
            objective_sense = "minimize"
        
        # Asignar coeficientes basados en contexto
        for i, var in enumerate(variables):
            if "ganancia" in text or "beneficio" in text or "utilidad" in text:
                # Coeficientes para maximizar ganancia
                objective_terms[var['name']] = numbers[i] if i < len(numbers) else (50 + i * 10)
            elif "costo" in text:
                # Coeficientes para minimizar costo  
                objective_terms[var['name']] = numbers[i] if i < len(numbers) else (30 + i * 5)
            else:
                # Coeficientes por defecto
                objective_terms[var['name']] = numbers[i] if i < len(numbers) else (40 + i * 8)
        
        objective = {
            'sense': objective_sense,
            'variables': objective_terms,
            'name': 'objetivo_principal'
        }
        
        # Detectar restricciones
        constraints = []
        constraint_patterns = [
            (r"no\s+más\s+de\s+(\d+(?:\.\d+)?)", "<="),
            (r"máximo\s+(\d+(?:\.\d+)?)", "<="), 
            (r"al\s+menos\s+(\d+(?:\.\d+)?)", ">="),
            (r"mínimo\s+(\d+(?:\.\d+)?)", ">="),
            (r"capacidad\s+de\s+(\d+(?:\.\d+)?)", "<="),
            (r"disponible\s+(\d+(?:\.\d+)?)", "<=")
        ]
        
        for i, (pattern, operator) in enumerate(constraint_patterns):
            matches = re.findall(pattern, text)
            for j, match in enumerate(matches):
                rhs_value = float(match)
                
                # Crear coeficientes para las variables
                var_coefficients = {}
                for var in variables:
                    # Coeficientes basados en el contexto
                    if "hora" in text or "tiempo" in text:
                        var_coefficients[var['name']] = 1 + (j * 0.5)
                    elif "material" in text or "recursos" in text:
                        var_coefficients[var['name']] = 2 + j
                    else:
                        var_coefficients[var['name']] = 1
                
                constraints.append({
                    'name': f'restriccion_{len(constraints) + 1}',
                    'variables': var_coefficients,
                    'operator': operator,
                    'rhs': rhs_value,
                    'description': f'Restricción de {["recursos", "capacidad", "tiempo", "material"][i % 4]}'
                })
        
        # Si no hay restricciones explícitas, crear algunas por defecto
        if not constraints:
            for i, var in enumerate(variables):
                constraints.append({
                    'name': f'capacidad_{i+1}',
                    'variables': {var['name']: 1},
                    'operator': '<=',
                    'rhs': numbers[i] if i < len(numbers) else 100,
                    'description': f'Capacidad máxima para {var["description"]}'
                })
        
        suggestions = [
            "Revise que las variables representen decisiones reales del problema",
            "Verifique que las restricciones capturen todas las limitaciones",
            "Confirme que la función objetivo refleje el verdadero propósito",
            "Considere agregar restricciones de no negatividad si es necesario"
        ]
        
        return ProblemAnalysis(
            problem_type="linear_programming",
            variables=variables,
            constraints=constraints,
            objective=objective,
            parameters={},
            confidence=0.85,
            suggestions=suggestions
        )
    
    def _analyze_transportation(self, text: str, numbers: List[float]) -> ProblemAnalysis:
        """Análisis específico para problemas de transporte"""
        
        # Detectar número de orígenes y destinos con patrones más específicos
        origin_patterns = [
            r"(\d+)\s+plantas?",
            r"(\d+)\s+f[aá]bricas?",
            r"(\d+)\s+almacenes?",
            r"(\d+)\s+or[íi]genes?",
            r"(\d+)\s+proveedores?",
            r"(\d+)\s+centros?\s+de\s+producción"
        ]
        
        destination_patterns = [
            r"(\d+)\s+destinos?",
            r"(\d+)\s+clientes?",
            r"(\d+)\s+centros?\s+de\s+distribución",
            r"(\d+)\s+mercados?",
            r"(\d+)\s+tiendas?",
            r"(\d+)\s+puntos?\s+de\s+venta"
        ]
        
        num_origins = 3  # Valor por defecto
        num_destinations = 4  # Valor por defecto
        
        # Buscar orígenes
        for pattern in origin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                num_origins = int(matches[0])
                break
        
        # Buscar destinos
        for pattern in destination_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                num_destinations = int(matches[0])
                break
        
        # Extraer nombres específicos de orígenes y destinos
        origin_names = self._extract_entity_names(text, ["planta", "fábrica", "almacén", "origen", "proveedor"])
        destination_names = self._extract_entity_names(text, ["destino", "cliente", "centro", "mercado", "tienda"])
        
        # Completar con nombres genéricos si es necesario
        while len(origin_names) < num_origins:
            origin_names.append(f"Origen_{len(origin_names) + 1}")
        while len(destination_names) < num_destinations:
            destination_names.append(f"Destino_{len(destination_names) + 1}")
        
        # Generar variables de decisión
        variables = []
        for i in range(num_origins):
            for j in range(num_destinations):
                variables.append({
                    'name': f'x_{i+1}_{j+1}',
                    'description': f'Cantidad enviada desde {origin_names[i]} a {destination_names[j]}',
                    'lower_bound': 0,
                    'upper_bound': None,
                    'var_type': 'continuous'
                })
        
        # Extraer oferta y demanda de los números encontrados
        supply = []
        demand = []
        costs_matrix = []
        
        # Buscar patrones específicos para capacidades y demandas
        supply_patterns = [
            r"capacidad\s+de\s+(\d+)",
            r"puede\s+producir\s+(\d+)",
            r"disponible\s+(\d+)",
            r"oferta\s+de\s+(\d+)"
        ]
        
        demand_patterns = [
            r"necesita\s+(\d+)",
            r"demanda\s+de\s+(\d+)",
            r"requiere\s+(\d+)",
            r"solicita\s+(\d+)"
        ]
        
        # Extraer capacidades específicas
        for pattern in supply_patterns:
            found_supply = re.findall(pattern, text, re.IGNORECASE)
            supply.extend([float(s) for s in found_supply])
        
        # Extraer demandas específicas
        for pattern in demand_patterns:
            found_demand = re.findall(pattern, text, re.IGNORECASE)
            demand.extend([float(d) for d in found_demand])
        
        # Completar con números del texto si no hay suficientes datos específicos
        remaining_numbers = [n for n in numbers if n not in supply and n not in demand]
        
        # Completar oferta si es necesario
        while len(supply) < num_origins:
            if remaining_numbers:
                supply.append(remaining_numbers.pop(0))
            else:
                supply.append(100 + len(supply) * 20)
        
        # Completar demanda si es necesario
        while len(demand) < num_destinations:
            if remaining_numbers:
                demand.append(remaining_numbers.pop(0))
            else:
                demand.append(80 + len(demand) * 15)
        
        # Balancear oferta y demanda
        total_supply = sum(supply[:num_origins])
        total_demand = sum(demand[:num_destinations])
        
        if abs(total_supply - total_demand) > 0.1:
            # Ajustar demanda para balancear
            factor = total_supply / total_demand if total_demand > 0 else 1
            demand = [d * factor for d in demand[:num_destinations]]
        
        # Generar matriz de costos
        cost_numbers = remaining_numbers if remaining_numbers else []
        costs_matrix = []
        cost_idx = 0
        
        for i in range(num_origins):
            row = []
            for j in range(num_destinations):
                if cost_idx < len(cost_numbers):
                    row.append(cost_numbers[cost_idx])
                    cost_idx += 1
                else:
                    # Generar costo basado en distancia estimada
                    base_cost = 10
                    variation = np.random.uniform(-3, 5)
                    row.append(max(1, base_cost + variation))
            costs_matrix.append(row)
        
        # Función objetivo (minimizar costos)
        objective_vars = {}
        for i in range(num_origins):
            for j in range(num_destinations):
                objective_vars[f'x_{i+1}_{j+1}'] = costs_matrix[i][j]
        
        objective = {
            'sense': 'minimize',
            'variables': objective_vars,
            'name': 'costo_total_transporte'
        }
        
        # Restricciones de oferta y demanda
        constraints = []
        
        # Restricciones de oferta
        for i in range(num_origins):
            constraint_vars = {}
            for j in range(num_destinations):
                constraint_vars[f'x_{i+1}_{j+1}'] = 1
            
            constraints.append({
                'name': f'oferta_{origin_names[i]}',
                'variables': constraint_vars,
                'operator': '=',
                'rhs': supply[i],
                'description': f'Restricción de oferta de {origin_names[i]}: {supply[i]} unidades'
            })
        
        # Restricciones de demanda
        for j in range(num_destinations):
            constraint_vars = {}
            for i in range(num_origins):
                constraint_vars[f'x_{i+1}_{j+1}'] = 1
            
            constraints.append({
                'name': f'demanda_{destination_names[j]}',
                'variables': constraint_vars,
                'operator': '=',
                'rhs': demand[j],
                'description': f'Restricción de demanda de {destination_names[j]}: {demand[j]} unidades'
            })
        
        # Parámetros adicionales para el formulario
        parameters = {
            'num_origins': num_origins,
            'num_destinations': num_destinations,
            'sources': [{'name': name, 'capacity': supply[i]} for i, name in enumerate(origin_names[:num_origins])],
            'destinations': [{'name': name, 'demand': demand[j]} for j, name in enumerate(destination_names[:num_destinations])],
            'supply': supply[:num_origins],
            'demand': demand[:num_destinations],
            'costs': costs_matrix,
            'cost_matrix': costs_matrix,  # Alias para compatibilidad
            'origins': origin_names[:num_origins],  # Alias para compatibilidad
            'total_supply': sum(supply[:num_origins]),
            'total_demand': sum(demand[:num_destinations]),
            'is_balanced': abs(sum(supply[:num_origins]) - sum(demand[:num_destinations])) < 0.1
        }
        
        suggestions = [
            f"Problema {'balanceado' if parameters['is_balanced'] else 'no balanceado'}: Oferta total = {parameters['total_supply']}, Demanda total = {parameters['total_demand']}",
            "Verifique que los costos de transporte reflejen la realidad (distancia, combustible, etc.)",
            "Considere restricciones adicionales como capacidad de vehículos o rutas exclusivas",
            "Evalúe si hay economías de escala en el transporte",
            "Revise si algunos destinos tienen preferencias por ciertos orígenes"
        ]
        
        return ProblemAnalysis(
            problem_type="transportation",
            variables=variables,
            constraints=constraints,
            objective=objective,
            parameters=parameters,
            confidence=0.85,
            suggestions=suggestions
        )
    
    def _extract_entity_names(self, text: str, keywords: List[str]) -> List[str]:
        """Extrae nombres específicos de entidades basados en palabras clave"""
        names = []
        
        for keyword in keywords:
            # Buscar patrones como "planta A", "fábrica Norte", etc.
            pattern = rf"{keyword}\s+([A-Z]\w*|[A-Z]|Norte|Sur|Este|Oeste|Central|\d+)"
            matches = re.findall(pattern, text, re.IGNORECASE)
            names.extend([match for match in matches if match not in names])
        
        return names
    
    def _analyze_network(self, text: str, numbers: List[float]) -> ProblemAnalysis:
        """Análisis específico para problemas de redes"""
        
        # Detectar número de nodos
        node_patterns = [
            r"(\d+)\s+nodos?",
            r"(\d+)\s+ciudades?",
            r"(\d+)\s+puntos?"
        ]
        
        num_nodes = 5  # Por defecto
        for pattern in node_patterns:
            matches = re.findall(pattern, text)
            if matches:
                num_nodes = int(matches[0])
                break
        
        # Generar nodos
        nodes = [f'Nodo_{i}' for i in range(1, num_nodes + 1)]
        
        # Generar aristas (grafo relativamente conectado)
        edges = []
        for i in range(num_nodes):
            for j in range(i+1, min(i+3, num_nodes)):  # Cada nodo conecta con 1-2 siguientes
                capacity = numbers[len(edges)] if len(edges) < len(numbers) else np.random.uniform(10, 50)
                cost = np.random.uniform(1, 10)
                edges.append((nodes[i], nodes[j], capacity, cost))
        
        # Variables de flujo
        variables = []
        for edge in edges:
            source, target, capacity, cost = edge
            variables.append({
                'name': f'flow_{source}_{target}',
                'description': f'Flujo desde {source} hacia {target}',
                'lower_bound': 0,
                'upper_bound': capacity,
                'var_type': 'continuous'
            })
        
        # Determinar tipo de problema de red
        if "flujo máximo" in text or "máximo flujo" in text:
            problem_subtype = "max_flow"
            objective_sense = "maximize"
        elif "costo mínimo" in text or "menor costo" in text:
            problem_subtype = "min_cost_flow"
            objective_sense = "minimize"
        else:
            problem_subtype = "min_cost_flow"
            objective_sense = "minimize"
        
        # Función objetivo
        objective = {
            'sense': objective_sense,
            'variables': {var['name']: np.random.uniform(1, 5) for var in variables},
            'name': f'objetivo_{problem_subtype}'
        }
        
        # Generar demandas de nodos
        node_demands = {}
        node_demands[nodes[0]] = -sum(numbers[:2]) if len(numbers) >= 2 else -100  # Fuente
        node_demands[nodes[-1]] = abs(node_demands[nodes[0]])  # Sumidero
        for node in nodes[1:-1]:
            node_demands[node] = 0  # Nodos intermedios
        
        parameters = {
            'nodes': nodes,
            'edges': edges,
            'node_demands': node_demands,
            'problem_subtype': problem_subtype
        }
        
        # Restricciones de conservación de flujo
        constraints = []
        for node in nodes:
            constraint_vars = {}
            
            # Flujos que salen del nodo
            for edge in edges:
                source, target, _, _ = edge
                if source == node:
                    constraint_vars[f'flow_{source}_{target}'] = 1
                elif target == node:
                    constraint_vars[f'flow_{source}_{target}'] = -1
            
            if constraint_vars:  # Solo si el nodo tiene conexiones
                constraints.append({
                    'name': f'conservacion_{node}',
                    'variables': constraint_vars,
                    'operator': '=',
                    'rhs': node_demands.get(node, 0),
                    'description': f'Conservación de flujo en {node}'
                })
        
        suggestions = [
            "Verifique que la red esté correctamente conectada",
            "Confirme las capacidades de las aristas",
            "Revise que las demandas de los nodos estén balanceadas",
            "Considere si hay nodos fuente y sumidero claramente definidos"
        ]
        
        return ProblemAnalysis(
            problem_type="network",
            variables=variables,
            constraints=constraints,
            objective=objective,
            parameters=parameters,
            confidence=0.88,
            suggestions=suggestions
        )
    
    def _analyze_inventory(self, text: str, numbers: List[float]) -> ProblemAnalysis:
        """Análisis específico para problemas de inventario"""
        
        # Detectar períodos
        period_patterns = [
            r"(\d+)\s+(?:períodos?|meses|semanas|días)",
            r"durante\s+(\d+)",
            r"por\s+(\d+)\s+(?:períodos?|meses)"
        ]
        
        num_periods = 12  # Por defecto
        for pattern in period_patterns:
            matches = re.findall(pattern, text)
            if matches:
                num_periods = int(matches[0])
                break
        
        # Extraer costos del texto
        holding_cost = 0.5  # Por defecto
        ordering_cost = 50  # Por defecto
        unit_cost = 10  # Por defecto
        
        if "costo de mantener" in text or "costo de almacenamiento" in text:
            holding_cost = numbers[0] if numbers else 0.5
            
        if "costo de ordenar" in text or "costo de pedido" in text:
            ordering_cost = numbers[1] if len(numbers) > 1 else 50
            
        if "costo unitario" in text or "precio unitario" in text:
            unit_cost = numbers[2] if len(numbers) > 2 else 10
        
        # Generar demanda
        if len(numbers) >= num_periods:
            demand = numbers[:num_periods]
        else:
            # Generar demanda con variabilidad
            base_demand = numbers[0] if numbers else 100
            demand = [base_demand * (0.8 + 0.4 * np.random.random()) for _ in range(num_periods)]
        
        # Variables de decisión
        variables = []
        for i in range(1, num_periods + 1):
            variables.extend([
                {
                    'name': f'production_{i}',
                    'description': f'Cantidad a producir en período {i}',
                    'lower_bound': 0,
                    'upper_bound': None,
                    'var_type': 'continuous'
                },
                {
                    'name': f'inventory_{i}',
                    'description': f'Inventario al final del período {i}',
                    'lower_bound': 0,
                    'upper_bound': None,
                    'var_type': 'continuous'
                }
            ])
        
        # Función objetivo (minimizar costos totales)
        objective_terms = {}
        for i in range(1, num_periods + 1):
            objective_terms[f'production_{i}'] = unit_cost
            objective_terms[f'inventory_{i}'] = holding_cost
        
        objective = {
            'sense': 'minimize',
            'variables': objective_terms,
            'name': 'costo_total_inventario'
        }
        
        # Restricciones de balance de inventario
        constraints = []
        initial_inventory = numbers[-1] if len(numbers) > num_periods else 0
        
        for i in range(1, num_periods + 1):
            if i == 1:
                # Primer período
                constraint_vars = {
                    f'production_{i}': 1,
                    f'inventory_{i}': 1
                }
                rhs = demand[i-1] + initial_inventory
            else:
                # Períodos siguientes
                constraint_vars = {
                    f'production_{i}': 1,
                    f'inventory_{i-1}': 1,
                    f'inventory_{i}': 1
                }
                rhs = demand[i-1]
            
            constraints.append({
                'name': f'balance_periodo_{i}',
                'variables': constraint_vars,
                'operator': '=',
                'rhs': rhs,
                'description': f'Balance de inventario en período {i}'
            })
        
        parameters = {
            'periods': num_periods,
            'demand': demand,
            'holding_cost': holding_cost,
            'ordering_cost': ordering_cost,
            'unit_cost': unit_cost,
            'initial_inventory': initial_inventory
        }
        
        suggestions = [
            "Verifique que los costos reflejen la realidad del negocio",
            "Confirme que la demanda proyectada sea precisa",
            "Considere restricciones de capacidad de almacenamiento",
            "Evalúe si hay costos de faltantes (stockout costs)"
        ]
        
        return ProblemAnalysis(
            problem_type="inventory",
            variables=variables,
            constraints=constraints,
            objective=objective,
            parameters=parameters,
            confidence=0.87,
            suggestions=suggestions
        )
    
    def _analyze_dynamic_programming(self, text: str, numbers: List[float]) -> ProblemAnalysis:
        """Análisis específico para problemas de programación dinámica"""
        
        # Detectar número de etapas
        stage_patterns = [
            r"(\d+)\s+etapas?",
            r"(\d+)\s+pasos?",
            r"(\d+)\s+niveles?"
        ]
        
        num_stages = 5  # Por defecto
        for pattern in stage_patterns:
            matches = re.findall(pattern, text)
            if matches:
                num_stages = int(matches[0])
                break
        
        # Detectar si es problema de mochila
        if "mochila" in text:
            return self._analyze_knapsack_problem(text, numbers)
        
        # Variables de decisión por etapa
        variables = []
        for i in range(1, num_stages + 1):
            variables.append({
                'name': f'decision_stage_{i}',
                'description': f'Decisión óptima en etapa {i}',
                'lower_bound': 0,
                'upper_bound': None,
                'var_type': 'integer'
            })
        
        # Estados posibles
        if numbers:
            max_state = int(max(numbers))
        else:
            max_state = 10
        
        states = list(range(max_state + 1))
        decisions = list(range(3))  # 0, 1, 2 como decisiones por defecto
        
        # Función objetivo
        objective = {
            'sense': 'maximize',
            'variables': {var['name']: 1 for var in variables},
            'name': 'valor_total_optimo'
        }
        
        parameters = {
            'stages': num_stages,
            'states': states,
            'decisions': decisions,
            'transition_function': 'custom',
            'reward_function': 'custom'
        }
        
        # Las restricciones en DP son implícitas en la estructura del problema
        constraints = []
        
        suggestions = [
            "Defina claramente los estados posibles en cada etapa",
            "Especifique la función de transición entre estados",
            "Determine la función de recompensa o costo",
            "Verifique que el problema tenga estructura recursiva"
        ]
        
        return ProblemAnalysis(
            problem_type="dynamic_programming",
            variables=variables,
            constraints=constraints,
            objective=objective,
            parameters=parameters,
            confidence=0.75,
            suggestions=suggestions
        )
    
    def _analyze_knapsack_problem(self, text: str, numbers: List[float]) -> ProblemAnalysis:
        """Análisis específico para problema de la mochila"""
        
        # Detectar número de items
        item_patterns = [
            r"(\d+)\s+(?:objetos?|items?|elementos?|productos?)"
        ]
        
        num_items = 5  # Por defecto
        for pattern in item_patterns:
            matches = re.findall(pattern, text)
            if matches:
                num_items = int(matches[0])
                break
        
        # Capacidad de la mochila
        capacity = numbers[0] if numbers else 100
        
        # Generar pesos y valores
        if len(numbers) >= 1 + 2 * num_items:
            weights = numbers[1:1+num_items]
            values = numbers[1+num_items:1+2*num_items]
        else:
            weights = [np.random.uniform(5, 20) for _ in range(num_items)]
            values = [np.random.uniform(10, 50) for _ in range(num_items)]
        
        # Variables binarias para cada item
        variables = []
        for i in range(num_items):
            variables.append({
                'name': f'item_{i+1}',
                'description': f'Seleccionar item {i+1} (1=sí, 0=no)',
                'lower_bound': 0,
                'upper_bound': 1,
                'var_type': 'binary'
            })
        
        # Función objetivo (maximizar valor)
        objective = {
            'sense': 'maximize',
            'variables': {f'item_{i+1}': values[i] for i in range(num_items)},
            'name': 'valor_total_mochila'
        }
        
        # Restricción de capacidad
        constraint_vars = {f'item_{i+1}': weights[i] for i in range(num_items)}
        constraints = [{
            'name': 'capacidad_mochila',
            'variables': constraint_vars,
            'operator': '<=',
            'rhs': capacity,
            'description': 'Restricción de capacidad de la mochila'
        }]
        
        parameters = {
            'num_items': num_items,
            'capacity': capacity,
            'weights': weights,
            'values': values
        }
        
        suggestions = [
            "Verifique que los pesos y valores sean realistas",
            "Confirme que la capacidad de la mochila sea correcta",
            "Considere si hay restricciones adicionales entre items",
            "Evalúe si algunos items son mutuamente excluyentes"
        ]
        
        return ProblemAnalysis(
            problem_type="knapsack",
            variables=variables,
            constraints=constraints,
            objective=objective,
            parameters=parameters,
            confidence=0.92,
            suggestions=suggestions
        )
    
    def _analyze_generic(self, text: str, numbers: List[float], problem_type: str) -> ProblemAnalysis:
        """Análisis genérico para otros tipos de problemas"""
        
        # Crear variables genéricas
        num_vars = min(len(numbers), 5) if numbers else 2
        variables = []
        
        for i in range(num_vars):
            variables.append({
                'name': f'x_{i+1}',
                'description': f'Variable de decisión {i+1}',
                'lower_bound': 0,
                'upper_bound': None,
                'var_type': 'continuous'
            })
        
        # Función objetivo genérica
        objective = {
            'sense': 'maximize',
            'variables': {var['name']: 1 for var in variables},
            'name': 'objetivo_generico'
        }
        
        # Restricción genérica
        constraints = [{
            'name': 'restriccion_generica',
            'variables': {var['name']: 1 for var in variables},
            'operator': '<=',
            'rhs': numbers[0] if numbers else 100,
            'description': 'Restricción genérica del problema'
        }]
        
        suggestions = [
            "Especifique mejor la descripción del problema",
            "Incluya más detalles sobre las variables y restricciones",
            "Defina claramente el objetivo a optimizar"
        ]
        
        return ProblemAnalysis(
            problem_type=problem_type,
            variables=variables,
            constraints=constraints,
            objective=objective,
            parameters={},
            confidence=0.60,
            suggestions=suggestions
        )
    
    def generate_problem_object(self, analysis: ProblemAnalysis):
        """Convierte el análisis en un objeto de problema de optimización"""
        
        if analysis.problem_type == "linear_programming":
            return self._create_linear_programming_problem(analysis)
        elif analysis.problem_type == "transportation":
            return self._create_transportation_problem(analysis)
        elif analysis.problem_type == "network":
            return self._create_network_problem(analysis)
        elif analysis.problem_type == "inventory":
            return self._create_inventory_problem(analysis)
        elif analysis.problem_type == "dynamic_programming":
            return self._create_dynamic_programming_problem(analysis)
        else:
            return self._create_linear_programming_problem(analysis)  # Por defecto
    
    def _create_linear_programming_problem(self, analysis: ProblemAnalysis) -> LinearProgrammingProblem:
        """Crea objeto de problema de programación lineal"""
        
        # Crear variables
        variables = []
        for var_data in analysis.variables:
            variables.append(Variable(
                name=var_data['name'],
                lower_bound=var_data['lower_bound'],
                upper_bound=var_data.get('upper_bound'),
                var_type=var_data['var_type']
            ))
        
        # Crear restricciones
        constraints = []
        for const_data in analysis.constraints:
            constraints.append(Constraint(
                name=const_data['name'],
                variables=const_data['variables'],
                operator=const_data['operator'],
                rhs=const_data['rhs']
            ))
        
        # Crear función objetivo
        objective = ObjectiveFunction(
            variables=analysis.objective['variables'],
            sense=analysis.objective['sense'],
            name=analysis.objective['name']
        )
        
        problem = LinearProgrammingProblem(
            name="Problema Generado por IA",
            description="Problema generado automáticamente basado en descripción de texto",
            variables=variables,
            constraints=constraints,
            objective=objective
        )
        
        return problem
    
    def _create_transportation_problem(self, analysis: ProblemAnalysis) -> TransportationProblem:
        """Crea objeto de problema de transporte"""
        
        params = analysis.parameters
        
        problem = TransportationProblem(
            name="Problema de Transporte Generado por IA",
            description="Problema de transporte generado automáticamente",
            supply=params['supply'],
            demand=params['demand'],
            costs=params['costs'],
            origins=params['origins'],
            destinations=params['destinations']
        )
        
        return problem
    
    def _create_network_problem(self, analysis: ProblemAnalysis) -> NetworkProblem:
        """Crea objeto de problema de redes"""
        
        params = analysis.parameters
        
        problem = NetworkProblem(
            name="Problema de Redes Generado por IA",
            description="Problema de redes generado automáticamente",
            nodes=params['nodes'],
            edges=params['edges'],
            node_demands=params['node_demands']
        )
        
        return problem
    
    def _create_inventory_problem(self, analysis: ProblemAnalysis) -> InventoryProblem:
        """Crea objeto de problema de inventario"""
        
        params = analysis.parameters
        
        problem = InventoryProblem(
            name="Problema de Inventario Generado por IA",
            description="Problema de inventario generado automáticamente",
            periods=params['periods'],
            demand=params['demand'],
            holding_cost=params['holding_cost'],
            ordering_cost=params['ordering_cost'],
            unit_cost=params['unit_cost'],
            initial_inventory=params['initial_inventory']
        )
        
        return problem
    
    def _create_dynamic_programming_problem(self, analysis: ProblemAnalysis) -> DynamicProgrammingProblem:
        """Crea objeto de problema de programación dinámica"""
        
        params = analysis.parameters
        
        problem = DynamicProgrammingProblem(
            name="Problema de PD Generado por IA",
            description="Problema de programación dinámica generado automáticamente",
            stages=params['stages'],
            states=params['states'],
            decisions=params['decisions']
        )
        
        return problem
