import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple
import time

from ...core.entities.optimization_entities import NetworkProblem, Solution
from ...core.interfaces.repositories import INetworkSolver


class NetworkXSolver(INetworkSolver):
    """Implementación de solucionador de redes usando NetworkX"""
    
    def solve(self, problem: NetworkProblem) -> Solution:
        """Resuelve el problema de redes según su tipo"""
        # Por defecto, resuelve flujo de costo mínimo
        return self.solve_min_cost_flow(problem)
    
    def solve_max_flow(self, problem: NetworkProblem) -> Solution:
        """Resuelve problema de flujo máximo"""
        try:
            start_time = time.time()
            
            # Crear grafo dirigido
            G = nx.DiGraph()
            
            # Agregar nodos
            for node in problem.nodes:
                G.add_node(node)
            
            # Agregar aristas con capacidades
            for edge in problem.edges:
                source, target, capacity, cost = edge
                G.add_edge(source, target, capacity=capacity, weight=cost)
            
            # Encontrar nodos fuente y sumidero
            sources = [node for node, demand in problem.node_demands.items() if demand < 0]
            sinks = [node for node, demand in problem.node_demands.items() if demand > 0]
            
            if not sources or not sinks:
                return Solution(
                    problem_id=problem.id,
                    status="infeasible",
                    execution_time=time.time() - start_time,
                    solver_info={'error': 'No se encontraron fuentes o sumideros'}
                )
            
            # Resolver flujo máximo entre primera fuente y primer sumidero
            source = sources[0]
            sink = sinks[0]
            
            flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='capacity')
            
            # Convertir resultado
            variables_values = {}
            for u in flow_dict:
                for v in flow_dict[u]:
                    if flow_dict[u][v] > 0:
                        var_name = f"flow_{u}_{v}"
                        variables_values[var_name] = flow_dict[u][v]
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=flow_value,
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'NetworkX',
                    'algorithm': 'Maximum Flow',
                    'source': source,
                    'sink': sink
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
    
    def solve_min_cost_flow(self, problem: NetworkProblem) -> Solution:
        """Resuelve problema de flujo de costo mínimo"""
        try:
            start_time = time.time()
            
            # Crear grafo dirigido
            G = nx.DiGraph()
            
            # Agregar nodos con demandas
            for node in problem.nodes:
                demand = problem.node_demands.get(node, 0)
                G.add_node(node, demand=demand)
            
            # Agregar aristas con capacidades y costos
            for edge in problem.edges:
                source, target, capacity, cost = edge
                G.add_edge(source, target, capacity=capacity, weight=cost)
            
            # Resolver flujo de costo mínimo
            try:
                flow_dict = nx.min_cost_flow(G, demand='demand', capacity='capacity', weight='weight')
                
                # Calcular costo total
                total_cost = 0
                variables_values = {}
                
                for u in flow_dict:
                    for v in flow_dict[u]:
                        if flow_dict[u][v] > 0:
                            var_name = f"flow_{u}_{v}"
                            variables_values[var_name] = flow_dict[u][v]
                            
                            # Encontrar el costo de esta arista
                            edge_cost = G[u][v]['weight']
                            total_cost += flow_dict[u][v] * edge_cost
                
                execution_time = time.time() - start_time
                
                solution = Solution(
                    problem_id=problem.id,
                    status="optimal",
                    objective_value=total_cost,
                    variables_values=variables_values,
                    execution_time=execution_time,
                    solver_info={
                        'solver_name': 'NetworkX',
                        'algorithm': 'Min Cost Flow'
                    }
                )
                
                return solution
                
            except nx.NetworkXUnfeasible:
                return Solution(
                    problem_id=problem.id,
                    status="infeasible",
                    execution_time=time.time() - start_time,
                    solver_info={'error': 'Problema infactible'}
                )
            
        except Exception as e:
            return Solution(
                problem_id=problem.id,
                status="error",
                execution_time=0.0,
                solver_info={'error': str(e)}
            )
    
    def solve_shortest_path(self, problem: NetworkProblem, source: str, target: str) -> Solution:
        """Resuelve problema de camino más corto"""
        try:
            start_time = time.time()
            
            # Crear grafo
            G = nx.DiGraph()
            
            # Agregar nodos
            for node in problem.nodes:
                G.add_node(node)
            
            # Agregar aristas con pesos (costos)
            for edge in problem.edges:
                s, t, capacity, cost = edge
                G.add_edge(s, t, weight=cost)
            
            # Resolver camino más corto
            try:
                path = nx.shortest_path(G, source, target, weight='weight')
                path_length = nx.shortest_path_length(G, source, target, weight='weight')
                
                # Crear variables de decisión para el camino
                variables_values = {}
                for i in range(len(path) - 1):
                    var_name = f"path_{path[i]}_{path[i+1]}"
                    variables_values[var_name] = 1.0
                
                execution_time = time.time() - start_time
                
                solution = Solution(
                    problem_id=problem.id,
                    status="optimal",
                    objective_value=path_length,
                    variables_values=variables_values,
                    execution_time=execution_time,
                    solver_info={
                        'solver_name': 'NetworkX',
                        'algorithm': 'Shortest Path',
                        'path': path,
                        'source': source,
                        'target': target
                    }
                )
                
                return solution
                
            except nx.NetworkXNoPath:
                return Solution(
                    problem_id=problem.id,
                    status="infeasible",
                    execution_time=time.time() - start_time,
                    solver_info={'error': f'No existe camino de {source} a {target}'}
                )
            
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
            'name': 'NetworkX',
            'type': 'Network Flow',
            'library': 'NetworkX',
            'algorithms': ['max_flow', 'min_cost_flow', 'shortest_path']
        }


class CustomNetworkSolver(INetworkSolver):
    """Implementación personalizada de algoritmos de redes"""
    
    def solve(self, problem: NetworkProblem) -> Solution:
        """Resuelve usando algoritmos personalizados"""
        return self.solve_min_cost_flow(problem)
    
    def solve_max_flow(self, problem: NetworkProblem) -> Solution:
        """Implementación del algoritmo Ford-Fulkerson"""
        try:
            start_time = time.time()
            
            # Construcción de la matriz de capacidades
            node_to_idx = {node: i for i, node in enumerate(problem.nodes)}
            n = len(problem.nodes)
            capacity = [[0] * n for _ in range(n)]
            
            for edge in problem.edges:
                source, target, cap, cost = edge
                i, j = node_to_idx[source], node_to_idx[target]
                capacity[i][j] = cap
            
            # Encontrar fuente y sumidero
            sources = [node for node, demand in problem.node_demands.items() if demand < 0]
            sinks = [node for node, demand in problem.node_demands.items() if demand > 0]
            
            if not sources or not sinks:
                return Solution(
                    problem_id=problem.id,
                    status="infeasible",
                    execution_time=time.time() - start_time,
                    solver_info={'error': 'No se encontraron fuentes o sumideros'}
                )
            
            source_idx = node_to_idx[sources[0]]
            sink_idx = node_to_idx[sinks[0]]
            
            # Algoritmo Ford-Fulkerson
            max_flow_value = self._ford_fulkerson(capacity, source_idx, sink_idx)
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=max_flow_value,
                variables_values={},  # Se podría incluir el flujo por cada arista
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'Custom Ford-Fulkerson',
                    'algorithm': 'Ford-Fulkerson'
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
    
    def solve_min_cost_flow(self, problem: NetworkProblem) -> Solution:
        """Implementación personalizada de flujo de costo mínimo"""
        # Por simplicidad, delega a NetworkX
        networkx_solver = NetworkXSolver()
        return networkx_solver.solve_min_cost_flow(problem)
    
    def solve_shortest_path(self, problem: NetworkProblem, source: str, target: str) -> Solution:
        """Implementación del algoritmo de Dijkstra"""
        try:
            start_time = time.time()
            
            # Crear matriz de adyacencia
            node_to_idx = {node: i for i, node in enumerate(problem.nodes)}
            n = len(problem.nodes)
            weights = [[float('inf')] * n for _ in range(n)]
            
            # Inicializar diagonal
            for i in range(n):
                weights[i][i] = 0
            
            # Llenar matriz con costos de aristas
            for edge in problem.edges:
                source_node, target_node, capacity, cost = edge
                i, j = node_to_idx[source_node], node_to_idx[target_node]
                weights[i][j] = cost
            
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            
            # Algoritmo de Dijkstra
            distance, path = self._dijkstra(weights, source_idx, target_idx)
            
            if distance == float('inf'):
                return Solution(
                    problem_id=problem.id,
                    status="infeasible",
                    execution_time=time.time() - start_time,
                    solver_info={'error': f'No existe camino de {source} a {target}'}
                )
            
            # Convertir path de índices a nombres de nodos
            path_nodes = [problem.nodes[i] for i in path]
            
            variables_values = {}
            for i in range(len(path_nodes) - 1):
                var_name = f"path_{path_nodes[i]}_{path_nodes[i+1]}"
                variables_values[var_name] = 1.0
            
            execution_time = time.time() - start_time
            
            solution = Solution(
                problem_id=problem.id,
                status="optimal",
                objective_value=distance,
                variables_values=variables_values,
                execution_time=execution_time,
                solver_info={
                    'solver_name': 'Custom Dijkstra',
                    'algorithm': 'Dijkstra',
                    'path': path_nodes
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
    
    def _ford_fulkerson(self, capacity: List[List[int]], source: int, sink: int) -> int:
        """Implementación del algoritmo Ford-Fulkerson"""
        n = len(capacity)
        flow = [[0] * n for _ in range(n)]
        max_flow = 0
        
        def bfs_find_path():
            visited = [False] * n
            parent = [-1] * n
            queue = [source]
            visited[source] = True
            
            while queue:
                u = queue.pop(0)
                for v in range(n):
                    if not visited[v] and capacity[u][v] - flow[u][v] > 0:
                        queue.append(v)
                        visited[v] = True
                        parent[v] = u
                        if v == sink:
                            return parent
            return None
        
        while True:
            parent = bfs_find_path()
            if parent is None:
                break
            
            # Encontrar capacidad mínima del camino
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, capacity[parent[s]][s] - flow[parent[s]][s])
                s = parent[s]
            
            # Actualizar flujos
            v = sink
            while v != source:
                u = parent[v]
                flow[u][v] += path_flow
                flow[v][u] -= path_flow
                v = parent[v]
            
            max_flow += path_flow
        
        return max_flow
    
    def _dijkstra(self, weights: List[List[float]], source: int, target: int) -> Tuple[float, List[int]]:
        """Implementación del algoritmo de Dijkstra"""
        n = len(weights)
        distance = [float('inf')] * n
        distance[source] = 0
        visited = [False] * n
        parent = [-1] * n
        
        for _ in range(n):
            # Encontrar el nodo no visitado con menor distancia
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or distance[v] < distance[u]):
                    u = v
            
            if distance[u] == float('inf'):
                break
            
            visited[u] = True
            
            # Actualizar distancias
            for v in range(n):
                if weights[u][v] != float('inf'):
                    alt = distance[u] + weights[u][v]
                    if alt < distance[v]:
                        distance[v] = alt
                        parent[v] = u
        
        # Reconstruir camino
        path = []
        current = target
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()
        
        if path[0] != source:
            return float('inf'), []
        
        return distance[target], path
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Retorna información del solucionador"""
        return {
            'name': 'Custom Network Solver',
            'type': 'Network Flow',
            'algorithms': ['Ford-Fulkerson', 'Dijkstra'],
            'implementation': 'Custom'
        }
