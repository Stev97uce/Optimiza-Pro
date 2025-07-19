import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, Any, List
import base64
from io import BytesIO

from ...core.entities.optimization_entities import OptimizationProblem, Solution
from ...core.interfaces.repositories import IVisualizationService


class PlotlyVisualizationService(IVisualizationService):
    """Servicio de visualización usando Plotly para gráficos interactivos"""
    
    def plot_solution(self, problem: OptimizationProblem, solution: Solution) -> Dict[str, Any]:
        """Genera gráficos de la solución"""
        try:
            problem_type = type(problem).__name__
            
            if problem_type == "LinearProgrammingProblem":
                return self._plot_linear_programming(problem, solution)
            elif problem_type == "TransportationProblem":
                return self._plot_transportation(problem, solution)
            elif problem_type == "NetworkProblem":
                return self._plot_network(problem, solution)
            elif problem_type == "InventoryProblem":
                return self._plot_inventory(problem, solution)
            else:
                return self._plot_generic(problem, solution)
                
        except Exception as e:
            return {'error': f"Error en visualización: {str(e)}"}
    
    def _plot_linear_programming(self, problem, solution: Solution) -> Dict[str, Any]:
        """Visualización específica para problemas de PL"""
        plots = {}
        
        # Gráfico de variables de decisión
        if solution.variables_values:
            variables = list(solution.variables_values.keys())
            values = list(solution.variables_values.values())
            
            fig = go.Figure(data=[
                go.Bar(x=variables, y=values, 
                      text=[f'{v:.2f}' for v in values],
                      textposition='auto')
            ])
            fig.update_layout(
                title="Variables de Decisión - Solución Óptima",
                xaxis_title="Variables",
                yaxis_title="Valores",
                template="plotly_white"
            )
            plots['variables_chart'] = fig.to_json()
        
        # Gráfico de análisis de sensibilidad si está disponible
        if solution.sensitivity_analysis:
            plots.update(self._plot_sensitivity_charts(solution.sensitivity_analysis))
        
        return plots
    
    def _plot_transportation(self, problem, solution: Solution) -> Dict[str, Any]:
        """Visualización específica para problemas de transporte"""
        plots = {}
        
        if hasattr(problem, 'origins') and hasattr(problem, 'destinations'):
            # Crear matriz de flujos
            flows = []
            for origin in problem.origins:
                flow_row = []
                for destination in problem.destinations:
                    var_name = f"x_{origin}_{destination}"
                    flow = solution.variables_values.get(var_name, 0)
                    flow_row.append(flow)
                flows.append(flow_row)
            
            # Heatmap de flujos
            fig = go.Figure(data=go.Heatmap(
                z=flows,
                x=problem.destinations,
                y=problem.origins,
                colorscale='Viridis',
                text=[[f'{flows[i][j]:.1f}' for j in range(len(problem.destinations))] 
                      for i in range(len(problem.origins))],
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            fig.update_layout(
                title="Matriz de Flujos de Transporte",
                xaxis_title="Destinos",
                yaxis_title="Orígenes",
                template="plotly_white"
            )
            plots['flow_matrix'] = fig.to_json()
            
            # Gráfico de barras comparativo
            origins_supply = problem.supply if hasattr(problem, 'supply') else []
            destinations_demand = problem.demand if hasattr(problem, 'demand') else []
            
            if origins_supply and destinations_demand:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Oferta por Origen', 'Demanda por Destino')
                )
                
                fig.add_trace(
                    go.Bar(x=problem.origins, y=origins_supply, name="Oferta"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=problem.destinations, y=destinations_demand, name="Demanda"),
                    row=1, col=2
                )
                
                fig.update_layout(
                    title="Análisis de Oferta y Demanda",
                    template="plotly_white"
                )
                plots['supply_demand'] = fig.to_json()
        
        return plots
    
    def _plot_network(self, problem, solution: Solution) -> Dict[str, Any]:
        """Visualización específica para problemas de redes"""
        plots = {}
        
        if hasattr(problem, 'nodes') and hasattr(problem, 'edges'):
            # Crear gráfico de red
            node_trace = go.Scatter(
                x=[], y=[], 
                mode='markers+text',
                marker=dict(size=20, color='lightblue'),
                text=problem.nodes,
                textposition="middle center",
                name="Nodos"
            )
            
            edge_trace = go.Scatter(
                x=[], y=[], 
                mode='lines',
                line=dict(width=2, color='gray'),
                name="Aristas"
            )
            
            # Posicionar nodos en círculo
            n_nodes = len(problem.nodes)
            angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
            node_positions = {
                node: (np.cos(angle), np.sin(angle)) 
                for node, angle in zip(problem.nodes, angles)
            }
            
            # Agregar posiciones de nodos
            for node in problem.nodes:
                x, y = node_positions[node]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
            
            # Agregar aristas
            for edge in problem.edges:
                source, target = edge[0], edge[1]
                x0, y0 = node_positions[source]
                x1, y1 = node_positions[target]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])
            
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title="Visualización de Red",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="Red de Flujo",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="#000000", size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_white"
            )
            plots['network_graph'] = fig.to_json()
        
        return plots
    
    def _plot_inventory(self, problem, solution: Solution) -> Dict[str, Any]:
        """Visualización específica para problemas de inventario"""
        plots = {}
        
        if hasattr(problem, 'demand') and problem.demand:
            periods = list(range(1, len(problem.demand) + 1))
            
            # Gráfico de demanda
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=periods, 
                y=problem.demand,
                mode='lines+markers',
                name='Demanda',
                line=dict(color='red', width=2)
            ))
            
            # Agregar niveles de inventario si están en la solución
            inventory_levels = []
            for i in range(len(problem.demand)):
                var_name = f'inventory_end_period_{i+1}'
                if var_name in solution.variables_values:
                    inventory_levels.append(solution.variables_values[var_name])
            
            if inventory_levels:
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=inventory_levels,
                    mode='lines+markers',
                    name='Inventario',
                    line=dict(color='blue', width=2)
                ))
            
            fig.update_layout(
                title="Análisis de Demanda e Inventario por Período",
                xaxis_title="Período",
                yaxis_title="Cantidad",
                template="plotly_white"
            )
            plots['inventory_analysis'] = fig.to_json()
            
            # Gráfico de producción si está disponible
            production_schedule = []
            for i in range(len(problem.demand)):
                var_name = f'production_period_{i+1}'
                if var_name in solution.variables_values:
                    production_schedule.append(solution.variables_values[var_name])
            
            if production_schedule:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=periods,
                    y=production_schedule,
                    name='Producción',
                    marker_color='green'
                ))
                fig2.update_layout(
                    title="Programa de Producción",
                    xaxis_title="Período",
                    yaxis_title="Cantidad a Producir",
                    template="plotly_white"
                )
                plots['production_schedule'] = fig2.to_json()
        
        return plots
    
    def _plot_generic(self, problem: OptimizationProblem, solution: Solution) -> Dict[str, Any]:
        """Visualización genérica para cualquier tipo de problema"""
        plots = {}
        
        if solution.variables_values:
            variables = list(solution.variables_values.keys())
            values = list(solution.variables_values.values())
            
            # Filtrar solo valores numéricos
            numeric_vars = []
            numeric_values = []
            for var, val in zip(variables, values):
                try:
                    numeric_val = float(val)
                    numeric_vars.append(var)
                    numeric_values.append(numeric_val)
                except (ValueError, TypeError):
                    continue
            
            if numeric_vars:
                fig = go.Figure(data=[
                    go.Bar(x=numeric_vars, y=numeric_values,
                          text=[f'{v:.2f}' for v in numeric_values],
                          textposition='auto')
                ])
                fig.update_layout(
                    title="Variables de Decisión",
                    xaxis_title="Variables",
                    yaxis_title="Valores",
                    template="plotly_white"
                )
                plots['variables_chart'] = fig.to_json()
        
        return plots
    
    def _plot_sensitivity_charts(self, sensitivity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera gráficos de análisis de sensibilidad"""
        plots = {}
        
        # Aquí se implementarían gráficos específicos de sensibilidad
        # Por ahora, retorna placeholder
        
        return plots
    
    def plot_sensitivity_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera gráficos del análisis de sensibilidad"""
        return self._plot_sensitivity_charts(analysis)
    
    def generate_report(self, problem: OptimizationProblem, solution: Solution) -> str:
        """Genera reporte completo en HTML"""
        plots = self.plot_solution(problem, solution)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Optimización - {problem.name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .plot-container {{ margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="row">
                    <div class="col-12">
                        <h1>Reporte de Optimización</h1>
                        <h2>{problem.name}</h2>
                        <p><strong>Descripción:</strong> {problem.description}</p>
                        <p><strong>Tipo de Problema:</strong> {type(problem).__name__}</p>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="metric-card">
                            <h4>Estado de la Solución</h4>
                            <p><span class="badge bg-{'success' if solution.status == 'optimal' else 'warning'}">{solution.status}</span></p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="metric-card">
                            <h4>Valor Objetivo</h4>
                            <p><strong>{solution.objective_value:.2f if solution.objective_value else 'N/A'}</strong></p>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="metric-card">
                            <h4>Tiempo de Ejecución</h4>
                            <p><strong>{solution.execution_time:.4f} segundos</strong></p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="metric-card">
                            <h4>Solucionador</h4>
                            <p><strong>{solution.solver_info.get('solver_name', 'N/A')}</strong></p>
                        </div>
                    </div>
                </div>
        """
        
        # Agregar gráficos
        plot_id = 0
        for plot_name, plot_json in plots.items():
            if 'error' not in plot_name:
                html_content += f"""
                <div class="row">
                    <div class="col-12">
                        <div class="plot-container">
                            <div id="plot_{plot_id}"></div>
                            <script>
                                var plotData = {plot_json};
                                Plotly.newPlot('plot_{plot_id}', plotData.data, plotData.layout);
                            </script>
                        </div>
                    </div>
                </div>
                """
                plot_id += 1
        
        # Agregar tabla de variables
        if solution.variables_values:
            html_content += """
                <div class="row">
                    <div class="col-12">
                        <h3>Variables de Decisión</h3>
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Variable</th>
                                    <th>Valor</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            for var, value in solution.variables_values.items():
                html_content += f"""
                    <tr>
                        <td>{var}</td>
                        <td>{value}</td>
                    </tr>
                """
            
            html_content += """
                            </tbody>
                        </table>
                    </div>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content


class MatplotlibVisualizationService(IVisualizationService):
    """Servicio de visualización usando Matplotlib para gráficos estáticos"""
    
    def plot_solution(self, problem: OptimizationProblem, solution: Solution) -> Dict[str, Any]:
        """Genera gráficos estáticos de la solución"""
        plots = {}
        
        try:
            if solution.variables_values:
                # Gráfico de barras de variables
                plt.figure(figsize=(10, 6))
                variables = list(solution.variables_values.keys())
                values = list(solution.variables_values.values())
                
                # Filtrar valores numéricos
                numeric_vars = []
                numeric_values = []
                for var, val in zip(variables, values):
                    try:
                        numeric_val = float(val)
                        numeric_vars.append(var)
                        numeric_values.append(numeric_val)
                    except (ValueError, TypeError):
                        continue
                
                if numeric_vars:
                    plt.bar(range(len(numeric_vars)), numeric_values)
                    plt.xlabel('Variables')
                    plt.ylabel('Valores')
                    plt.title('Variables de Decisión - Solución Óptima')
                    plt.xticks(range(len(numeric_vars)), numeric_vars, rotation=45)
                    plt.tight_layout()
                    
                    # Convertir a base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plot_data = buffer.getvalue()
                    buffer.close()
                    plt.close()
                    
                    plots['variables_chart'] = base64.b64encode(plot_data).decode()
            
        except Exception as e:
            plots['error'] = str(e)
        
        return plots
    
    def plot_sensitivity_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera gráficos del análisis de sensibilidad"""
        # Implementación básica
        return {}
    
    def generate_report(self, problem: OptimizationProblem, solution: Solution) -> str:
        """Genera reporte básico en HTML"""
        return f"""
        <h1>Reporte de Optimización</h1>
        <h2>{problem.name}</h2>
        <p>Estado: {solution.status}</p>
        <p>Valor Objetivo: {solution.objective_value}</p>
        <p>Tiempo: {solution.execution_time:.4f}s</p>
        """
