/**
 * Componente de IA para análisis automático de problemas de optimización
 * Este componente puede ser incluido en cualquier página del sistema
 */

class AIOptimizationAssistant {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            problemType: options.problemType || null,
            autoSolve: options.autoSolve || false,
            onAnalysisComplete: options.onAnalysisComplete || this.defaultAnalysisHandler,
            onProblemGenerated: options.onProblemGenerated || this.defaultProblemHandler,
            onSolutionComplete: options.onSolutionComplete || this.defaultSolutionHandler,
            ...options
        };
        
        this.init();
    }
    
    init() {
        this.createHTML();
        this.attachEventListeners();
    }
    
    createHTML() {
        const html = `
            <div class="ai-assistant-container">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">
                            <i class="fas fa-robot me-2"></i>
                            Asistente IA de Optimización
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="ai-description" class="form-label">
                                <i class="fas fa-comment-alt me-2"></i>
                                Describe tu problema de optimización:
                            </label>
                            <textarea 
                                id="ai-description" 
                                class="form-control" 
                                rows="4" 
                                placeholder="Ejemplo: Una empresa tiene 3 fábricas que producen widgets. La fábrica A puede producir hasta 100 unidades por día, la fábrica B hasta 150 unidades, y la fábrica C hasta 200 unidades. El costo de producción es $10, $12 y $8 por unidad respectivamente. La empresa necesita producir al menos 300 unidades diarias para satisfacer la demanda. ¿Cómo debe distribuir la producción para minimizar costos?"
                            ></textarea>
                            <div class="form-text">
                                <i class="fas fa-lightbulb text-warning me-1"></i>
                                <strong>Tip:</strong> Sé específico con números, restricciones y objetivos. 
                                La IA analizará tu descripción automáticamente.
                            </div>
                        </div>
                        
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <label for="ai-problem-type" class="form-label">Tipo de Problema (Opcional)</label>
                                <select id="ai-problem-type" class="form-control">
                                    <option value="">Detectar automáticamente</option>
                                    <option value="linear_programming">Programación Lineal</option>
                                    <option value="transportation">Transporte</option>
                                    <option value="network">Redes</option>
                                    <option value="inventory">Inventario</option>
                                    <option value="dynamic_programming">Programación Dinámica</option>
                                </select>
                            </div>
                            <div class="col-md-6 d-flex align-items-end">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="ai-auto-solve" checked>
                                    <label class="form-check-label" for="ai-auto-solve">
                                        Resolver automáticamente
                                    </label>
                                </div>
                            </div>
                        </div>
                        
                        <div class="d-flex justify-content-between">
                            <button type="button" class="btn btn-outline-primary" id="ai-analyze-btn">
                                <i class="fas fa-search me-2"></i>Analizar Descripción
                            </button>
                            <button type="button" class="btn btn-primary" id="ai-generate-btn">
                                <i class="fas fa-magic me-2"></i>Generar y Resolver
                            </button>
                        </div>
                        
                        <!-- Loading Spinner -->
                        <div id="ai-loading" class="text-center mt-3" style="display: none;">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Analizando...</span>
                            </div>
                            <p class="mt-2">La IA está analizando tu problema...</p>
                        </div>
                        
                        <!-- Results Container -->
                        <div id="ai-results" class="mt-4" style="display: none;"></div>
                    </div>
                </div>
            </div>
        `;
        
        this.container.innerHTML = html;
    }
    
    attachEventListeners() {
        const analyzeBtn = document.getElementById('ai-analyze-btn');
        const generateBtn = document.getElementById('ai-generate-btn');
        
        analyzeBtn.addEventListener('click', () => this.analyzeDescription());
        generateBtn.addEventListener('click', () => this.generateAndSolve());
        
        // Auto-resize textarea
        const textarea = document.getElementById('ai-description');
        textarea.addEventListener('input', this.autoResize);
    }
    
    autoResize(event) {
        const textarea = event.target;
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }
    
    async analyzeDescription() {
        const description = document.getElementById('ai-description').value.trim();
        const problemType = document.getElementById('ai-problem-type').value;
        
        if (!description) {
            this.showError('Por favor, describe tu problema antes de analizar.');
            return;
        }
        
        this.showLoading('Analizando descripción...');
        
        try {
            const response = await fetch('/ai_analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    description: description,
                    problem_type: problemType || null
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayAnalysis(data.analysis);
                this.options.onAnalysisComplete(data.analysis);
            } else {
                this.showError(data.message);
            }
        } catch (error) {
            this.showError('Error al conectar con el servidor: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }
    
    async generateAndSolve() {
        const description = document.getElementById('ai-description').value.trim();
        const problemType = document.getElementById('ai-problem-type').value;
        const autoSolve = document.getElementById('ai-auto-solve').checked;
        
        if (!description) {
            this.showError('Por favor, describe tu problema antes de generar.');
            return;
        }
        
        this.showLoading('Generando problema y resolviendo...');
        
        try {
            const response = await fetch('/ai_generate_problem', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    description: description,
                    problem_type: problemType || null,
                    auto_solve: autoSolve
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayFullResults(data);
                
                if (data.analysis) {
                    this.options.onAnalysisComplete(data.analysis);
                }
                
                if (data.problem_generated) {
                    this.options.onProblemGenerated(data);
                }
                
                if (data.problem_solved && data.solution) {
                    this.options.onSolutionComplete(data.solution, data);
                }
            } else {
                this.showError(data.message);
            }
        } catch (error) {
            this.showError('Error al conectar con el servidor: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }
    
    displayAnalysis(analysis) {
        const resultsContainer = document.getElementById('ai-results');
        
        const html = `
            <div class="analysis-results">
                <h6><i class="fas fa-chart-line me-2"></i>Análisis del Problema</h6>
                
                <div class="row mb-3">
                    <div class="col-md-6">
                        <div class="alert alert-info">
                            <strong>Tipo Detectado:</strong> ${this.translateProblemType(analysis.problem_type)}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="alert alert-${this.getConfidenceClass(analysis.confidence)}">
                            <strong>Confianza:</strong> ${(analysis.confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
                
                <!-- Variables -->
                <div class="mb-3">
                    <h6><i class="fas fa-x me-2"></i>Variables de Decisión (${analysis.variables.length})</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-bordered">
                            <thead>
                                <tr>
                                    <th>Variable</th>
                                    <th>Descripción</th>
                                    <th>Límites</th>
                                    <th>Tipo</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${analysis.variables.map(v => `
                                    <tr>
                                        <td><code>${v.name}</code></td>
                                        <td>${v.description || 'N/A'}</td>
                                        <td>[${v.lower_bound}, ${v.upper_bound || '∞'}]</td>
                                        <td><span class="badge bg-secondary">${v.var_type}</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Función Objetivo -->
                <div class="mb-3">
                    <h6><i class="fas fa-target me-2"></i>Función Objetivo</h6>
                    <div class="alert alert-light">
                        <strong>${this.translateObjectiveSense(analysis.objective.sense)}:</strong>
                        ${Object.entries(analysis.objective.variables).map(([var_name, coeff]) => 
                            `${coeff}·${var_name}`
                        ).join(' + ')}
                    </div>
                </div>
                
                <!-- Restricciones -->
                <div class="mb-3">
                    <h6><i class="fas fa-lock me-2"></i>Restricciones (${analysis.constraints.length})</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-bordered">
                            <thead>
                                <tr>
                                    <th>Restricción</th>
                                    <th>Expresión</th>
                                    <th>Descripción</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${analysis.constraints.map(c => `
                                    <tr>
                                        <td><code>${c.name}</code></td>
                                        <td>
                                            ${Object.entries(c.variables).map(([var_name, coeff]) => 
                                                `${coeff}·${var_name}`
                                            ).join(' + ')} ${c.operator} ${c.rhs}
                                        </td>
                                        <td>${c.description || 'N/A'}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Sugerencias -->
                ${analysis.suggestions.length > 0 ? `
                <div class="mb-3">
                    <h6><i class="fas fa-lightbulb me-2"></i>Sugerencias de la IA</h6>
                    <ul class="list-group">
                        ${analysis.suggestions.map(suggestion => `
                            <li class="list-group-item">
                                <i class="fas fa-check-circle text-success me-2"></i>
                                ${suggestion}
                            </li>
                        `).join('')}
                    </ul>
                </div>
                ` : ''}
            </div>
        `;
        
        resultsContainer.innerHTML = html;
        resultsContainer.style.display = 'block';
    }
    
    displayFullResults(data) {
        this.displayAnalysis(data.analysis);
        
        if (data.solution) {
            this.appendSolutionResults(data.solution, data);
        }
    }
    
    appendSolutionResults(solution, fullData) {
        const resultsContainer = document.getElementById('ai-results');
        
        const solutionHtml = `
            <div class="solution-results mt-4">
                <hr>
                <h6><i class="fas fa-check-circle me-2"></i>Resultados de la Solución</h6>
                
                <div class="row mb-3">
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h5 class="card-title">Estado</h5>
                                <span class="badge bg-${solution.status === 'optimal' ? 'success' : 'warning'} fs-6">
                                    ${solution.status}
                                </span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h5 class="card-title">Valor Objetivo</h5>
                                <p class="card-text h4 text-primary">
                                    ${solution.objective_value ? solution.objective_value.toFixed(2) : 'N/A'}
                                </p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h5 class="card-title">Tiempo</h5>
                                <p class="card-text">${solution.execution_time.toFixed(4)}s</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h5 class="card-title">Solucionador</h5>
                                <p class="card-text small">${solution.solver_info.solver_name || 'N/A'}</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Variables Values -->
                <div class="mb-3">
                    <h6><i class="fas fa-list me-2"></i>Valores de Variables</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    <th>Variable</th>
                                    <th>Valor</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${Object.entries(solution.variables_values).map(([varName, value]) => `
                                    <tr>
                                        <td><code>${varName}</code></td>
                                        <td>${typeof value === 'number' ? value.toFixed(4) : value}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- AI Recommendations -->
                ${fullData.ai_recommendations && fullData.ai_recommendations.length > 0 ? `
                <div class="mb-3">
                    <h6><i class="fas fa-robot me-2"></i>Recomendaciones IA Adicionales</h6>
                    <ul class="list-group">
                        ${fullData.ai_recommendations.map(rec => `
                            <li class="list-group-item">
                                <i class="fas fa-arrow-right text-primary me-2"></i>
                                ${rec}
                            </li>
                        `).join('')}
                    </ul>
                </div>
                ` : ''}
                
                <div class="text-center mt-3">
                    <button class="btn btn-success" onclick="window.aiAssistant.exportResults()">
                        <i class="fas fa-download me-2"></i>Exportar Resultados
                    </button>
                </div>
            </div>
        `;
        
        resultsContainer.innerHTML += solutionHtml;
    }
    
    showLoading(message = 'Procesando...') {
        const loadingDiv = document.getElementById('ai-loading');
        loadingDiv.querySelector('p').textContent = message;
        loadingDiv.style.display = 'block';
    }
    
    hideLoading() {
        document.getElementById('ai-loading').style.display = 'none';
    }
    
    showError(message) {
        const resultsContainer = document.getElementById('ai-results');
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Error:</strong> ${message}
            </div>
        `;
        resultsContainer.style.display = 'block';
    }
    
    translateProblemType(type) {
        const translations = {
            'linear_programming': 'Programación Lineal',
            'transportation': 'Problemas de Transporte',
            'network': 'Problemas de Redes',
            'inventory': 'Gestión de Inventario',
            'dynamic_programming': 'Programación Dinámica',
            'knapsack': 'Problema de la Mochila'
        };
        return translations[type] || type;
    }
    
    translateObjectiveSense(sense) {
        return sense === 'minimize' ? 'Minimizar' : 'Maximizar';
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'success';
        if (confidence >= 0.6) return 'warning';
        return 'danger';
    }
    
    exportResults() {
        const resultsContent = document.getElementById('ai-results').innerHTML;
        const blob = new Blob([`
            <!DOCTYPE html>
            <html>
            <head>
                <title>Resultados de Optimización IA</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            </head>
            <body class="container mt-4">
                <h1>Resultados de Optimización con IA</h1>
                <hr>
                ${resultsContent}
            </body>
            </html>
        `], { type: 'text/html' });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'resultados_optimizacion_ia.html';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    // Default handlers
    defaultAnalysisHandler(analysis) {
        console.log('Análisis completado:', analysis);
    }
    
    defaultProblemHandler(data) {
        console.log('Problema generado:', data);
    }
    
    defaultSolutionHandler(solution, fullData) {
        console.log('Solución completada:', solution);
    }
    
    // Utility methods
    populateFormFromAnalysis(analysis) {
        // Esta función puede ser sobrescrita para poblar formularios específicos
        // basándose en el análisis de IA
        console.log('Populando formulario con análisis IA:', analysis);
    }
    
    clear() {
        document.getElementById('ai-description').value = '';
        document.getElementById('ai-problem-type').value = '';
        document.getElementById('ai-auto-solve').checked = true;
        document.getElementById('ai-results').style.display = 'none';
    }
}

// Make it globally available
window.AIOptimizationAssistant = AIOptimizationAssistant;
