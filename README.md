# Sistema de Optimización Empresarial

Un sistema completo para resolver problemas de optimización empresarial implementando técnicas de Investigación Operativa con arquitectura limpia.

## Características

- **Programación Lineal**: Optimización de recursos y producción
- **Problemas de Transporte**: Optimización de distribución y logística  
- **Problemas de Redes**: Optimización de flujos y rutas
- **Gestión de Inventario**: Optimización de stocks y costos
- **Programación Dinámica**: Optimización secuencial de decisiones
- **Análisis de Sensibilidad**: Con técnicas de IA para apoyar toma de decisiones
- **Interfaz Responsive**: Desarrollada con Bootstrap

## Arquitectura

El proyecto implementa Clean Architecture con separación clara de responsabilidades:

```
src/
├── core/
│   ├── entities/
│   ├── usecases/
│   └── interfaces/
├── infrastructure/
│   ├── repositories/
│   ├── services/
│   └── web/
└── presentation/
    ├── controllers/
    └── views/
```

## Instalación

```bash
pip install -r requirements.txt
python app.py
```

## Uso

Accede a `http://localhost:5000` para utilizar la interfaz web del sistema.

## Caso de Uso Empresarial

El sistema está aplicado a una empresa manufacturera que necesita optimizar:
- Producción de múltiples productos
- Distribución a diferentes centros
- Gestión de inventarios
- Planificación de recursos
