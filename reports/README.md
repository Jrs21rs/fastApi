# 📊 Reportes de Pruebas

Este directorio contiene los reportes generados de las pruebas unitarias.

## 📁 Estructura

```
reports/
├── test_report_YYYYMMDD_HHMMSS.html    # Reporte HTML de resultados de pruebas
├── test_report_YYYYMMDD_HHMMSS.pdf     # Reporte PDF (opcional)
├── coverage/                            # Reportes de cobertura de código
│   ├── index.html                       # Reporte HTML de cobertura
│   └── ...
└── coverage.xml                         # Reporte XML de cobertura
```

## 🚀 Generar Reportes

### Opción 1: Usando el script (Recomendado)

```bash
# Solo generar reporte HTML
python generate_reports.py

# Generar HTML + PDF
python generate_reports.py --pdf

# Generar y abrir en navegador
python generate_reports.py --open

# Todo: HTML + PDF + abrir navegador
python generate_reports.py --pdf --open
```

### Opción 2: Usando pytest directamente

```bash
# Generar reporte HTML
pytest --html=reports/test_report.html --self-contained-html

# Con cobertura
pytest --html=reports/test_report.html --self-contained-html --cov=app --cov-report=html:reports/coverage
```

## 📄 Ver los Reportes

### Reporte HTML de Pruebas
- Abre `reports/test_report_YYYYMMDD_HHMMSS.html` en tu navegador
- Contiene:
  - Resumen de resultados
  - Lista de todas las pruebas
  - Detalles de pruebas fallidas
  - Estadísticas de tiempo

### Reporte de Cobertura
- Abre `reports/coverage/index.html` en tu navegador
- Muestra:
  - Porcentaje de cobertura por archivo
  - Líneas cubiertas/no cubiertas
  - Gráficos y estadísticas

### Reporte PDF
- Abre `reports/test_report_YYYYMMDD_HHMMSS.pdf`
- Versión PDF del reporte HTML (útil para compartir o archivar)

## 📋 Información en los Reportes

### Reporte HTML de Pruebas
- ✅/❌ Estado de cada prueba
- ⏱️ Tiempo de ejecución
- 📝 Mensajes de error detallados
- 📊 Estadísticas generales

### Reporte de Cobertura
- 📈 Porcentaje de cobertura total
- 📁 Cobertura por archivo
- 📝 Líneas específicas no cubiertas
- 🎯 Métricas de calidad de código

## 🔧 Configuración

Los reportes se configuran en `pytest.ini`:

```ini
addopts = 
    --html=reports/test_report.html
    --self-contained-html
    --cov-report=html:reports/coverage
```

## 💡 Tips

1. **Reportes con timestamp**: El script genera reportes con fecha/hora para mantener historial
2. **Self-contained HTML**: Los reportes HTML incluyen todos los estilos, no necesitas archivos externos
3. **PDF opcional**: Solo se genera si usas `--pdf` y tienes weasyprint instalado
4. **Abrir automáticamente**: Usa `--open` para abrir el reporte en tu navegador predeterminado

## 🐛 Troubleshooting

### PDF no se genera
- Asegúrate de tener `weasyprint` instalado: `pip install weasyprint`
- O abre el HTML en el navegador e imprímelo como PDF (Ctrl+P)

### Reportes no se crean
- Verifica que el directorio `reports/` existe
- Revisa los permisos de escritura
- Ejecuta desde el directorio raíz del proyecto

