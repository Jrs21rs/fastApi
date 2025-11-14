# Pruebas Unitarias - Detección de Estrabismo

Este directorio contiene las pruebas unitarias para la API de detección de estrabismo.

## Estructura

- `conftest.py`: Configuración y fixtures compartidas para pytest
- `test_main.py`: Pruebas unitarias para todas las funciones y endpoints
- `__init__.py`: Archivo de inicialización del paquete de tests

## Instalación de Dependencias

Asegúrate de tener todas las dependencias instaladas:

```bash
pip install -r app/requirements.txt
```

## Ejecutar las Pruebas

### Ejecutar todas las pruebas

```bash
pytest
```

### Ejecutar con cobertura de código

```bash
pytest --cov=app --cov-report=html
```

Esto generará un reporte HTML en `htmlcov/index.html`

### Ejecutar pruebas específicas

```bash
# Ejecutar una clase de tests específica
pytest tests/test_main.py::TestLoadModel

# Ejecutar un test específico
pytest tests/test_main.py::TestLoadModel::test_load_model_success

# Ejecutar tests con un patrón
pytest -k "test_load_model"
```

### Ejecutar con más verbosidad

```bash
pytest -v
```

### Ejecutar con salida detallada

```bash
pytest -vv
```

## Cobertura de Tests

Las pruebas cubren:

### Funciones
- ✅ `load_model()`: Carga del modelo TensorFlow Lite
- ✅ `preprocess_image()`: Preprocesamiento de imágenes
- ✅ `guardar_en_springboot()`: Envío de resultados al backend

### Endpoints
- ✅ `POST /predict/{documento_identidad}`: Endpoint de predicción
- ✅ `GET /health`: Health check
- ✅ `GET /`: Endpoint raíz

### Casos de Prueba Incluidos

1. **Carga de Modelo**
   - Carga exitosa del modelo
   - Modelo no encontrado
   - Manejo de excepciones
   - Verificación de múltiples rutas

2. **Preprocesamiento de Imágenes**
   - Imágenes RGB
   - Imágenes RGBA (conversión a RGB)
   - Imágenes en escala de grises
   - Diferentes tamaños de imagen
   - Normalización correcta

3. **Guardado en Spring Boot**
   - Envío exitoso
   - Manejo de errores HTTP
   - Manejo de errores de conexión
   - Manejo de timeouts

4. **Endpoints de la API**
   - Predicción exitosa
   - Predicción sin estrabismo
   - Modelo no cargado
   - Archivos inválidos
   - Errores de procesamiento
   - Health check con y sin modelo

5. **Casos Límite**
   - Desajuste de dimensiones
   - Errores de procesamiento
   - Imágenes con diferentes formatos

## Notas

- Los tests utilizan mocks para evitar depender del modelo real de TensorFlow
- Las pruebas no requieren que el modelo esté presente en el sistema
- Se utilizan fixtures de pytest para reutilizar código común
- Los tests están diseñados para ejecutarse de forma aislada

## Troubleshooting

Si encuentras errores al ejecutar las pruebas:

1. Asegúrate de que todas las dependencias estén instaladas
2. Verifica que estés en el directorio raíz del proyecto
3. Revisa que el path de Python incluya el directorio `app`

