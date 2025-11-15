# Pruebas - Detección de Estrabismo

Este directorio contiene las pruebas unitarias y de integración para la API de detección de estrabismo.

## Estructura

- `conftest.py`: Configuración y fixtures compartidas para pytest
- `test_main.py`: Pruebas unitarias para todas las funciones y endpoints
- `test_integration.py`: Pruebas de integración para flujos completos de la aplicación
- `__init__.py`: Archivo de inicialización del paquete de tests

## Instalación de Dependencias

Asegúrate de tener todas las dependencias instaladas:

```bash
pip install -r app/requirements.txt
```

## Ejecutar las Pruebas

### Ejecutar todas las pruebas (unitarias + integración)

```bash
pytest
```

O usando el script:

```bash
python run_tests.py
```

### Ejecutar solo pruebas unitarias

```bash
pytest -m unit
```

O:

```bash
pytest tests/test_main.py
```

### Ejecutar solo pruebas de integración

```bash
pytest -m integration
```

O usando el script dedicado:

```bash
python run_integration_tests.py
```

### Ejecutar con cobertura de código

```bash
pytest --cov=app --cov-report=html
```

Esto generará un reporte HTML en `reports/coverage/index.html`

### Ejecutar pruebas específicas

```bash
# Ejecutar una clase de tests específica
pytest tests/test_main.py::TestLoadModel

# Ejecutar un test específico
pytest tests/test_main.py::TestLoadModel::test_load_model_success

# Ejecutar tests con un patrón
pytest -k "test_load_model"

# Ejecutar una clase de pruebas de integración
pytest tests/test_integration.py::TestModelIntegration
```

### Ejecutar con más verbosidad

```bash
pytest -v
```

### Ejecutar con salida detallada

```bash
pytest -vv
```

### Excluir pruebas lentas

```bash
pytest -m "not slow"
```

## Cobertura de Tests

### Pruebas Unitarias (`test_main.py`)

Las pruebas unitarias cubren:

#### Funciones
- ✅ `load_model()`: Carga del modelo TensorFlow Lite
- ✅ `preprocess_image()`: Preprocesamiento de imágenes
- ✅ `guardar_en_springboot()`: Envío de resultados al backend

#### Endpoints
- ✅ `POST /predict/{documento_identidad}`: Endpoint de predicción
- ✅ `GET /health`: Health check
- ✅ `GET /`: Endpoint raíz

### Pruebas de Integración (`test_integration.py`)

Las pruebas de integración cubren:

#### Integración con el Modelo
- ✅ Flujo completo de predicción con imágenes reales
- ✅ Procesamiento de diferentes formatos de imagen (PNG, JPEG)
- ✅ Procesamiento de imágenes de diferentes tamaños
- ✅ Carga del modelo real si está disponible

#### Integración de API
- ✅ Flujo completo de endpoints
- ✅ Health check en contexto de integración
- ✅ Manejo de parámetros (documento_identidad, save_result)
- ✅ Validación de respuestas completas

#### Integración con Spring Boot
- ✅ Envío exitoso de resultados al backend
- ✅ Manejo de errores de conexión
- ✅ Flujo completo con guardado en Spring Boot
- ✅ Recuperación ante errores del backend

#### Pruebas End-to-End
- ✅ Flujo completo de usuario (health → predicción → guardado)
- ✅ Múltiples predicciones secuenciales
- ✅ Recuperación de errores
- ✅ Pruebas de rendimiento
- ✅ Solicitudes concurrentes

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

## Tipos de Pruebas

### Pruebas Unitarias
- Utilizan mocks extensivos para aislar componentes
- No requieren el modelo real de TensorFlow
- Ejecución rápida
- Cubren casos específicos y edge cases

### Pruebas de Integración
- Prueban flujos completos de la aplicación
- Intentan usar el modelo real si está disponible (fallback a mock)
- Validan integraciones entre componentes
- Incluyen pruebas de rendimiento y concurrencia
- Marcadas con `@pytest.mark.integration`

### Pruebas Lentas
- Algunas pruebas de integración están marcadas como `@pytest.mark.slow`
- Pueden excluirse con `pytest -m "not slow"`

## Notas

- **Pruebas Unitarias**: Utilizan mocks para evitar depender del modelo real de TensorFlow
- **Pruebas de Integración**: Intentan usar el modelo real si está disponible en `modelos/strabismus_model.tflite`
- Se utilizan fixtures de pytest para reutilizar código común
- Los tests están diseñados para ejecutarse de forma aislada
- Las pruebas de integración pueden tardar más tiempo en ejecutarse

## Troubleshooting

Si encuentras errores al ejecutar las pruebas:

1. Asegúrate de que todas las dependencias estén instaladas
2. Verifica que estés en el directorio raíz del proyecto
3. Revisa que el path de Python incluya el directorio `app`

