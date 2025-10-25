# 🤖 Strabismus Detection API - FastAPI

API de detección de estrabismo usando TensorFlow Lite y FastAPI.

## 📋 Descripción

Esta API proporciona un endpoint para detectar estrabismo en imágenes de ojos usando un modelo de machine learning entrenado con TensorFlow Lite.

## 🚀 Despliegue en Render

### Opción 1: Despliegue automático

1. Haz push de este código a GitHub
2. Ve a [Render Dashboard](https://dashboard.render.com/)
3. Click en **"New +"** → **"Web Service"**
4. Conecta tu repositorio
5. Render detectará automáticamente el `Dockerfile`
6. Configura las variables de entorno:
   - `PORT=5000`
   - `PYTHONUNBUFFERED=1`
7. Click en **"Create Web Service"**

### Opción 2: Usando render.yaml

El archivo `render.yaml` está configurado para despliegue automático.

## 📦 Estructura del Proyecto

```
fastApi/
├── app/
│   ├── main.py              # Aplicación FastAPI principal
│   └── requirements.txt     # Dependencias Python
├── modelos/
│   └── strabismus_model.tflite  # Modelo TensorFlow Lite
├── Dockerfile               # Configuración Docker
├── render.yaml             # Configuración Render
└── README.md               # Este archivo
```

## 🔧 Instalación Local

### Requisitos
- Python 3.10+
- pip

### Pasos

```bash
# 1. Navegar a la carpeta
cd fastApi

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r app/requirements.txt

# 5. Ejecutar la aplicación
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

La API estará disponible en: `http://localhost:5000`

## 📡 Endpoints

### 1. **GET /** - Información de la API
```bash
curl http://localhost:5000/
```

**Respuesta:**
```json
{
  "message": "Strabismus Detection API",
  "status": "running",
  "model_loaded": true,
  "model_input_shape": [1, 224, 224, 3],
  "version": "1.0.0"
}
```

### 2. **GET /health** - Health Check
```bash
curl http://localhost:5000/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_input_shape": [1, 224, 224, 3],
  "timestamp": 1729812345.67
}
```

### 3. **POST /predict** - Predicción de Estrabismo
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@imagen_ojo.jpg"
```

**Respuesta:**
```json
{
  "tieneEstrabismo": true,
  "confianza": 0.8542,
  "tiempoProcesamiento": "245.67ms",
  "mensaje": "Estrabismo detectado"
}
```

## 🧪 Pruebas con Postman

1. Importa la colección `Postman_Collection.json` (en la raíz del proyecto)
2. Configura la variable `FASTAPI_URL` con tu URL de Render
3. Prueba los endpoints

## 🐳 Docker

### Construir imagen
```bash
docker build -t strabismus-api .
```

### Ejecutar contenedor
```bash
docker run -p 5000:5000 strabismus-api
```

## 📊 Modelo

- **Tipo**: TensorFlow Lite
- **Input**: Imagen RGB de 224x224 píxeles
- **Output**: Probabilidad de estrabismo (0-1)
- **Threshold**: 0.6 (valores > 0.6 indican estrabismo)

## ⚙️ Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `PORT` | Puerto de la aplicación | `5000` |
| `PYTHONUNBUFFERED` | Logs sin buffer | `1` |

## 🔒 CORS

La API tiene CORS habilitado para todos los orígenes (`*`). En producción, considera restringir a dominios específicos.

## 📝 Logs

Los logs incluyen:
- ✅ Modelo cargado correctamente
- 🖼️ Procesamiento de imágenes
- 📊 Resultados de predicción
- ❌ Errores y excepciones

## 🐛 Troubleshooting

### El modelo no se carga
- Verifica que `modelos/strabismus_model.tflite` exista
- Revisa los logs para ver la ruta buscada
- Asegúrate de que el archivo no esté corrupto

### Error de dimensiones
- El modelo espera imágenes de 224x224x3
- La API redimensiona automáticamente, pero verifica el formato

### Error 500 en /predict
- Verifica que la imagen sea válida (JPG, PNG, etc.)
- Revisa los logs para más detalles

## 📚 Dependencias

- **FastAPI**: Framework web moderno y rápido
- **Uvicorn**: Servidor ASGI
- **TensorFlow**: Para ejecutar el modelo TFLite
- **Pillow**: Procesamiento de imágenes
- **NumPy**: Operaciones numéricas

## 🚀 Mejoras Futuras

- [ ] Agregar autenticación con API Keys
- [ ] Implementar rate limiting
- [ ] Cachear resultados
- [ ] Agregar más endpoints (batch prediction)
- [ ] Métricas con Prometheus
- [ ] Documentación interactiva con Swagger UI

## 📄 Licencia

Este proyecto es parte del sistema de Detección de Estrabismo.

## 👥 Contacto

Para soporte o preguntas, contacta al equipo de desarrollo.
