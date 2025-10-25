from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time
import os

app = FastAPI(title="Strabismus Model API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
interpreter = None
input_details = None
output_details = None
MODEL_LOADED = False

def load_model():
    global interpreter, input_details, output_details, MODEL_LOADED
    try:
        print("🔍 Buscando modelo...")

        # Buscar en todas las rutas posibles
        search_paths = [
            "modelos/strabismus_model.tflite",
            "../modelos/strabismus_model.tflite",
            "../../modelos/strabismus_model.tflite",
            "./strabismus_model.tflite",
            "modelos/estrabismo_model.tflite",
            "../modelos/estrabismo_model.tflite",
            "estrabismo_model.tflite"
        ]

        model_path = None
        for path in search_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Modelo encontrado: {path}")
                break

        if not model_path:
            print("❌ No se encontró el modelo. Rutas verificadas:")
            for path in search_paths:
                exists = "✅" if os.path.exists(path) else "❌"
                print(f"   {exists} {path}")
            return False

        print(f"📦 Cargando modelo: {model_path}")

        # Cargar modelo
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Obtener detalles de entrada y salida
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        MODEL_LOADED = True
        print("🎉 Modelo cargado exitosamente!")
        print(f"📊 Input shape: {input_details[0]['shape']}")
        print(f"📊 Input dtype: {input_details[0]['dtype']}")
        print(f"📊 Output shape: {output_details[0]['shape']}")
        print(f"🔧 TensorFlow version: {tf.__version__}")

        return True

    except Exception as e:
        print(f"💥 Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesar imagen al tamaño esperado por el modelo"""
    # Obtener el tamaño esperado del modelo
    expected_height = input_details[0]['shape'][1]
    expected_width = input_details[0]['shape'][2]

    print(f"🔄 Redimensionando a: {expected_width}x{expected_height}")

    # Redimensionar al tamaño esperado
    image = image.resize((expected_width, expected_height))
    image_array = np.array(image, dtype=np.float32)

    # Convertir RGBA a RGB si es necesario
    if len(image_array.shape) == 3 and image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Asegurar que sea RGB (3 canales)
    if len(image_array.shape) == 2:  # Si es escala de grises
        image_array = np.stack([image_array] * 3, axis=-1)
    elif len(image_array.shape) == 3 and image_array.shape[-1] == 1:  # Si es 1 canal
        image_array = np.concatenate([image_array] * 3, axis=-1)

    # Normalizar a [0, 1]
    image_array = image_array / 255.0

    # Agregar batch dimension [1, height, width, 3]
    image_array = np.expand_dims(image_array, axis=0)

    print(f" Imagen preprocesada: {image_array.shape}")
    return image_array

@app.on_event("startup")
async def startup_event():
    print(" Iniciando Strabismus Detection API...")
    success = load_model()
    if success:
        print(" API lista para recibir peticiones")
    else:
        print(" API iniciada pero modelo no cargado")

@app.post("/predict")
async def predict_strabismus(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    try:
        start_time = time.time()

        # Validaciones
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

        print(f"🖼️ Procesando: {file.filename} ({file.content_type})")

        # Leer y convertir imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        print(f"📐 Dimensiones originales: {image.size}")

        # Preprocesar
        input_data = preprocess_image(image)

        # Verificar que las dimensiones coincidan
        expected_shape = input_details[0]['shape']
        if input_data.shape != tuple(expected_shape):
            print(f"❌ Error de dimensiones: Esperado {expected_shape}, Obtenido {input_data.shape}")
            raise HTTPException(status_code=500, detail=f"Error de dimensiones del modelo")

        # Ejecutar inferencia
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Procesar resultado
        prediction = float(output_data[0][0])
        has_strabismus = prediction > 0.6
        processing_time = round((time.time() - start_time) * 1000, 2)

        # Resultado simple - solo si tiene o no estrabismo
        result = {
            "tieneEstrabismo": has_strabismus,
            "confianza": round(prediction, 4),
            "tiempoProcesamiento": f"{processing_time}ms",
            "mensaje": "Estrabismo detectado" if has_strabismus else "No se detectó estrabismo"
        }

        print(f"📊 Predicción completada: {result}")
        return result

    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_input_shape": input_details[0]['shape'] if MODEL_LOADED else None,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    return {
        "message": "Strabismus Detection API",
        "status": "running",
        "model_loaded": MODEL_LOADED,
        "model_input_shape": input_details[0]['shape'] if MODEL_LOADED else None,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)