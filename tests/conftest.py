"""
Configuración y fixtures compartidas para las pruebas
"""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, MagicMock, patch
import io
import sys
import os

# Mock TensorFlow ANTES de cualquier importación para evitar carga lenta
# Solo se aplica si TensorFlow no está instalado o para pruebas unitarias
# Las pruebas de integración pueden usar el modelo real si está disponible
try:
    import tensorflow as tf
    # TensorFlow está disponible, no hacer mock
    TF_AVAILABLE = True
except ImportError:
    # TensorFlow no está disponible, usar mock
    _mock_tf = MagicMock()
    _mock_tf.lite = MagicMock()
    _mock_tf.lite.Interpreter = MagicMock()
    _mock_tf.__version__ = "2.15.0"
    
    sys.modules['tensorflow'] = _mock_tf
    sys.modules['tensorflow.lite'] = _mock_tf.lite
    sys.modules['tensorflow.lite.Interpreter'] = _mock_tf.lite.Interpreter
    TF_AVAILABLE = False

# Agregar el directorio app al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

@pytest.fixture
def sample_image():
    """Crea una imagen de prueba RGB"""
    img = Image.new('RGB', (224, 224), color='red')
    return img

@pytest.fixture
def sample_image_rgba():
    """Crea una imagen de prueba RGBA"""
    img = Image.new('RGBA', (224, 224), color=(255, 0, 0, 255))
    return img

@pytest.fixture
def sample_image_grayscale():
    """Crea una imagen de prueba en escala de grises"""
    img = Image.new('L', (224, 224), color=128)
    return img

@pytest.fixture
def sample_image_bytes(sample_image):
    """Convierte una imagen a bytes para simular upload"""
    img_byte_arr = io.BytesIO()
    sample_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.read()

@pytest.fixture
def mock_interpreter():
    """Mock del TensorFlow Lite Interpreter"""
    interpreter = MagicMock()
    interpreter.get_input_details.return_value = [{
        'shape': np.array([1, 224, 224, 3]),
        'dtype': np.float32,
        'index': 0
    }]
    interpreter.get_output_details.return_value = [{
        'shape': np.array([1, 1]),
        'dtype': np.float32,
        'index': 0
    }]
    interpreter.get_tensor.return_value = np.array([[0.7]], dtype=np.float32)
    return interpreter

@pytest.fixture
def mock_tflite_interpreter(mock_interpreter):
    """Mock de tf.lite.Interpreter"""
    with patch('tensorflow.lite.Interpreter') as mock_interpreter_class:
        mock_interpreter_class.return_value = mock_interpreter
        yield mock_interpreter_class

@pytest.fixture
def mock_requests():
    """Mock del módulo requests"""
    with patch('requests.post') as mock_post:
        yield mock_post

@pytest.fixture
def app_client():
    """Cliente de prueba para la aplicación FastAPI"""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)

@pytest.fixture
def reset_model_state():
    """Fixture para resetear el estado del modelo entre tests"""
    import main
    main.interpreter = None
    main.input_details = None
    main.output_details = None
    main.MODEL_LOADED = False
    yield
    # Cleanup después del test
    main.interpreter = None
    main.input_details = None
    main.output_details = None
    main.MODEL_LOADED = False

