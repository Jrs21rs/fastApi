"""
Pruebas de integración para la aplicación de detección de estrabismo

Estas pruebas validan el flujo completo de la aplicación, incluyendo:
- Carga del modelo
- Procesamiento de imágenes
- Integración con endpoints
- Integración con Spring Boot
"""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import io
import os
import sys
import json
from fastapi.testclient import TestClient
import requests
from datetime import datetime

# Agregar el directorio app al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import main


@pytest.mark.integration
class TestModelIntegration:
    """Pruebas de integración con el modelo de TensorFlow Lite"""

    @pytest.fixture(autouse=True)
    def setup_model(self, reset_model_state):
        """Configura el modelo para las pruebas de integración"""
        # Intentar cargar el modelo real si existe
        model_paths = [
            "modelos/strabismus_model.tflite",
            "../modelos/strabismus_model.tflite",
            "../../modelos/strabismus_model.tflite",
        ]
        
        model_found = False
        for path in model_paths:
            if os.path.exists(path):
                # Cargar modelo real
                try:
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=path)
                    interpreter.allocate_tensors()
                    
                    main.interpreter = interpreter
                    main.input_details = interpreter.get_input_details()
                    main.output_details = interpreter.get_output_details()
                    main.MODEL_LOADED = True
                    model_found = True
                    break
                except Exception as e:
                    print(f"Error cargando modelo real: {e}")
                    continue
        
        if not model_found:
            # Usar mock si no hay modelo real
            mock_interpreter = MagicMock()
            mock_interpreter.get_input_details.return_value = [{
                'shape': np.array([1, 224, 224, 3]),
                'dtype': np.float32,
                'index': 0
            }]
            mock_interpreter.get_output_details.return_value = [{
                'shape': np.array([1, 1]),
                'dtype': np.float32,
                'index': 0
            }]
            mock_interpreter.get_tensor.return_value = np.array([[0.7]], dtype=np.float32)
            
            main.interpreter = mock_interpreter
            main.input_details = [{
                'shape': np.array([1, 224, 224, 3]),
                'dtype': np.float32,
                'index': 0
            }]
            main.output_details = [{
                'shape': np.array([1, 1]),
                'dtype': np.float32,
                'index': 0
            }]
            main.MODEL_LOADED = True
        
        yield
        
        # Cleanup
        main.interpreter = None
        main.input_details = None
        main.output_details = None
        main.MODEL_LOADED = False

    def test_full_prediction_flow(self, app_client):
        """Test del flujo completo de predicción con imagen real"""
        # Crear una imagen de prueba
        img = Image.new('RGB', (500, 400), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test_image.png", img_bytes.read(), "image/png")},
            params={"save_result": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verificar estructura de respuesta
        assert "tieneEstrabismo" in data
        assert "confianza" in data
        assert "tiempoProcesamiento" in data
        assert "mensaje" in data
        
        # Verificar tipos de datos
        assert isinstance(data["tieneEstrabismo"], bool)
        assert isinstance(data["confianza"], (int, float))
        assert 0 <= data["confianza"] <= 1
        assert "ms" in data["tiempoProcesamiento"]

    def test_prediction_with_different_image_formats(self, app_client):
        """Test que la API acepta diferentes formatos de imagen"""
        formats = ['PNG', 'JPEG']
        
        for fmt in formats:
            img = Image.new('RGB', (300, 300), color='green')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=fmt)
            img_bytes.seek(0)
            
            response = app_client.post(
                "/predict/987654321",
                files={"file": (f"test.{fmt.lower()}", img_bytes.read(), f"image/{fmt.lower()}")},
                params={"save_result": False}
            )
            
            assert response.status_code == 200, f"Formato {fmt} debería ser aceptado"
            data = response.json()
            assert "tieneEstrabismo" in data

    def test_prediction_with_different_image_sizes(self, app_client):
        """Test que la API procesa imágenes de diferentes tamaños"""
        sizes = [(100, 100), (224, 224), (500, 500), (1000, 800)]
        
        for width, height in sizes:
            img = Image.new('RGB', (width, height), color='red')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = app_client.post(
                "/predict/111222333",
                files={"file": ("test.png", img_bytes.read(), "image/png")},
                params={"save_result": False}
            )
            
            assert response.status_code == 200, f"Tamaño {width}x{height} debería ser procesado"
            data = response.json()
            assert "tieneEstrabismo" in data


@pytest.mark.integration
class TestAPIIntegration:
    """Pruebas de integración de los endpoints de la API"""

    @pytest.fixture(autouse=True)
    def setup_api(self, reset_model_state):
        """Configura el estado de la API para las pruebas"""
        # Configurar modelo mock para pruebas de API
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [{
            'shape': np.array([1, 224, 224, 3]),
            'dtype': np.float32,
            'index': 0
        }]
        mock_interpreter.get_output_details.return_value = [{
            'shape': np.array([1, 1]),
            'dtype': np.float32,
            'index': 0
        }]
        mock_interpreter.get_tensor.return_value = np.array([[0.65]], dtype=np.float32)
        
        main.interpreter = mock_interpreter
        main.input_details = [{
            'shape': np.array([1, 224, 224, 3]),
            'dtype': np.float32,
            'index': 0
        }]
        main.output_details = [{
            'shape': np.array([1, 1]),
            'dtype': np.float32,
            'index': 0
        }]
        main.MODEL_LOADED = True
        
        yield
        
        # Cleanup
        main.interpreter = None
        main.input_details = None
        main.output_details = None
        main.MODEL_LOADED = False

    def test_health_check_integration(self, app_client):
        """Test del endpoint de health check en flujo completo"""
        response = app_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_root_endpoint_integration(self, app_client):
        """Test del endpoint raíz en flujo completo"""
        response = app_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Strabismus Detection API"
        assert data["status"] == "running"
        assert data["model_loaded"] is True
        assert "version" in data
        assert "model_input_shape" in data

    def test_predict_endpoint_with_documento_identidad(self, app_client):
        """Test que el endpoint de predicción maneja correctamente el documento de identidad"""
        img = Image.new('RGB', (224, 224), color='yellow')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        documento_identidad = 1234567890
        
        response = app_client.post(
            f"/predict/{documento_identidad}",
            files={"file": ("test.png", img_bytes.read(), "image/png")},
            params={"save_result": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "tieneEstrabismo" in data

    def test_predict_endpoint_save_result_parameter(self, app_client):
        """Test que el parámetro save_result controla si se guarda en Spring Boot"""
        img = Image.new('RGB', (224, 224), color='purple')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        with patch('main.guardar_en_springboot') as mock_guardar:
            # Test con save_result=True
            response = app_client.post(
                "/predict/123456789",
                files={"file": ("test.png", img_bytes.read(), "image/png")},
                params={"save_result": True}
            )
            
            assert response.status_code == 200
            mock_guardar.assert_called_once()
            
            # Reset mock
            mock_guardar.reset_mock()
            img_bytes.seek(0)
            
            # Test con save_result=False
            response = app_client.post(
                "/predict/123456789",
                files={"file": ("test.png", img_bytes.read(), "image/png")},
                params={"save_result": False}
            )
            
            assert response.status_code == 200
            mock_guardar.assert_not_called()


@pytest.mark.integration
class TestSpringBootIntegration:
    """Pruebas de integración con el backend de Spring Boot"""

    def test_guardar_en_springboot_success_integration(self):
        """Test de integración exitosa con Spring Boot"""
        resultado = {
            "tieneEstrabismo": True,
            "confianza": 0.85
        }
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_post.return_value = mock_response
            
            result = main.guardar_en_springboot(123456789, resultado)
            
            assert result is True
            mock_post.assert_called_once()
            
            # Verificar que se llamó con los parámetros correctos
            call_args = mock_post.call_args
            assert call_args[1]['json']['documentoIdentidad'] == 123456789
            assert call_args[1]['json']['resultado'] is True
            assert call_args[1]['json']['confianzaPrediccion'] == 0.85
            assert 'fechaEvaluacion' in call_args[1]['json']
            assert call_args[1]['headers']['Content-Type'] == 'application/json'

    def test_guardar_en_springboot_with_full_flow(self, app_client):
        """Test del flujo completo incluyendo guardado en Spring Boot"""
        img = Image.new('RGB', (224, 224), color='orange')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        with patch('main.guardar_en_springboot') as mock_guardar:
            mock_guardar.return_value = True
            
            # Configurar modelo
            mock_interpreter = MagicMock()
            mock_interpreter.get_tensor.return_value = np.array([[0.75]], dtype=np.float32)
            main.interpreter = mock_interpreter
            main.input_details = [{
                'shape': np.array([1, 224, 224, 3]),
                'index': 0
            }]
            main.output_details = [{'index': 0}]
            main.MODEL_LOADED = True
            
            response = app_client.post(
                "/predict/555666777",
                files={"file": ("test.png", img_bytes.read(), "image/png")},
                params={"save_result": True}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "tieneEstrabismo" in data
            
            # Verificar que se intentó guardar
            mock_guardar.assert_called_once()
            call_args = mock_guardar.call_args[0]
            assert call_args[0] == 555666777
            assert call_args[1]["tieneEstrabismo"] == data["tieneEstrabismo"]

    def test_springboot_integration_error_handling(self, app_client):
        """Test que los errores de Spring Boot no afectan la respuesta de la API"""
        img = Image.new('RGB', (224, 224), color='cyan')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        with patch('main.guardar_en_springboot') as mock_guardar:
            mock_guardar.return_value = False  # Simular error en Spring Boot
            
            # Configurar modelo
            mock_interpreter = MagicMock()
            mock_interpreter.get_tensor.return_value = np.array([[0.5]], dtype=np.float32)
            main.interpreter = mock_interpreter
            main.input_details = [{
                'shape': np.array([1, 224, 224, 3]),
                'index': 0
            }]
            main.output_details = [{'index': 0}]
            main.MODEL_LOADED = True
            
            response = app_client.post(
                "/predict/888999000",
                files={"file": ("test.png", img_bytes.read(), "image/png")},
                params={"save_result": True}
            )
            
            # La API debe responder exitosamente incluso si Spring Boot falla
            assert response.status_code == 200
            data = response.json()
            assert "tieneEstrabismo" in data


@pytest.mark.integration
class TestEndToEndIntegration:
    """Pruebas end-to-end del flujo completo de la aplicación"""

    @pytest.fixture(autouse=True)
    def setup_e2e(self, reset_model_state):
        """Configura el estado para pruebas end-to-end"""
        # Configurar modelo mock
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [{
            'shape': np.array([1, 224, 224, 3]),
            'dtype': np.float32,
            'index': 0
        }]
        mock_interpreter.get_output_details.return_value = [{
            'shape': np.array([1, 1]),
            'dtype': np.float32,
            'index': 0
        }]
        mock_interpreter.get_tensor.return_value = np.array([[0.7]], dtype=np.float32)
        
        main.interpreter = mock_interpreter
        main.input_details = [{
            'shape': np.array([1, 224, 224, 3]),
            'dtype': np.float32,
            'index': 0
        }]
        main.output_details = [{
            'shape': np.array([1, 1]),
            'dtype': np.float32,
            'index': 0
        }]
        main.MODEL_LOADED = True
        
        yield
        
        # Cleanup
        main.interpreter = None
        main.input_details = None
        main.output_details = None
        main.MODEL_LOADED = False

    def test_complete_user_journey(self, app_client):
        """Test del flujo completo de un usuario"""
        # 1. Verificar que la API está funcionando
        health_response = app_client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # 2. Subir una imagen para predicción
        img = Image.new('RGB', (400, 300), color='magenta')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        with patch('main.guardar_en_springboot') as mock_guardar:
            mock_guardar.return_value = True
            
            predict_response = app_client.post(
                "/predict/111222333",
                files={"file": ("patient_image.png", img_bytes.read(), "image/png")},
                params={"save_result": True}
            )
            
            assert predict_response.status_code == 200
            prediction_data = predict_response.json()
            
            # Verificar estructura de respuesta
            assert "tieneEstrabismo" in prediction_data
            assert "confianza" in prediction_data
            assert "tiempoProcesamiento" in prediction_data
            assert "mensaje" in prediction_data
            
            # Verificar que se guardó en Spring Boot
            mock_guardar.assert_called_once()
            
            # Verificar que el resultado guardado coincide con la respuesta
            saved_result = mock_guardar.call_args[0][1]
            assert saved_result["tieneEstrabismo"] == prediction_data["tieneEstrabismo"]
            assert saved_result["confianza"] == prediction_data["confianza"]

    def test_multiple_predictions_sequence(self, app_client):
        """Test de múltiples predicciones secuenciales"""
        documentos = [111111111, 222222222, 333333333]
        colors = ['red', 'green', 'blue']
        
        for doc, color in zip(documentos, colors):
            img = Image.new('RGB', (224, 224), color=color)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = app_client.post(
                f"/predict/{doc}",
                files={"file": (f"test_{doc}.png", img_bytes.read(), "image/png")},
                params={"save_result": False}
            )
            
            assert response.status_code == 200, f"Predicción falló para documento {doc}"
            data = response.json()
            assert "tieneEstrabismo" in data
            assert "confianza" in data

    def test_error_recovery_flow(self, app_client):
        """Test que la API se recupera correctamente de errores"""
        # 1. Enviar archivo inválido
        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        
        # 2. Verificar que la API sigue funcionando después del error
        health_response = app_client.get("/health")
        assert health_response.status_code == 200
        
        # 3. Enviar una imagen válida después del error
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test.png", img_bytes.read(), "image/png")},
            params={"save_result": False}
        )
        assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Pruebas de rendimiento e integración"""

    @pytest.fixture(autouse=True)
    def setup_performance(self, reset_model_state):
        """Configura el modelo para las pruebas de rendimiento"""
        # Configurar modelo mock
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [{
            'shape': np.array([1, 224, 224, 3]),
            'dtype': np.float32,
            'index': 0
        }]
        mock_interpreter.get_output_details.return_value = [{
            'shape': np.array([1, 1]),
            'dtype': np.float32,
            'index': 0
        }]
        mock_interpreter.get_tensor.return_value = np.array([[0.7]], dtype=np.float32)
        
        main.interpreter = mock_interpreter
        main.input_details = [{
            'shape': np.array([1, 224, 224, 3]),
            'dtype': np.float32,
            'index': 0
        }]
        main.output_details = [{
            'shape': np.array([1, 1]),
            'dtype': np.float32,
            'index': 0
        }]
        main.MODEL_LOADED = True
        
        yield
        
        # Cleanup
        main.interpreter = None
        main.input_details = None
        main.output_details = None
        main.MODEL_LOADED = False

    def test_prediction_performance(self, app_client):
        """Test que las predicciones se completan en tiempo razonable"""
        import time
        
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        start_time = time.time()
        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test.png", img_bytes.read(), "image/png")},
            params={"save_result": False}
        )
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        # Las predicciones deberían completarse en menos de 5 segundos
        assert elapsed_time < 5.0, f"Predicción tardó {elapsed_time}s, debería ser < 5s"
        
        data = response.json()
        # Verificar que el tiempo de procesamiento está en la respuesta
        assert "tiempoProcesamiento" in data

    def test_concurrent_requests(self, app_client):
        """Test que la API maneja múltiples solicitudes concurrentes"""
        import concurrent.futures
        
        def make_request(doc_id):
            img = Image.new('RGB', (224, 224), color='blue')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = app_client.post(
                f"/predict/{doc_id}",
                files={"file": ("test.png", img_bytes.read(), "image/png")},
                params={"save_result": False}
            )
            return response.status_code == 200
        
        # Hacer 5 solicitudes concurrentes
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(100, 105)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Todas las solicitudes deberían ser exitosas
        assert all(results), "Algunas solicitudes concurrentes fallaron"

