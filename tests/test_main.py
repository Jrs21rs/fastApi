"""
Pruebas unitarias para la aplicación de detección de estrabismo
"""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, MagicMock, patch, mock_open
import io
import os
import sys
import json
from fastapi.testclient import TestClient
from fastapi import UploadFile

# Agregar el directorio app al path
# TensorFlow ya está mockeado en conftest.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import main


class TestLoadModel:
    """Tests para la función load_model()"""

    @patch('os.path.exists')
    @patch('tensorflow.lite.Interpreter')
    def test_load_model_success(self, mock_interpreter_class, mock_exists, reset_model_state):
        """Test que el modelo se carga correctamente cuando existe"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [{
            'shape': np.array([1, 224, 224, 3]),
            'dtype': np.float32
        }]
        mock_interpreter.get_output_details.return_value = [{
            'shape': np.array([1, 1]),
            'dtype': np.float32
        }]
        mock_interpreter_class.return_value = mock_interpreter

        # Ejecutar
        result = main.load_model()

        # Verificar
        assert result is True
        assert main.MODEL_LOADED is True
        assert main.interpreter is not None
        assert main.input_details is not None
        assert main.output_details is not None
        mock_interpreter.allocate_tensors.assert_called_once()

    @patch('os.path.exists')
    def test_load_model_not_found(self, mock_exists, reset_model_state):
        """Test que retorna False cuando el modelo no se encuentra"""
        # Configurar mock para que ninguna ruta exista
        mock_exists.return_value = False

        # Ejecutar
        result = main.load_model()

        # Verificar
        assert result is False
        assert main.MODEL_LOADED is False

    @patch('os.path.exists')
    @patch('tensorflow.lite.Interpreter')
    def test_load_model_exception(self, mock_interpreter_class, mock_exists, reset_model_state):
        """Test que maneja excepciones al cargar el modelo"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_interpreter_class.side_effect = Exception("Error al cargar modelo")

        # Ejecutar
        result = main.load_model()

        # Verificar
        assert result is False
        assert main.MODEL_LOADED is False

    @patch('os.path.exists')
    @patch('tensorflow.lite.Interpreter')
    def test_load_model_checks_multiple_paths(self, mock_interpreter_class, mock_exists, reset_model_state):
        """Test que verifica múltiples rutas en orden"""
        # Configurar mock para que la segunda ruta exista
        def exists_side_effect(path):
            return path == "../modelos/strabismus_model.tflite"
        
        mock_exists.side_effect = exists_side_effect
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [{
            'shape': np.array([1, 224, 224, 3]),
            'dtype': np.float32
        }]
        mock_interpreter.get_output_details.return_value = [{'shape': np.array([1, 1])}]
        mock_interpreter_class.return_value = mock_interpreter

        # Ejecutar
        result = main.load_model()

        # Verificar
        assert result is True
        # Verificar que se llamó con la ruta correcta
        assert mock_interpreter_class.called


class TestPreprocessImage:
    """Tests para la función preprocess_image()"""

    def setup_method(self):
        """Configurar estado necesario para los tests"""
        main.input_details = [{
            'shape': np.array([1, 224, 224, 3]),
            'dtype': np.float32
        }]

    def test_preprocess_image_rgb(self, sample_image):
        """Test preprocesamiento de imagen RGB"""
        result = main.preprocess_image(sample_image)

        assert result.shape == (1, 224, 224, 3)
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_preprocess_image_rgba(self, sample_image_rgba):
        """Test preprocesamiento de imagen RGBA (debe convertir a RGB)"""
        result = main.preprocess_image(sample_image_rgba)

        assert result.shape == (1, 224, 224, 3)
        assert result.dtype == np.float32
        assert len(result.shape) == 4

    def test_preprocess_image_grayscale(self, sample_image_grayscale):
        """Test preprocesamiento de imagen en escala de grises"""
        result = main.preprocess_image(sample_image_grayscale)

        assert result.shape == (1, 224, 224, 3)
        assert result.dtype == np.float32

    def test_preprocess_image_different_size(self):
        """Test preprocesamiento de imagen con tamaño diferente"""
        img = Image.new('RGB', (500, 300), color='blue')
        result = main.preprocess_image(img)

        assert result.shape == (1, 224, 224, 3)

    def test_preprocess_image_normalization(self, sample_image):
        """Test que la imagen se normaliza correctamente a [0, 1]"""
        result = main.preprocess_image(sample_image)

        assert np.max(result) <= 1.0
        assert np.min(result) >= 0.0


class TestGuardarEnSpringBoot:
    """Tests para la función guardar_en_springboot()"""

    @patch('requests.post')
    @patch('main.datetime')
    def test_guardar_en_springboot_success(self, mock_datetime, mock_post):
        """Test que guarda exitosamente en Spring Boot"""
        # Configurar mocks
        mock_datetime.now.return_value.strftime.return_value = "2024-01-15"
        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        # Ejecutar
        resultado = {"tieneEstrabismo": True, "confianza": 0.85}
        result = main.guardar_en_springboot(123456789, resultado)

        # Verificar
        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['documentoIdentidad'] == 123456789
        assert call_args[1]['json']['resultado'] is True
        assert call_args[1]['json']['confianzaPrediccion'] == 0.85

    @patch('requests.post')
    @patch('main.datetime')
    def test_guardar_en_springboot_error_status(self, mock_datetime, mock_post):
        """Test que maneja errores de estado HTTP"""
        # Configurar mocks
        mock_datetime.now.return_value.strftime.return_value = "2024-01-15"
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        # Ejecutar
        resultado = {"tieneEstrabismo": False, "confianza": 0.3}
        result = main.guardar_en_springboot(987654321, resultado)

        # Verificar
        assert result is False

    @patch('requests.post')
    @patch('main.datetime')
    def test_guardar_en_springboot_connection_error(self, mock_datetime, mock_post):
        """Test que maneja errores de conexión"""
        # Configurar mocks
        mock_datetime.now.return_value.strftime.return_value = "2024-01-15"
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection error")

        # Ejecutar
        resultado = {"tieneEstrabismo": True, "confianza": 0.9}
        result = main.guardar_en_springboot(111222333, resultado)

        # Verificar
        assert result is False

    @patch('requests.post')
    @patch('main.datetime')
    def test_guardar_en_springboot_timeout(self, mock_datetime, mock_post):
        """Test que maneja timeouts"""
        # Configurar mocks
        mock_datetime.now.return_value.strftime.return_value = "2024-01-15"
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")

        # Ejecutar
        resultado = {"tieneEstrabismo": False, "confianza": 0.2}
        result = main.guardar_en_springboot(444555666, resultado)

        # Verificar
        assert result is False


class TestEndpoints:
    """Tests para los endpoints de la API"""

    def test_predict_endpoint_model_not_loaded(self, app_client, sample_image_bytes, reset_model_state):
        """Test que retorna error cuando el modelo no está cargado"""
        # Asegurar que el modelo no está cargado
        main.MODEL_LOADED = False
        
        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test.png", sample_image_bytes, "image/png")}
        )

        assert response.status_code == 500
        assert "Modelo no cargado" in response.json()["detail"]

    @patch('main.preprocess_image')
    @patch('main.guardar_en_springboot')
    def test_predict_endpoint_success(
        self, mock_guardar, mock_preprocess, app_client, sample_image_bytes, reset_model_state
    ):
        """Test endpoint de predicción exitoso"""
        # Configurar estado del modelo
        mock_interpreter = MagicMock()
        mock_interpreter.get_tensor.return_value = np.array([[0.75]], dtype=np.float32)
        main.interpreter = mock_interpreter
        main.input_details = [{'shape': np.array([1, 224, 224, 3]), 'index': 0}]
        main.output_details = [{'index': 0}]
        main.MODEL_LOADED = True
        
        # Configurar mocks
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3), dtype=np.float32)
        mock_guardar.return_value = True

        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            params={"save_result": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert "tieneEstrabismo" in data
        assert "confianza" in data
        assert "tiempoProcesamiento" in data
        assert "mensaje" in data
        assert data["tieneEstrabismo"] is True  # 0.75 > 0.6
        mock_guardar.assert_called_once()

    def test_predict_endpoint_invalid_file_type(self, app_client, reset_model_state):
        """Test que rechaza archivos que no son imágenes"""
        main.MODEL_LOADED = True
        main.input_details = [{'shape': np.array([1, 224, 224, 3]), 'index': 0}]
        main.output_details = [{'index': 0}]
        main.interpreter = MagicMock()
        
        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )

        assert response.status_code == 400
        assert "debe ser una imagen" in response.json()["detail"]

    @patch('main.preprocess_image')
    def test_predict_endpoint_no_strabismus(
        self, mock_preprocess, app_client, sample_image_bytes, reset_model_state
    ):
        """Test predicción cuando no hay estrabismo"""
        # Configurar estado del modelo
        mock_interpreter = MagicMock()
        mock_interpreter.get_tensor.return_value = np.array([[0.4]], dtype=np.float32)
        main.interpreter = mock_interpreter
        main.input_details = [{'shape': np.array([1, 224, 224, 3]), 'index': 0}]
        main.output_details = [{'index': 0}]
        main.MODEL_LOADED = True
        
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3), dtype=np.float32)

        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            params={"save_result": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tieneEstrabismo"] is False  # 0.4 < 0.6

    def test_health_endpoint_model_not_loaded(self, app_client, reset_model_state):
        """Test endpoint de health check cuando el modelo no está cargado"""
        main.MODEL_LOADED = False
        
        response = app_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["model_loaded"] is False

    def test_health_endpoint_model_loaded(self, app_client, reset_model_state):
        """Test endpoint de health check cuando el modelo está cargado"""
        main.MODEL_LOADED = True
        main.input_details = [{'shape': np.array([1, 224, 224, 3])}]

        response = app_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_input_shape" in data
        assert "timestamp" in data

    def test_root_endpoint(self, app_client, reset_model_state):
        """Test endpoint raíz"""
        main.MODEL_LOADED = False
        
        response = app_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Strabismus Detection API"
        assert data["status"] == "running"
        assert "version" in data

    def test_root_endpoint_with_model(self, app_client, reset_model_state):
        """Test endpoint raíz con modelo cargado"""
        main.MODEL_LOADED = True
        main.input_details = [{'shape': np.array([1, 224, 224, 3])}]

        response = app_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert "model_input_shape" in data


class TestEdgeCases:
    """Tests para casos límite y edge cases"""

    @patch('main.preprocess_image')
    def test_predict_with_shape_mismatch(
        self, mock_preprocess, app_client, sample_image_bytes, reset_model_state
    ):
        """Test que maneja error cuando las dimensiones no coinciden"""
        # Configurar estado del modelo
        main.MODEL_LOADED = True
        main.input_details = [{'shape': np.array([1, 224, 224, 3]), 'index': 0}]
        main.output_details = [{'index': 0}]
        main.interpreter = MagicMock()
        
        # Configurar mock para que retorne shape diferente
        mock_preprocess.return_value = np.zeros((1, 256, 256, 3), dtype=np.float32)  # Shape diferente

        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test.png", sample_image_bytes, "image/png")}
        )

        assert response.status_code == 500
        detail = response.json()["detail"].lower()
        # El mensaje ahora debería ser directamente "Error de dimensiones del modelo"
        assert "dimensiones" in detail or "dimension" in detail or "dimensiones del modelo" in detail

    @patch('main.preprocess_image')
    def test_predict_with_processing_error(self, mock_preprocess, app_client, sample_image_bytes, reset_model_state):
        """Test que maneja errores durante el procesamiento"""
        main.MODEL_LOADED = True
        mock_preprocess.side_effect = Exception("Error de procesamiento")

        response = app_client.post(
            "/predict/123456789",
            files={"file": ("test.png", sample_image_bytes, "image/png")}
        )

        assert response.status_code == 500
        assert "Error procesando imagen" in response.json()["detail"]

    def test_preprocess_image_with_single_channel(self):
        """Test preprocesamiento de imagen con un solo canal"""
        main.input_details = [{'shape': np.array([1, 224, 224, 3])}]
        img = Image.new('L', (224, 224), color=128)
        
        result = main.preprocess_image(img)
        
        assert result.shape == (1, 224, 224, 3)
        assert result.dtype == np.float32

