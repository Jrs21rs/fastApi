#!/usr/bin/env python3
"""
Script de verificación pre-despliegue para FastAPI
Verifica que todos los archivos necesarios estén presentes
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Verifica si un archivo existe"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: {filepath} NO ENCONTRADO")
        return False

def main():
    print("🔍 Verificando archivos para despliegue en Render...\n")
    
    all_ok = True
    
    # Archivos requeridos
    files_to_check = [
        ("app/main.py", "Aplicación FastAPI"),
        ("app/requirements.txt", "Dependencias Python"),
        ("modelos/strabismus_model.tflite", "Modelo TensorFlow Lite"),
        ("Dockerfile", "Configuración Docker"),
        ("render.yaml", "Configuración Render"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_ok = False
    
    print("\n" + "="*60)
    
    # Verificar contenido de requirements.txt
    print("\n📦 Verificando dependencias...")
    try:
        with open("app/requirements.txt", "r") as f:
            requirements = f.read()
            required_packages = ["fastapi", "uvicorn", "tensorflow", "Pillow", "numpy"]
            for package in required_packages:
                if package.lower() in requirements.lower():
                    print(f"✅ {package} encontrado")
                else:
                    print(f"❌ {package} NO encontrado")
                    all_ok = False
    except FileNotFoundError:
        print("❌ No se pudo leer requirements.txt")
        all_ok = False
    
    # Verificar tamaño del modelo
    print("\n📊 Verificando modelo...")
    model_path = "modelos/strabismus_model.tflite"
    if os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📏 Tamaño del modelo: {model_size_mb:.2f} MB")
        if model_size_mb > 100:
            print("⚠️  ADVERTENCIA: El modelo es grande, puede tardar en desplegarse")
        if model_size_mb > 500:
            print("❌ ERROR: El modelo excede el límite de Render Free (512MB RAM)")
            all_ok = False
    
    print("\n" + "="*60)
    
    if all_ok:
        print("\n✅ ¡Todo listo para desplegar en Render!")
        print("\n📝 Próximos pasos:")
        print("1. Haz commit de los cambios:")
        print("   git add .")
        print("   git commit -m 'Ready for Render deployment'")
        print("   git push")
        print("\n2. Ve a https://dashboard.render.com/")
        print("3. Crea un nuevo Web Service")
        print("4. Conecta tu repositorio")
        print("5. Render detectará automáticamente el Dockerfile")
        print("\n🚀 ¡Buena suerte con el despliegue!")
        return 0
    else:
        print("\n❌ Hay problemas que deben resolverse antes del despliegue")
        return 1

if __name__ == "__main__":
    sys.exit(main())
