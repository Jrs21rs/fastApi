#!/usr/bin/env python
"""
Script para ejecutar las pruebas de integración del proyecto
"""
import subprocess
import sys

def run_integration_tests():
    """Ejecuta las pruebas de integración"""
    print("🔗 Ejecutando pruebas de integración...\n")
    
    # Ejecutar pytest solo con pruebas de integración
    result = subprocess.run(
        [
            "pytest", 
            "tests/test_integration.py", 
            "-v", 
            "-m", "integration",
            "--cov=app", 
            "--cov-report=term-missing",
            "--cov-report=html:reports/coverage",
            "--html=reports/integration_test_report.html",
            "--self-contained-html"
        ],
        cwd="."
    )
    
    if result.returncode == 0:
        print("\n✅ Todas las pruebas de integración pasaron exitosamente!")
    else:
        print("\n❌ Algunas pruebas de integración fallaron")
        sys.exit(1)

if __name__ == "__main__":
    run_integration_tests()

