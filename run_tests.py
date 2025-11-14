#!/usr/bin/env python
"""
Script para ejecutar las pruebas unitarias del proyecto
"""
import subprocess
import sys

def run_tests():
    """Ejecuta las pruebas unitarias"""
    print("🧪 Ejecutando pruebas unitarias...\n")
    
    # Ejecutar pytest con cobertura
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--cov=app", "--cov-report=term-missing"],
        cwd="."
    )
    
    if result.returncode == 0:
        print("\n✅ Todas las pruebas pasaron exitosamente!")
    else:
        print("\n❌ Algunas pruebas fallaron")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()

