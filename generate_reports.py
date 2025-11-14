#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para generar reportes HTML y PDF de las pruebas unitarias
"""
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def create_reports_dir():
    """Crea el directorio de reportes si no existe"""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "coverage").mkdir(exist_ok=True)
    return reports_dir

def generate_html_report():
    """Genera reporte HTML de las pruebas"""
    print("Ejecutando pruebas y generando reportes...\n")
    
    reports_dir = create_reports_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    html_file = reports_dir / f"test_report_{timestamp}.html"
    coverage_dir = reports_dir / "coverage"
    
    # Ejecutar pytest con todos los reportes
    result = subprocess.run(
        [
            "pytest",
            "tests/",
            "-v",
            "--cov=app",
            f"--cov-report=html:{coverage_dir}",
            "--cov-report=term-missing",
            "--cov-report=xml:reports/coverage.xml",
            f"--html={html_file}",
            "--self-contained-html"
        ],
        cwd="."
    )
    
    print(f"\nReportes generados:")
    print(f"   - Reporte de pruebas HTML: {html_file}")
    print(f"   - Reporte de cobertura: {coverage_dir}/index.html")
    print(f"   - Reporte XML de cobertura: reports/coverage.xml")
    
    return html_file, result.returncode

def generate_pdf_from_html(html_file):
    """Convierte el reporte HTML a PDF usando weasyprint"""
    try:
        import weasyprint
        
        reports_dir = create_reports_dir()
        pdf_file = reports_dir / f"{html_file.stem}.pdf"
        
        print(f"\nGenerando PDF desde HTML...")
        
        # Generar PDF
        html_doc = weasyprint.HTML(filename=str(html_file))
        html_doc.write_pdf(str(pdf_file))
        
        print(f"Reporte PDF generado: {pdf_file}")
        return pdf_file
    except ImportError:
        print("\nweasyprint no esta instalado.")
        print("   Instala con: pip install weasyprint")
        print("   O usa el navegador para imprimir el HTML a PDF (Ctrl+P)")
        return None
    except Exception as e:
        print(f"\nError generando PDF: {e}")
        print("   Puedes abrir el HTML en el navegador e imprimirlo como PDF")
        return None

def open_report(html_file):
    """Abre el reporte HTML en el navegador"""
    try:
        import webbrowser
        import os
        
        file_path = os.path.abspath(html_file)
        webbrowser.open(f"file://{file_path}")
        print(f"\nAbriendo reporte en el navegador...")
    except Exception as e:
        print(f"\nNo se pudo abrir automaticamente: {e}")
        print(f"   Abre manualmente: {html_file}")

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Generador de Reportes de Pruebas")
        print("=" * 60)
    except UnicodeEncodeError:
        print("=" * 60)
        print("Generador de Reportes de Pruebas")
        print("=" * 60)
    
    # Generar reportes HTML
    html_file, exit_code = generate_html_report()
    
    # Generar PDF si se solicita
    if "--pdf" in sys.argv or "-p" in sys.argv:
        generate_pdf_from_html(html_file)
    
    # Abrir en navegador si se solicita
    if "--open" in sys.argv or "-o" in sys.argv:
        open_report(html_file)
    
    print(f"\nTodos los reportes estan en: reports/")
    print(f"\nUso:")
    print(f"   python generate_reports.py          # Solo HTML")
    print(f"   python generate_reports.py --pdf    # HTML + PDF")
    print(f"   python generate_reports.py --open  # HTML + abrir navegador")
    print(f"   python generate_reports.py --pdf --open  # Todo")
    
    sys.exit(exit_code)

