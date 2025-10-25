# Dockerfile para FastAPI - Detección de Estrabismo
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY app/requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY app/ ./app/

# Copiar modelo (debe estar en la raíz del proyecto)
COPY modelos/ ./modelos/

# Exponer puerto
EXPOSE 5000

# Variables de entorno
ENV PYTHONUNBUFFERED=1

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
