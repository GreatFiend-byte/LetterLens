# Usa una imagen base con Python y Tesseract preinstalado
FROM python:3.9-slim

# Instala Tesseract y sus dependencias
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configura el entorno de trabajo
WORKDIR /app

# Copia los archivos necesarios
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Configura la variable de entorno para Tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Ejecuta la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "LetterLens:app"]
