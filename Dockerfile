FROM python:3.9-slim

# Instala Tesseract y los paquetes de idioma español
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \  # Paquete específico para español
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configuración del entorno para Tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Verifica la instalación de los idiomas
RUN ls -la /usr/share/tesseract-ocr/4.00/tessdata/ && \
    tesseract --list-langs

# Instala dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "LetterLens:app"]
