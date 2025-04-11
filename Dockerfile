FROM python:3.9-slim

# 1. Instalar Tesseract y dependencias
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Configurar la ubicación correcta de los datos de Tesseract
# En nuevas versiones, los datos están en /usr/share/tesseract-ocr/tessdata/
RUN mkdir -p /usr/share/tesseract-ocr/tessdata/ && \
    ln -s /usr/share/tesseract-ocr/tessdata/ /usr/share/tesseract-ocr/4.00/tessdata

# 3. Verificar la instalación
RUN tesseract --list-langs && \
    ls -la /usr/share/tesseract-ocr/tessdata/

WORKDIR /app

# 4. Configurar variable de entorno
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata

# 5. Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "LetterLens:app"]
