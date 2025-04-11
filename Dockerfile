FROM python:3.9-slim

# 1. Instalar Tesseract y dependencias con ubicación explícita de los idiomas
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-all \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Verificar la instalación de idiomas
RUN mkdir -p /usr/share/tesseract-ocr/tessdata && \
    find /usr -name "*.traineddata" -exec cp {} /usr/share/tesseract-ocr/tessdata/ \; && \
    tesseract --list-langs

WORKDIR /app

# 3. Configurar variable de entorno (ruta absoluta)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata

# 4. Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 5. Verificación final durante el build
RUN echo "Idiomas instalados:" && tesseract --list-langs && \
    echo "Archivos en TESSDATA_PREFIX:" && ls -la $TESSDATA_PREFIX

CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--timeout", "120", "--workers", "4", "--threads", "2", "LetterLens:app"]
