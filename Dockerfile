FROM python:3.9-slim

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configuración del entorno para matplotlib
ENV MPLBACKEND=Agg
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Instala dependencias con versiones específicas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "LetterLens:app"]
