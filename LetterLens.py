import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session  # Añadido session aquí
from werkzeug.utils import secure_filename
import pytesseract
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from flask import send_file
from collections import Counter
import re
from datetime import datetime
from PIL import Image
from helpers import check_continuity, analyze_word_continuity_from_text
from glob import glob
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuración de Flask-Session con manejo de errores
try:
    from flask_session import Session
    USE_FLASK_SESSION = True
except ImportError:
    USE_FLASK_SESSION = False
    print("Advertencia: Flask-Session no está instalado. Usando sesiones estándar de Flask.")

# Configura Tesseract para Railway o local
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Configuración del entorno
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/imagenes'
app.config['TEMP_LETTER_FOLDER'] = 'static/temp_letters'
app.config['SECRET_KEY'] = 'supersecretkey'  # Necesario para sesiones

# Configuración de Flask-Session si está disponible
if USE_FLASK_SESSION:
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = './flask_session'
    app.config['SESSION_FILE_THRESHOLD'] = 500
    app.config['SESSION_PERMANENT'] = True
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    Session(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Creación de directorios necesarios
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_LETTER_FOLDER'], exist_ok=True)
if USE_FLASK_SESSION:
    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def cleanup_old_sessions():
    if USE_FLASK_SESSION:
        session_dir = app.config['SESSION_FILE_DIR']
        if os.path.exists(session_dir):
            now = time()
            for f in glob(f"{session_dir}/*"):
                if os.stat(f).st_mtime < now - 3600:
                    os.remove(f)

@app.route('/')
def index():
    cleanup_temp_images()
    return render_template('index.html')

def cleanup_temp_images():
    for filename in os.listdir(app.config['TEMP_LETTER_FOLDER']):
        file_path = os.path.join(app.config['TEMP_LETTER_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    session.clear()
    flash('Las imágenes temporales se han eliminado.')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        flash('No se encontró el archivo.')
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        flash('No seleccionaste ningún archivo.')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = cv2.imread(filepath)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
        
        #custom_config = r'--oem 3 --psm 6'
        custom_config = r'--oem 3 --psm 6 -l spa'
        recognized_text = pytesseract.image_to_string(thresh_image, lang='spa', config=custom_config)
    
        # Extraer cada letra y su posición
        boxes = pytesseract.image_to_boxes(thresh_image, lang='spa')
        letters = []

        # equipo3
        ###################
        areas = []  # Lista para almacenar las áreas de las letras detectadas
        classified_letters = {"pequenia": [], "mediana": [], "grande": []}
        
        for box in boxes.splitlines():
            b = box.split()
            char = b[0]
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cropped_letter = image[image.shape[0] - h:image.shape[0] - y, x:w]

            # equipo3
            ###################
            width = w - x
            height = h - y
            area = width * height
            areas.append(area)  # Agregar área a la lista

            # Guardar la letra recortada como archivo temporal
            temp_letter_path = os.path.join(app.config['TEMP_LETTER_FOLDER'], f"{char}_{x}_{y}.png")
            cv2.imwrite(temp_letter_path, cropped_letter)

            # Convertir la imagen a base64 para mostrar en la interfaz
            _, buffer = cv2.imencode('.png', cropped_letter)
            letter_base64 = base64.b64encode(buffer).decode('utf-8')
            letters.append({'char': char, 'img_data': letter_base64, 'path': temp_letter_path, 'coords': (x, y, w, h), 'area': area})
        
        # equipo3
        ###################
        if areas:
            avg_area = sum(areas) / len(areas)  # Área promedio
            small_threshold = avg_area * 0.5   # Definir límite para letras pequeñas
            medium_threshold = avg_area * 1.5  # Definir límite para letras medianas
        else:
            small_threshold = 0
            medium_threshold = 0

        for letter in letters:
            if letter['area'] < small_threshold:
                classified_letters["pequenia"].append(letter['path'])
            elif letter['area'] < medium_threshold:
                classified_letters["mediana"].append(letter['path'])
            else:
                classified_letters["grande"].append(letter['path'])

        max_length = max(len(classified_letters["pequenia"]), len(classified_letters["mediana"]), len(classified_letters["grande"]))
        total_letters = len(classified_letters["pequenia"]) + len(classified_letters["mediana"]) + len(classified_letters["grande"])
        
        #equipo 5
        text = process_image_text(filepath)
        letter_counts = count_letters(text)
       
        #equipo 6
        angles = get_letter_angles(filepath)
        
        letters_with_angles = [{'char': letter['char'], 'path': letter['path'], 'angle': angle['angle']} 
                       for letter, angle in zip(letters, angles)]

        #Equipo 7
        word_continuity_results = analyze_word_continuity_from_text(recognized_text)
        continuity_results = check_continuity(filepath)
        
        # Optimización de datos para la sesión
        session_data = {
            'recognized_text': recognized_text,
            'image_path': filename,
            'max_length': max_length,
            'letter_counts': letter_counts,
            'continuity_results': continuity_results,
            'is_continuous': continuity_results.get('is_continuous', False),
            'total_letters': total_letters,
            # Solo guardar paths relativos
            'letters': [{'char': l['char'], 'path': l['path']} for l in letters],
            'letters_with_angles': [{'char': l['char'], 'angle': l['angle']} for l in letters_with_angles],
            'word_continuity_results': [{'word': w['word'], 'is_continuous': w['is_continuous']} 
                                     for w in word_continuity_results],
            
            # Clasificación de letras (solo paths)
            'lista_pequenias': classified_letters["pequenia"],
            'lista_medianas': classified_letters["mediana"],
            'lista_grandes': classified_letters["grande"]
        }
        
        # Guardar en sesión
        session.update(session_data)
        session.modified = True  # Forzar guardado
        
        # Renderizar el resultado con los análisis de continuidad
        return render_template('index.html', 
                               letters_with_angles=letters_with_angles, 
                               recognized_text=recognized_text, 
                               letters=letters, 
                               image_path=filename, 
                               letter_counts=letter_counts,
                               lista_pequenias=classified_letters["pequenia"],
                               lista_medianas=classified_letters["mediana"],
                               lista_grandes=classified_letters["grande"],
                               max_length=max_length,
                               word_continuity_results=word_continuity_results,
                               continuity_results=continuity_results)
    
    else:
        flash('Formato de archivo no permitido. Solo se aceptan archivos PNG, JPG y JPEG.')
        return redirect(request.url)

#equipo 5
def process_image_text(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    scale_factor = 300 / 72
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    img = img.resize(new_size, Image.LANCZOS)
    text = pytesseract.image_to_string(img, lang='spa', config='--psm 6')
    return text
    
def count_letters(text):
    counts = Counter(re.findall(r'[A-Za-zÑñ]', text))
    letter_counts = {chr(i): counts.get(chr(i), 0) for i in range(ord('A'), ord('Z') + 1)}
    letter_counts.update({chr(i): counts.get(chr(i), 0) for i in range(ord('a'), ord('z') + 1)})
    letter_counts['Ñ'] = counts.get('Ñ', 0)
    letter_counts['ñ'] = counts.get('ñ', 0)
    return letter_counts

def get_letter_angles(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    letter_angles = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            rect = cv2.minAreaRect(contour)
            angle = rect[-1]
            if angle < -45:
                angle += 90
            letter_angles.append({'angle': angle})
    return letter_angles

@app.route('/process_corrections', methods=['POST'])
def process_corrections():
    action = request.form.get('action')
    letters = session.get('letters', [])
    
    if action.startswith('remove_'):
        letra_path = action.split('_', 1)[1]
        letra_a_eliminar = next((letter for letter in letters if letter['path'] == letra_path), None)
        if letra_a_eliminar:
            letters.remove(letra_a_eliminar)
            if os.path.exists(letra_a_eliminar['path']):
                os.remove(letra_a_eliminar['path'])
        
        session['letters'] = letters
        session.modified = True
        flash('Letra eliminada correctamente.')

    elif action == 'save':
        for idx, letter in enumerate(letters):
            corrected_char = request.form.get(f'letters[{idx}]')
            if corrected_char and corrected_char != letter['char']:
                old_path = os.path.abspath(letter['path'])
                directory = os.path.dirname(old_path)
                old_filename = os.path.basename(old_path)
                new_filename = old_filename.replace(letter['char'], corrected_char)
                new_path = os.path.join(directory, new_filename)
                
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                
                letters[idx]['char'] = corrected_char
                letters[idx]['path'] = new_path

        session['letters'] = letters
        session.modified = True
        flash('Correcciones guardadas correctamente.')

    recognized_text = session.get('recognized_text')
    filename = session.get('image_path')
    
    letters_with_images = []
    for letter in letters:
        if os.path.exists(letter['path']):
            with open(letter['path'], "rb") as img_file:
                letter_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            letters_with_images.append({'char': letter['char'], 'img_data': letter_base64, 'path': letter['path']})

    return render_template('index.html', recognized_text=recognized_text, letters=letters_with_images, image_path=filename)

@app.route('/cleanup_temp_images')
def cleanup_temp_images():
    for filename in os.listdir(app.config['TEMP_LETTER_FOLDER']):
        file_path = os.path.join(app.config['TEMP_LETTER_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    session.clear()
    flash('Las imágenes temporales se han eliminado.')
    return redirect(url_for('index'))

@app.route('/download_pdf')
def download_pdf():
    from reportlab.lib.pagesizes import letter as letter_size
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    import unicodedata
    import sys
    import base64
    from io import BytesIO

    pdf_buffer = BytesIO()
    
    # Obtener datos optimizados de la sesión
    recognized_text = session.get('recognized_text', '')
    letters = session.get('letters', [])
    image_path = session.get('image_path', '')
    letter_counts = session.get('letter_counts', {})
    letters_with_angles = session.get('letters_with_angles', [])
    continuity_results = session.get('continuity_results', {})
    word_continuity_results = session.get('word_continuity_results', [])
    classified_letters = {
        "pequenia": session.get('lista_pequenias', []),
        "mediana": session.get('lista_medianas', []),
        "grande": session.get('lista_grandes', [])
    }
    max_length = session.get('max_length', 0)
    
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter_size)
    styles = getSampleStyleSheet()
    elements = []
    
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=1,
        spaceAfter=20,
        textColor=colors.HexColor('#2C3E50')
    )
    
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#3498DB')
    )

    def safe_image(path, width=None, height=None):
        try:
            if os.path.exists(path):
                normalized_path = unicodedata.normalize('NFKD', path).encode('ascii', 'ignore').decode('ascii')
                if os.path.exists(normalized_path):
                    return Image(normalized_path, width=width, height=height)
                return Image(path, width=width, height=height)
            return None
        except Exception as e:
            print(f"Error al cargar imagen {path}: {str(e)}", file=sys.stderr)
            return None

    def base64_to_image(base64_str, width=None, height=None):
        try:
            if base64_str.startswith('data:image/png;base64,'):
                base64_str = base64_str.split(',')[1]
            img_data = base64.b64decode(base64_str)
            img_buffer = BytesIO(img_data)
            return Image(img_buffer, width=width, height=height)
        except Exception as e:
            print(f"Error al procesar imagen base64: {str(e)}", file=sys.stderr)
            return None

    elements.append(Paragraph("Reporte Completo de Análisis de Escritura", title_style))
    elements.append(Paragraph(f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    
    if image_path:
        elements.append(Paragraph("1. Imagen Analizada", section_style))
        try:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
            img = safe_image(img_path, width=5*inch, height=3*inch)
            if img:
                elements.append(img)
            else:
                elements.append(Paragraph("Imagen no disponible", styles['Italic']))
        except Exception as e:
            app.logger.error(f"Error loading main image: {str(e)}")
            elements.append(Paragraph(f"Error al cargar imagen: {str(e)}", styles['Italic']))
        elements.append(Spacer(1, 0.2*inch))

    if recognized_text:
        elements.append(Paragraph("2. Texto Reconocido", section_style))
        text_style = ParagraphStyle(
            'TextStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=12
        )
        elements.append(Paragraph(recognized_text, text_style))
        elements.append(Spacer(1, 0.2*inch))

    if letters:
        elements.append(Paragraph("3. Caracteres Segmentados", section_style))
        data = []
        row = []
        
        for i, letter in enumerate(letters):
            img = safe_image(letter['path'], width=0.5*inch, height=0.5*inch)
            char_info = f"Letra: {letter['char']}"
            
            if img:
                cell_content = [img, Paragraph(char_info, styles['Normal'])]
            else:
                cell_content = [Paragraph(f"{char_info} (sin imagen)", styles['Normal'])]
            
            row.append(cell_content)
            
            if (i+1) % 4 == 0 or i == len(letters)-1:
                data.append(row)
                row = []
        
        table = Table(data, colWidths=[1.5*inch]*4)
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))

    if max_length > 0:
        elements.append(Paragraph("4. Clasificador de Letras", section_style))
        data = [["Pequeñas", "Medianas", "Grandes"]]
        
        for i in range(max_length):
            row = []
            for category in ["pequenia", "mediana", "grande"]:
                if i < len(classified_letters[category]):
                    if category == "pequenia":
                        img = safe_image(classified_letters[category][i], width=0.12*inch, height=0.13*inch)
                    elif category == "mediana":
                        img = safe_image(classified_letters[category][i], width=0.25*inch, height=0.25*inch)
                    else:
                        img = safe_image(classified_letters[category][i], width=0.5*inch, height=0.5*inch)
                    row.append(img if img else "")
                else:
                    row.append("")
            data.append(row)
        
        table = Table(data, colWidths=[2*inch]*3)
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8F9FA')),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        # 1. Gráfica de pastel (clasificación de letras)
        try:
            grafica_pastel = generar_grafica_pastel()
            if grafica_pastel:
                img_pastel = base64_to_image(grafica_pastel, width=6*inch, height=5*inch)
                if img_pastel:
                    elements.append(img_pastel)
                    elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            print(f"Error al generar gráfica de pastel: {str(e)}", file=sys.stderr)


    if letters_with_angles:
        elements.append(Paragraph("5. Letras Reconocidas y Ángulos", section_style))
        data = [["Letra", "Ángulo (°)"]]
        for letter in letters_with_angles:
            data.append([letter.get('char', ''), str(letter.get('angle', 'N/A'))])
        
        table = Table(data, colWidths=[1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8F9FA')),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))

    if letter_counts:
        elements.append(Paragraph("6. Letras Comparadas", section_style))
        filtered_counts = {k: v for k, v in letter_counts.items() if v > 0}
        sorted_letters = sorted(filtered_counts.items(), key=lambda x: x[0])
        
        data = []
        row = []
        
        for i, (letter, count) in enumerate(sorted_letters):
            row.append(f"{letter}: {count}")
            if (i+1) % 6 == 0 or i == len(sorted_letters)-1:
                data.append(row)
                row = []
        
        table = Table(data, colWidths=[1*inch]*6)
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        try:
            grafica_letras = generar_grafica_letras()
            if grafica_letras:
                img_letras = base64_to_image(grafica_letras, width=7*inch, height=6*inch)
                if img_letras:
                    elements.append(img_letras)
                    elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            print(f"Error al generar gráfica de letras: {str(e)}", file=sys.stderr)

    elements.append(Paragraph("7. Continuidad del Trazo", section_style))
    if continuity_results:
        elements.append(Paragraph(
            f"Porcentaje general de arriba hacia abajo: {continuity_results.get('above_below', 'N/A')}%", 
            styles['Normal']))
        
        is_continuous = continuity_results.get('is_continuous', False)
        elements.append(Paragraph(
            f"¿El trazo general es continuo?: {'Sí' if is_continuous else 'No'}", 
            styles['Normal']))
    else:
        elements.append(Paragraph("No hay datos de continuidad general", styles['Italic']))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("8. Continuidad por Palabra", section_style))
    if word_continuity_results:
        data = [["Palabra", "Continuidad"]]
        for result in word_continuity_results[:15]:
            data.append([result.get('word', ''), 'Sí' if result.get('is_continuous', False) else 'No'])
        
        table = Table(data, colWidths=[3.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('ALIGN', (1,1), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8F9FA')),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("No hay datos de continuidad por palabra", styles['Italic']))
    
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("Reporte generado por Digitalizador de Letras", styles['Italic']))

    doc.build(elements)
    pdf_buffer.seek(0)
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"analisis_escritura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mimetype="application/pdf"
    )

def generar_grafica_pastel():
    # Obtener los datos de la sesión
    classified_letters = {
        "pequenia": session.get('lista_pequenias', []),
        "mediana": session.get('lista_medianas', []),
        "grande": session.get('lista_grandes', [])
    }
    
    # Calcular las cantidades
    cantidades = {
        "Pequeña": len(classified_letters["pequenia"]),
        "Mediana": len(classified_letters["mediana"]),
        "Grande": len(classified_letters["grande"])
    }
    
    total = sum(cantidades.values())
    
    # Preparar datos para la gráfica
    labels = [f"{k} ({v})" for k, v in cantidades.items()]
    sizes = cantidades.values()
    
    # Crear la gráfica
    plt.figure(figsize=(7, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Para que el pastel sea circular
    plt.title(f'Distribución de Letras\nTotal: {total} letras')
    
    # Convertir la gráfica a imagen para mostrar en HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Codificar la imagen en base64
    grafica_url = base64.b64encode(image_png).decode('utf-8')
    grafica_url = f'data:image/png;base64,{grafica_url}'
    
    # Cerrar la figura para liberar memoria
    plt.close()
    
    return grafica_url


@app.route('/mostrar-grafica')
def mostrar_grafica():
    grafica_url = generar_grafica_pastel()  # Esto ya es un string Base64
    return render_template('grafica.html', grafica_url=grafica_url)



def generar_grafica_letras():
    # Obtener datos de la sesión
    letter_counts = session.get('letter_counts', {})
    total_letters = session.get('total_letters', 1)  # Evitar división por cero
    
    # Filtrar letras con al menos 1 ocurrencia y ordenar por frecuencia
    filtered_letters = {k: v for k, v in sorted(letter_counts.items(), 
                         key=lambda item: item[1], reverse=True) if v >= 1}
    
    sum_counts = sum(filtered_letters.values())
    
    # Calcular margen de error
    margen_error = max(0, total_letters - sum_counts)
    
    # Preparar datos
    labels = [f"{letter} ({count})" for letter, count in filtered_letters.items()]
    sizes = [(count / total_letters) * 100 for count in filtered_letters.values()]
    
    # Añadir margen de error si existe
    if margen_error > 0:
        labels.append(f"Margen error ({margen_error})")
        sizes.append((margen_error / total_letters) * 100)
    
    # Configurar gráfica
    plt.figure(figsize=(10, 8))
    
    # Colores y estilo
    colors = plt.cm.tab20.colors[:len(labels)]
    explode = [0.02] * len(labels)
    
    # Generar gráfico de pastel con manejo seguro de autopct
    wedges, _, autotexts = plt.pie(
        sizes,
        labels=None,
        autopct=lambda p: f'{p:.1f}%' if p >= 1 else None,  # None en lugar de ''
        startangle=140,
        colors=colors,
        explode=explode,
        pctdistance=0.8,
        textprops={'fontsize': 8},
        wedgeprops={'width': 0.4}
    )
    
    # Ajustar porcentajes (solo para los que son visibles)
    for autotext in autotexts:
        if autotext.get_text():  # Solo si tiene texto
            p = float(autotext.get_text().replace('%', ''))
            autotext.set_color('white' if p > 5 else 'black')
    
    # Leyenda
    plt.legend(
        wedges,
        labels,
        title="Letras (conteo)",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8
    )
    
    # Título
    plt.title(f'Distribución de Letras (≥1 ocurrencia)\nTotal: {total_letters} letras | Margen error: {margen_error}',
             fontsize=11, pad=20)
    
    # Ajustar layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Convertir a imagen
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    grafica_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    
    return f'data:image/png;base64,{grafica_url}'

@app.route('/grafica_letras')
def grafica_letras():
    grafica_url = generar_grafica_letras()
    return render_template('grafica2.html', grafica_url=grafica_url)


if __name__ == '__main__':
    app.run(debug=True)
