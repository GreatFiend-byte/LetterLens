import pytesseract
import cv2
import os
from collections import Counter
from continuity import analyze_word_continuity

# Abecedario en inglés
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'pdf'}


def extract_text_from_image(filepath):
    # Leer la imagen con OpenCV
    image = cv2.imread(filepath)
    
    # Aplicar un filtro para mejorar la legibilidad
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(image, kernel, iterations=2)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    image = dilation
    
    # Usar pytesseract para extraer texto
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
    return text.strip()



def check_continuity(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_percentage_above_below = 0
    total_percentage_below_above = 0
    count_above_below = 0
    count_below_above = 0

    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Filtrar contornos muy pequeños
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        top_point = None
        bottom_point = None
        
        for point in contour:
            px, py = point[0]
            if top_point is None or py < top_point[1]:
                top_point = (px, py)
            if bottom_point is None or py > bottom_point[1]:
                bottom_point = (px, py)

        if top_point is None or bottom_point is None:
            continue
        
        top_y, bottom_y = top_point[1], bottom_point[1]
        
        if top_y < bottom_y:
            percentage = abs((top_y - bottom_y) / h) * 100
            total_percentage_above_below += percentage
            count_above_below += 1
        else:
            percentage = abs((bottom_y - top_y) / h) * 100
            total_percentage_below_above += percentage
            count_below_above += 1
    
    average_percentage_above_below = (total_percentage_above_below / count_above_below) if count_above_below else 0
    average_percentage_below_above = (total_percentage_below_above / count_below_above) if count_below_above else 0
    
    return {
        "above_below": average_percentage_above_below,
        "below_above": average_percentage_below_above
    }

def improve_letter_detection(image):
    # Aplicar umbralización y suavizado de bordes
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(binary_image, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    
    # Aplicar suavizado de bordes
    blurred_image = cv2.GaussianBlur(dilation, (5, 5), 0)
    
    return blurred_image

def find_closest_letter(detected_char, alphabet):
    return min(alphabet, key=lambda x: abs(ord(x) - ord(detected_char)))

def correct_misspelled_words(text, reference_text):
    words = text.split()
    corrected_words = []
    for word in words:
        if word in reference_text:
            corrected_words.append(word)
        else:
            closed_word = find_closest_letter(word, reference_text)
            corrected_words.append(closed_word)
    return ' '.join(corrected_words)

def main():
    filepath = input("Ingrese el nombre del archivo (incluya la extensión): ")
    filename = os.path.basename(filepath)
    
    extracted_text = extract_text(filepath, filename)
    print("\nTexto extraido:")
    print(extracted_text)
    
    continuity_result = check_continuity(filepath)
    print("\nResultado de la verificación de continuidad:")
    print(continuity_result)
    
    improved_image = improve_letter_detection(cv2.imread(filepath))
    detected_text = pytesseract.image_to_string(improved_image, lang='eng')
    print("\nTexto detectado mejorando las letras:")
    print(detected_text)
    
    reference_text = input("\nIngrese el texto de referencia para corregir errores tipográficos: ")
    
    corrected_text = correct_misspelled_words(detected_text, reference_text)
    print("\nTexto corregido:")
    print(corrected_text)

if __name__ == "__main__":
    main()


def analyze_word_continuity_from_text(text):
    """
    Analiza la continuidad palabra por palabra del texto extraído.
    """
    words = text.split()
    results = []
    
    for word in words:
        is_continuous = check_continuity_in_word_from_text(word)
        results.append({
            "word": word,
            "is_continuous": is_continuous
        })
    
    return results

def check_continuity_in_word_from_text(word):
    """
    Comprueba si una palabra tiene continuidad en su trazo (basado en reglas sencillas).
    Aquí se puede añadir una lógica más compleja si es necesario.
    """
    # Lógica simple para verificar continuidad (puedes personalizarla)
    for i in range(1, len(word)):
        if ord(word[i]) - ord(word[i - 1]) > 1:  # Si hay una gran diferencia en los caracteres
            return False
    return True
