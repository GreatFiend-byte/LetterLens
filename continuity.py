import cv2
import pytesseract

def analyze_word_continuity(image_path):
    """
    Analiza la continuidad en las palabras de una imagen y devuelve los resultados.
    """
    # Leer la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarizar la imagen
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Usar pytesseract para detectar palabras y sus cajas delimitadoras
    data = pytesseract.image_to_data(binary_image, lang='eng', config='--psm 6', output_type=pytesseract.Output.DICT)

    results = []
    for i in range(len(data['text'])):
        word = data['text'][i]
        if word.strip():  # Ignorar palabras vacías
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            word_region = binary_image[y:y+h, x:x+w]
            is_continuous = check_continuity_in_word(word_region)
            results.append({
                "word": word,
                "is_continuous": is_continuous
            })

    return results

def check_continuity_in_word(word_region):
    """
    Determina si una palabra tiene continuidad en sus letras.
    """
    contours, _ = cv2.findContours(word_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    prev_x = None
    for contour in contours:
        x, _, _, _ = cv2.boundingRect(contour)
        if prev_x is not None and abs(x - prev_x) > 10:  # Si hay separación entre letras
            return False
        prev_x = x

    return True
