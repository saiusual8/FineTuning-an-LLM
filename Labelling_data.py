import pytesseract
from PIL import Image
from pytesseract import Output

# Load the image
image_path = '/Users/saitejagudidevini/Documents/costcobills/costcobill.jpg.webp'
img = Image.open(image_path)

# Use Tesseract to extract data
ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)

# Initialize a list to store the prices
prices = []

# Iterate through the extracted text to find prices
for word in ocr_data['text']:
    # Detect prices: numbers with a decimal point
    if word.replace('.', '', 1).isdigit() and word.count('.') == 1 and len(word) <= 6:
        prices.append(word)

# Print the extracted prices
print("Extracted Prices:", prices)

# Optionally, you can save the prices to a file or return them for further processing
