import pytesseract
from PIL import Image
from pytesseract import Output
import pandas as pd
import re
import cv2

# Load the image
image_path = '/Users/saitejagudidevini/Documents/costcobills/costcobill.jpg.webp'
img = cv2.imread(image_path)

#Step1: Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Step2: Apply a binary threshold to make the text stand out
_, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

#Step3: Optional: Appply GaussianBlur to reduce noice (if necessary)
blurred_img = cv2.GaussianBlur(thresh_img, (5,5), 0)


# Use Tesseract to extract data with bounding box information
ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)
# Initialize lists to store product IDs, product names, and prices
product_ids = []
product_names = []
prices = []

# Variables to track the current state
current_product_id = ''
current_product_name = []
current_price = ''

price_pattern = r'\d+\.\d{2}'
# Iterate over each text element and its line number
for i in range(len(ocr_data['text'])):
    word = ocr_data['text'][i].strip()

    if word:  # Skip empty text
        # Detect product ID (numeric with length >= 5)
        if word.replace('/', '').isdigit() and len(word) >= 5:
            if current_product_id:  # If we already have a product ID, store the previous product info
                product_ids.append(current_product_id)
                product_names.append(' '.join(current_product_name))
                prices.append(current_price)
                
            # Start tracking the new product
            current_product_id = word
            current_product_name = []
            current_price = ''
        
        # Detect price (numbers with a decimal point)
        elif re.match(price_pattern, word):
            current_price = word
        
        # Collect product names (text between product IDs and prices)
        elif current_product_id and not current_price:  # Only collect product names after product ID and before price
            current_product_name.append(word)


price_pattern = re.compile(r'\d{1,3}(?:,\d{3})*(?:\.\d{2})')
for pr in ocr_data['text']:
    if re.match(price_pattern, pr):
        prices.append(word)

# Append the last product if valid
if current_product_id and current_product_name and current_price:
    product_ids.append(current_product_id)
    product_names.append(' '.join(current_product_name))
    prices.append(current_price)

# Handle any length mismatches by padding lists
max_length = max(len(product_ids), len(product_names), len(prices))

# Extend lists to be of equal length
product_ids.extend([""] * (max_length - len(product_ids)))
product_names.extend([""] * (max_length - len(product_names)))
prices.extend([""] * (max_length - len(prices)))

print(prices)


# Convert to pandas DataFrame
df = pd.DataFrame({
    'Product ID': product_ids,
    'Product Name': product_names,
    'Price': prices
})

#Remove rows with empty product ID's or names
df = df[df['Product ID'].astype(bool) & df['Product Name'].astype(bool)]

print(df)

# Save the dataframe to Excel
output_path = '/Users/saitejagudidevini/Documents/costcobills/extracted_products_refined.xlsx'
df.to_excel(output_path, index=False)
print(f"Excel file saved to {output_path}")
