import fitz  # PyMuPDF
import os


def extract_images_from_pdf(pdf_path, output_folder):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Check and create the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Initialize image counter
    image_count = 0

    # Traverse each page and extract images
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load the current page
        image_list = page.get_images(full=True)  # Get the list of images on the page

        # Skip if there are no images on the current page
        if not image_list:
            print(f"No images found on page {page_num + 1}")
            continue

        # Traverse the image list and save images
        for img_index, image in enumerate(image_list):
            xref = image[0]  # Get the xref of the image
            base_image = doc.extract_image(xref)  # Extract the image
            image_bytes = base_image["image"]  # Get the image byte data
            image_extension = base_image["ext"]  # Get the image format (such as png, jpeg)

            # Set the image save path
            image_path = os.path.join(
                output_folder,
                f"page_{page_num + 1}_img_{img_index + 1}.{image_extension}"
            )

            # Save the image
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            print(f"Extracted and saved image: {image_path}")
            image_count += 1

    print(f"A total of {image_count} images were extracted.")


# Set the PDF file path and image output folder path
pdf_path = r"D:\pycharm-code\文献检索\long.pdf"  # Replace with your PDF file path
output_folder = "extracted_images"  # Output folder

# Call the function to extract images
extract_images_from_pdf(pdf_path, output_folder)
