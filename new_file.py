from docx2pdf import convert
import os

def doc_to_pdf(input_path, output_path=None):
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".pdf"
    
    convert(input_path, output_path)
    print(f"Pdf create at path: {output_path}")

if __name__ =="__main__":
    input_file = input("Please type the correct file name: ")
    doc_to_pdf(input_file)