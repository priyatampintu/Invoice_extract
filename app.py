import os.path
import shutil
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import urllib.request
import img2pdf

def crop_img(x_center, y_center, w, h,  image):
    image_w,image_h = image.shape[1],image.shape[0]
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    boxedImage = image[int(y1):int(y1)+int(h), int(x1):int(x1)+int(w)]
    return boxedImage

def convert_pdf(img_path, pdf_path,number):
    # opening image
    image = Image.open(img_path)

    # converting into chunks using img2pdf
    pdf_bytes = img2pdf.convert(image.filename)

    # opening or creating pdf file
    file_pdf = open(pdf_path, "wb")

    # writing pdf files with chunks
    file_pdf.write(pdf_bytes)

    # closing image file
    image.close()

    # closing pdf file
    file_pdf.close()
    with open(pdf_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label="Export_Report",
                       data=PDFbyte,
                       file_name="invoice_{}.pdf".format(number),
                       mime='application/octet-stream')

def obj_detection(my_img):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    class_label = {0: "Invoice"}
    column1, column2 = st.columns(2)
    column1.subheader("Input image")
    st.text("")
    plt.figure(figsize = (16,16))
    plt.imshow(my_img)
    column1.image(my_img)

    # YOLO model
    model = YOLO("weights/best.pt")
    try:

        results = model.predict(source=my_img, conf=0.40, save=True, save_txt=True)
        out_dir = 'runs/detect/predict'
        out_path =''
        crop_img_list = []
        for file in os.listdir(out_dir):
            #print(file,'**************************************************')
            if file.endswith('.jpg'):
                f = 1
                out_path = os.path.join(out_dir, file)
                label_file = 'runs/detect/predict/labels/image0.txt'
                if os.path.isfile(label_file):
                    with open('runs/detect/predict/labels/image0.txt') as files:
                        lines = [line.rstrip() for line in files]
                        print(len(lines))
                        for line in lines:
                            print(line)
                            x_center, y_center, w, h = float(line.split(' ')[1]), float(line.split(' ')[2]), float(
                                line.split(' ')[3]), float(line.split(' ')[4])
                            img = cv2.cvtColor(np.array(my_img), cv2.COLOR_RGB2BGR)
                            crop = crop_img(x_center, y_center, w, h, img)
                            cv2.imwrite('{}_{}.jpg'.format(file.split('.')[0], str(f)), crop)
                            img = Image.open(file.split('.')[0] + '_' + str(f) + '.jpg')

                            # storing pdf path
                            pdf_path = file.split('.')[0] + '_' + str(f) + '.pdf'
                            img_path = file.split('.')[0] + '_' + str(f) + '.jpg'
                            convert_pdf(img_path, pdf_path, str(f))
                            f += 1
                            st.image(img, caption= 'Extract Invoice')

            else:
                pass

        labels = [class_label[int(x)] for x in results[0].boxes.cls.tolist()]

        # Image loading
        print(out_path)
        op_img = Image.open(out_path)
        newImage = np.array(op_img.convert('RGB'))
        img = cv2.cvtColor(newImage,1)

        st.text("")
        column2.subheader("Output image")
        st.text("")
        plt.figure(figsize = (15, 15))
        plt.imshow(img)
        column2.image(img)
        shutil.rmtree('runs/detect/predict')

        if len(labels)>=1:
            st.success("successfully extracted")
        else:
            st.success("Found {} Object - {}".format(len(labels), "Not detect any pose"))

    except Exception:
        shutil.rmtree('runs/detect/predict')
        st.exception("File is corrupted.Please try other image")

def main():

    st.title("Extract Invoice and save in PDF of each invoice")
    st.write("Upload  multi invoice image and get each invoice in pdf format:")

    choice = st.radio("", ("See an illustration", "Choose an image of your choice"))
    #st.write()

    if choice == "Choose an image of your choice":
        #st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            my_img = Image.open(image_file).convert('RGB')
            obj_detection(my_img)

    elif choice == "See an illustration":
        my_img = Image.open("test.jpeg")
        obj_detection(my_img)

if __name__ == '__main__':
    main()

