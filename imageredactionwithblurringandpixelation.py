import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2
import easyocr
import os
import spacy
import numpy as np

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])
nlp = spacy.load("en_core_web_sm")

# Initialize face detector
prototxt_path = 'models/deploy.prototxt'
weights_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

# Global variable to control the live feed loop
live_feed_active = False

# Function to anonymize detected faces in an image
def anonymize_image(input_path, output_path, confidence_threshold):
    # Load the image
    image = Image.open(input_path)
    draw = ImageDraw.Draw(image)

    # Perform OCR using EasyOCR
    results = reader.readtext(input_path)

    # Store bounding boxes of PII areas
    pii_bounding_boxes = []

    for bbox, text, prob in results:
        # Use spaCy to analyze the detected text
        doc = nlp(text)
        for ent in doc.ents:
            # Check if the detected entity is PII
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL',
                              'NORP', 'FAC', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                # Calculate bounding box coordinates
                top_left = (int(bbox[0][0]), int(bbox[0][1]))
                bottom_right = (int(bbox[2][0]), int(bbox[2][1]))

                # Draw a black rectangle over the PII text
                draw.rectangle([top_left, bottom_right], fill="black")

                # Store the bounding box of the PII area
                pii_bounding_boxes.append([top_left, bottom_right])

    # Convert to RGB mode if the image has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply blurring to detected faces
    image = apply_blurring(image, pii_bounding_boxes, confidence_threshold)

    # Save the anonymized image
    image.save(output_path)


# Function to apply blurring to detected faces in an image
def apply_blurring(image, pii_bounding_boxes, confidence_threshold):
    # Convert the image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect faces in the image
    blob = cv2.dnn.blobFromImage(cv_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Apply blurring to detected faces and draw confidence percentage
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # Get coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * [cv_image.shape[1], cv_image.shape[0], cv_image.shape[1], cv_image.shape[0]]
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face region
            face = cv_image[startY:endY, startX:endX]

            # Apply Gaussian blur to anonymize the face
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)

            # Replace the face region with the blurred face
            cv_image[startY:endY, startX:endX] = blurred_face

            # Draw a bounding box and confidence percentage
            label = f"{confidence * 100:.2f}%"
            cv_image = cv2.putText(cv_image, label, (startX, startY-10 if startY-10>10 else startY+10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv_image = cv2.rectangle(cv_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Apply blurring to areas containing PII
    for top_left, bottom_right in pii_bounding_boxes:
        # Blur the area containing PII
        blurred_area = cv_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        blurred_area = cv2.GaussianBlur(blurred_area, (99, 99), 30)

        # Replace the area containing PII with the blurred area
        cv_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_area

    # Convert the image back to PIL format
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))


# Function to display live feed from the camera
def display_live_feed():
    global live_feed_active
    live_feed_active = True
    cap = cv2.VideoCapture(0)

    def update_frame():
        if live_feed_active:
            ret, frame = cap.read()
            if not ret:
                stop_live_feed()
                return

            # Perform OCR using EasyOCR
            results = reader.readtext(frame)

            # Store bounding boxes of PII areas
            pii_bounding_boxes = []

            for bbox, text, prob in results:
                # Use spaCy to analyze the detected text
                doc = nlp(text)
                for ent in doc.ents:
                    # Check if the detected entity is PII
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL',
                                      'NORP', 'FAC', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                        # Calculate bounding box coordinates
                        top_left = (int(bbox[0][0]), int(bbox[0][1]))
                        bottom_right = (int(bbox[2][0]), int(bbox[2][1]))

                        # Store the bounding box of the PII area
                        pii_bounding_boxes.append([top_left, bottom_right])

            # Apply pixelation or blurring to detected faces and PII
            confidence_threshold = confidence_slider.get() / 100.0
            if anonymization_var.get() == "Blur":
                frame = apply_blurring_to_frame(frame, pii_bounding_boxes, confidence_threshold)
            else:
                frame = apply_pixelation_to_frame(frame, pii_bounding_boxes, confidence_threshold)

            # Display the frame
            cv2.imshow('Live Feed', frame)

            # Schedule the next frame update
            root.after(10, update_frame)
        else:
            cap.release()
            cv2.destroyAllWindows()
    # Start updating frames
    update_frame()


# Function to stop the live feed
def stop_live_feed():
    global live_feed_active
    live_feed_active = False


# Function to apply blurring to detected faces and PII areas in a frame
def apply_blurring_to_frame(frame, pii_bounding_boxes, confidence_threshold):
    # Detect faces in the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Apply blurring to detected faces and draw confidence percentage
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # Get coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face region
            face = frame[startY:endY, startX:endX]

            # Apply Gaussian blur to anonymize the face
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)

            # Replace the face region with the blurred face
            frame[startY:endY, startX:endX] = blurred_face

            # Draw a bounding box and confidence percentage
            label = f"{confidence * 100:.2f}%"
            frame = cv2.putText(frame, label, (startX, startY-10 if startY-10>10 else startY+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Apply blurring to areas containing PII
    for top_left, bottom_right in pii_bounding_boxes:
        # Blur the area containing PII
        blurred_area = cv2.GaussianBlur(frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], (99, 99), 30)

        # Replace the area containing PII with the blurred area
        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_area

    return frame


# Function to apply pixelation to detected faces and PII areas in a frame
def apply_pixelation_to_frame(frame, pii_bounding_boxes, confidence_threshold):
    # Detect faces in the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Apply pixelation to detected faces and draw confidence percentage
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # Get coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face region
            face = frame[startY:endY, startX:endX]

            # Apply pixelation to anonymize the face
            pixelated_face = anonymize_face_pixelate(face, blocks=10)

            # Replace the face region with the pixelated face
            frame[startY:endY, startX:endX] = pixelated_face

            # Draw a bounding box and confidence percentage
            label = f"{confidence * 100:.2f}%"
            frame = cv2.putText(frame, label, (startX, startY-10 if startY-10>10 else startY+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Apply pixelation to areas containing PII
    for top_left, bottom_right in pii_bounding_boxes:
        # Extract the PII region
        pii_area = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Apply pixelation to anonymize the PII
        pixelated_pii = anonymize_face_pixelate(pii_area, blocks=10)

        # Replace the PII region with the pixelated PII
        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = pixelated_pii

    return frame


# Function to anonymize detected faces in an image using pixelation
def anonymize_face_pixelate(face, blocks=10):
    # Get the dimensions of the face
    (h, w) = face.shape[:2]

    # Calculate the size of each block in pixels
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")

    # Loop over each block in the face
    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            # Compute the starting and ending (x, y)-coordinates of the current block
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]

            # Extract the ROI of the block from the face
            roi = face[start_y:end_y, start_x:end_x]

            # Fill the block with the mean pixel intensity value of the ROI
            B = roi[:, :, 0].mean()
            G = roi[:, :, 1].mean()
            R = roi[:, :, 2].mean()
            face[start_y:end_y, start_x:end_x] = (B, G, R)

    # Return the pixelated face
    return face


# Function to select and anonymize an image
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            output_path = os.path.splitext(file_path)[0] + "_anonymized.jpg"
            confidence_threshold = confidence_slider.get() / 100.0
            anonymize_image(file_path, output_path, confidence_threshold)
            messagebox.showinfo("Success", f"Anonymized image saved to {output_path}")

            # Display the anonymized image
            anonymized_image = Image.open(output_path)
            anonymized_image.thumbnail((300, 300))  # Resize the image if necessary
            anonymized_image_tk = ImageTk.PhotoImage(anonymized_image)
            output_image_label.configure(image=anonymized_image_tk)
            output_image_label.image = anonymized_image_tk  # Keep a reference to avoid garbage collection
        except Exception as e:
            messagebox.showerror("Error", str(e))


# Create the main window
root = tk.Tk()
root.title("Anonymizer")

# Create and place widgets
title_label = tk.Label(root, text="Image and Video Anonymizer", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

btn_select_image = ttk.Button(frame, text="Anonymize Image", command=select_image)
btn_select_image.pack(pady=10, padx=20, side=tk.LEFT)

anonymization_var = tk.StringVar(value="Blur")

blur_radio = ttk.Radiobutton(frame, text="Blur", variable=anonymization_var, value="Blur")
blur_radio.pack(pady=10, padx=20, side=tk.LEFT)

pixelate_radio = ttk.Radiobutton(frame, text="Pixelate", variable=anonymization_var, value="Pixelate")
pixelate_radio.pack(pady=10, padx=20, side=tk.LEFT)

confidence_label = tk.Label(frame, text="Confidence Threshold (%)")
confidence_label.pack(pady=10, padx=20, side=tk.LEFT)
confidence_slider = ttk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL)
confidence_slider.set(50)
confidence_slider.pack(pady=10, padx=20, side=tk.LEFT)

confidence_value_label = tk.Label(frame, text=f"Selected Threshold: {confidence_slider.get()}%")
confidence_value_label.pack(pady=10, padx=20, side=tk.LEFT)

def update_confidence_label(value):
    confidence_value_label.config(text=f"Selected Threshold: {confidence_slider.get()}%")

confidence_slider.config(command=update_confidence_label) 

btn_live_feed = ttk.Button(frame, text="Start Live Feed", command=display_live_feed)
btn_live_feed.pack(pady=10, padx=20, side=tk.LEFT)

btn_stop_live_feed = ttk.Button(frame, text="Stop Live Feed", command=stop_live_feed)
btn_stop_live_feed.pack(pady=10, padx=20, side=tk.LEFT)

btn_quit = ttk.Button(frame, text="Quit", command=root.quit)
btn_quit.pack(pady=10, padx=20, side=tk.LEFT)

# Output image label
output_image_label = tk.Label(root)
output_image_label.pack(pady=10)

# Run the application
root.mainloop()

