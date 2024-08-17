# Visual Privacy Protection Application

This project is an AI-based system designed to protect visual privacy by anonymizing faces and obfuscating sensitive textual information (PII) in both images and videos. The system is designed to be robust and efficient for real-time applications.

## Features

### Core Features

- **Face Detection and Anonymization**  
  - Detects faces in images and video frames using a deep learning-based model.
  - Provides options to anonymize detected faces through blurring or pixelation.

- **Sensitive Text Detection and Obfuscation**  
  - Utilizes Optical Character Recognition (OCR) to detect text within images.
  - Leverages Natural Language Processing (NLP) techniques to identify sensitive information such as Personally Identifiable Information (PII).
  - Obfuscates sensitive text using methods like drawing black rectangles or applying blurring.

### Additional Features

- **Real-Time Processing**  
  - Capable of processing live video feeds to anonymize faces and obfuscate sensitive text in real-time.

- **Confidence Threshold Adjustment**  
  - Allows users to adjust the confidence threshold for face detection, which controls the sensitivity and effectiveness of the anonymization process.

- **User-Friendly Interface**  
  - Provides a simple Graphical User Interface (GUI) that enables users to:
    - Select images for processing
    - Start and stop live video feeds
    - Choose between different anonymization methods (e.g., blurring, pixelation)
    - View the anonymized output for verification

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/visual-privacy-protection.git
   cd visual-privacy-protection
   
2. **Install dependencies:**

This project uses Python and a set of specific libraries. Install the dependencies with the following command:
    pip install -r requirements.txt

3. **Run the application:**
You can start the application with:
python imageredactionwithblurringandpixelation.py

GUI screenshot:
![image](https://github.com/user-attachments/assets/08a17f59-c765-49c8-a4e4-485d0b98c758)


## Example


Here’s an example of how the system works on a sample image:

Original Image:
![image](https://github.com/user-attachments/assets/53011744-1303-4655-8c8d-e48c4d7d8ad8)

Anonymized Image:
![image](https://github.com/user-attachments/assets/4fe1dff3-48c0-4036-a979-3c37a495d797)


## Contributions
We welcome contributions! Feel free to submit a pull request or open an issue if you encounter any bugs or have feature suggestions.

## License
This project is licensed under the MIT License.
