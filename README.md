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
```bash
pip install -r requirements.txt
```

3. **Run the application:**

You can start the application with:
```bash
python imageredactionwithblurringandpixelation.py
```

GUI screenshot:
![image](https://github.com/user-attachments/assets/08a17f59-c765-49c8-a4e4-485d0b98c758)

## Solution Overview / Architecture

### Input:
- Load images or video streams for processing.

### Processing:
1. **OCR Detection:**  
   - Uses `EasyOCR` to extract text from images.
  
2. **Text Analysis:**  
   - Uses `spaCy` to analyze the extracted text and identify PII (Personally Identifiable Information).
  
3. **Face Detection:**  
   - Utilizes OpenCV’s deep learning-based face detector to find faces in images and videos.

4. **Anonymization:**  
   - Applies chosen anonymization techniques (blurring or pixelation) to faces and PII in the media.

### Output:
- Save or display the anonymized images or video streams.

---

## Technical Details

- **Programming Language:** Python
- **Framework:** Tkinter for GUI

### Libraries:
- **EasyOCR:** For text detection and extraction from images.
- **Model** 
- **spaCy:** For text analysis and PII identification.
- **OpenCV:** For face detection and image processing (blurring and pixelation).
- **PIL (Pillow):** For image handling and manipulation.
- **NumPy:** For numerical operations and handling image data.

### Applications:
- **VSCode**, **PyCharm:** As the integrated development environment (IDE) for coding and testing.

---


## Example


Here’s an example of how the system works on a sample image:

Original Image:

![image](https://github.com/user-attachments/assets/87efeeb3-51ab-4e90-895c-af84507c9edd)


Anonymized Image:




## References

1. **L. Rakhmawati, Wirawan and Suwadi, "Image Privacy Protection Techniques: A Survey,"**  
   TENCON 2018 - 2018 IEEE Region 10 Conference, Jeju, Korea (South), 2018, pp. 0076-0080, doi: [10.1109/TENCON.2018.8650339](https://doi.org/10.1109/TENCON.2018.8650339)

2. **Senior, A.W., Pankanti, S. (2011). Privacy Protection and Face Recognition.**  
   In: Li, S., Jain, A. (eds) Handbook of Face Recognition. Springer, London.  
   [https://doi.org/10.1007/978-0-85729-932-1_27](https://doi.org/10.1007/978-0-85729-932-1_27)

3. **Ren, Zhongzheng & Lee, Yong & Ryoo, Michael. (2018). Learning to Anonymize Faces for Privacy Preserving Action Detection.**

4. **Hukkelås, H., Mester, R., Lindseth, F. (2019). DeepPrivacy: A Generative Adversarial Network for Face Anonymization.**  
   In: Bebis, G., et al. Advances in Visual Computing. ISVC 2019. Lecture Notes in Computer Science(), vol 11844. Springer, Cham.  
   [https://doi.org/10.1007/978-3-030-33720-9_44](https://doi.org/10.1007/978-3-030-33720-9_44)

5. **Padilla-Lopez, J. R., Chaaraoui, A. A., & Florez-Revuelta, F. Visual Privacy Protection Methods: A Survey.**  
   Department of Computer Technology, University of Alicante, P.O. Box 99, E-03080 Alicante, Spain, and Faculty of Science, Engineering and Computing, Kingston University.

6. **Li, Yiming & Liu, Peidong & Jiang, Yong & Xia, Shu-Tao. (2021). Visual Privacy Protection via Mapping Distortion.**  
   10.1109/ICASSP39728.2021.9414149.

7. **GitHub Repo:** [Blur and anonymize faces with OpenCV and Python](https://github.com/charlsefrancis/Blur-and-anonymize-faces-with-OpenCV-and-Python)


## Contributions
We welcome contributions! Feel free to submit a pull request or open an issue if you encounter any bugs or have feature suggestions.

## License
This project is licensed under the MIT License.
