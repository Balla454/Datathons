import os
import csv
from deepface import DeepFace

# Get the current working directory
current_directory = os.getcwd()
image_folder_path = os.path.join(current_directory, "faceimages")
output_directory = current_directory
output_csv_path = os.path.join(output_directory, "face_results.csv")

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Open the CSV file for writing
with open(output_csv_path, mode='w', newline='') as csv_file:
    fieldnames = ['filename', 'gender', 'race', 'age']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header row to the CSV file
    writer.writeheader()

    for filename in os.listdir(image_folder_path):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(image_folder_path, filename)

            try:
                # Analyze the current image with OpenCV backend
                results = DeepFace.analyze(img_path=image_path, actions=['gender', 'race', 'age'], detector_backend='opencv')

                # Write the results to the CSV file
                first_face_result = results[0]
                writer.writerow({
                    'filename': filename,
                    'gender': max(first_face_result["gender"], key=first_face_result["gender"].get),
                    'race': max(first_face_result["race"], key=first_face_result["race"].get),
                    'age': first_face_result["age"]
                })
            except ValueError as e_opencv:
                # Write a row with 'n/a' values if face detection fails with OpenCV
                writer.writerow({
                    'filename': filename,
                    'gender': 'unknown',
                    'race': 'unknown',
                    'age': int(-1)
                })
