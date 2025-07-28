# save this as predict.py and run it with python predict.py
# from ultralytics import YOLO


# model = YOLO(r"C:\Users\ait\Desktop\TestYo\runs\detect\train3\weights\best.pt")
 # results = model(r"C:\Users\ait\Desktop\TestYo\test3.png")

# Access the first (and only) result
# results[0].show()


import os
from ultralytics import YOLO

# Paths
image_folder = r"C:\Users\ait\Desktop\TestYo\SubBandSpectrum"
result_folder = r"C:\Users\ait\Desktop\TestYo\Result"
model_path = r"C:\Users\ait\Desktop\TestYo\runs\detect\train3\weights\best.pt"

# Create result folder if needed
os.makedirs(result_folder, exist_ok=True)

# List image files
image_files = [f for f in os.listdir(image_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

if not image_files:
    print("No image files found in the folder.")
    exit()

# Show image list
print("Select image(s) to run prediction (e.g. 1,2,5):")
for idx, filename in enumerate(image_files):
    print(f"{idx + 1}. {filename}")

# Get multiple selections from user
while True:
    input_str = input("Enter numbers separated by commas: ")
    try:
        indices = [int(i.strip()) - 1 for i in input_str.split(',')]
        if all(0 <= i < len(image_files) for i in indices):
            break
        else:
            print("One or more choices are out of range. Try again.")
    except ValueError:
        print("Invalid input. Use numbers separated by commas (e.g. 1,2,3).")

# Load model
model = YOLO(model_path)

# Process each selected image
for i in indices:
    filename = image_files[i]
    image_path = os.path.join(image_folder, filename)
    print(f"\nProcessing: {filename}")

    results = model(image_path)
    results[0].show()

    # Ask to save result
    save_choice = input(f"Save result for '{filename}'? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_path = os.path.join(result_folder, filename)
        results[0].save(filename=save_path)
        print(f"Saved to: {save_path}")
    else:
        print("Not saved.")

print("\nDone.")
