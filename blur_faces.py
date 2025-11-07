import cv2
import os
import face_recognition

# Paths
KNOWN_FACES_DIR = "known_faces"
INPUT_DIR = "input_faces"
OUTPUT_DIR = "output_images"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load known faces and their encodings
known_encodings = []
known_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_encodings.append(encodings[0])
        name = os.path.splitext(filename)[0]
        known_names.append(name)

        print(f"Loaded encoding for: {name}")
    else:
        print(f"⚠️ No face found in {filename}")

print(f"✅ Total known faces loaded: {len(known_encodings)}")

# Process input images
print("\nProcessing input images...")
for filename in os.listdir(INPUT_DIR):
    filepath = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(filepath)

    if image is None:
        print(f"❌ Could not read image {filename}, skipping...")
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the input image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    print(f"\n🎯 Found {len(face_locations)} face(s) in {filename}")

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check for a match with known faces
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

        if best_match_index is not None and face_recognition.compare_faces(
            [known_encodings[best_match_index]], face_encoding, tolerance=0.47
        )[0]:
            name = known_names[best_match_index]
            print(f"✅ Matched with {name} - Distance: {face_distances[best_match_index]:.2f}")

            # Draw a rectangle around the recognized face (optional)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print(f"❌ Unknown face found - Min distance: {min(face_distances):.2f}")

            # Blur the unknown face
            face_region = image[top:bottom, left:right]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            image[top:bottom, left:right] = blurred_face

    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, image)
    print(f"💾 Saved output image to: {output_path}")
