import cv2
import numpy as np
from keras.models import load_model
import time
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model("micro_expression_model.keras")
# Initialize OpenCV face cascade classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Positive', 'Negative', 'Surprise']

# Define video writer object
frame_width = int(640)
frame_height = int(480)
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Start video capture from webcam
cap = cv2.VideoCapture(1)

# Initialize variables to track expression changes
prev_emotion = None
start_time = None

# Initialize variables for emotion timeline
timeline_start = time.time()
timeline_data = []

# Initialize variables for graphical representation
x_values = []
y_values = []
colors = {'Positive': 'green', 'Negative': 'red', 'Surprise': 'blue'}

# Loop to capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Process each face in the frame
    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]  # Region of interest (ROI) in grayscale
        roi_color = frame[y:y + h, x:x + w]  # Region of interest (ROI) in color

        # Resize and normalize the ROI
        resized_roi = cv2.resize(roi_color, (48, 48))
        normalized_roi = resized_roi / 255.0

        # Perform prediction
        predictions = model.predict(np.expand_dims(normalized_roi, axis=0))
        print("Predictions:", predictions)  # Add this line for debugging
        emotion_label = emotion_labels[np.argmax(predictions)]

        # Draw rectangle around the face and label the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # If expression changes, update variables
        if emotion_label != prev_emotion:
            if start_time is not None:
                end_time = time.time()
                duration = end_time - start_time
                timeline_data.append((start_time - timeline_start, end_time - timeline_start, prev_emotion))
                start_time = None
            else:
                start_time = time.time()
            prev_emotion = emotion_label

    # Display the frame
    cv2.imshow('Live Video', frame)

    # Write frame to output video
    out.write(frame)

    # Append data for graphical representation
    x_values.append(time.time())
    y_values.append(emotion_label)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Save the emotion timeline as an image
plt.figure()
for start, end, emotion in timeline_data:
    plt.plot([start, end], [emotion_labels.index(emotion), emotion_labels.index(emotion)], color=colors[emotion], linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('Emotion')
plt.title('Emotion Timeline')
plt.yticks(np.arange(3), emotion_labels)
plt.grid(True)
plt.tight_layout()
plt.savefig('emotion_timeline.png')
plt.close()

# Save the graphical representation as an image
plt.scatter(x_values, y_values, c=[colors[val] for val in y_values], marker='|', s=100)
plt.xlabel('Time')
plt.ylabel('Emotion')
plt.title('Emotion Timeline')
plt.xticks(rotation=45)
plt.yticks(np.arange(3), emotion_labels)
plt.tight_layout()
plt.savefig('graphical_representation.png')
plt.close()

# Save the emotion timeline data to a text file
with open('emotion_timeline.txt', 'w') as f:
    for start, end, emotion in timeline_data:
        f.write(f"{start}: {emotion}\n")
        f.write(f"{end}: {emotion}\n")

print("Emotion timeline, graphical representation, and report saved successfully.")
