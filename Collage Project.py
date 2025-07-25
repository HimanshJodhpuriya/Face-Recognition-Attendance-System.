import os
import csv
import time
import cv2
import face_recognition
import datetime

class FaceAttendanceSystem:
    def __init__(self):
        self.images_dir = "images"
        self.attendance_file = "attendance.csv"
        self.known_faces = []
        self.known_names = []
        
        # Create images directory if it doesn't exist
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            
        # Create attendance CSV file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time'])
                
        # Load existing face encodings
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load all known face encodings from the images directory"""
        print("Loading known faces...")
        self.known_faces = []
        self.known_names = []
        
        # List all files in the images directory
        if os.path.exists(self.images_dir):
            for filename in os.listdir(self.images_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    # Get person name from filename (without extension)
                    name = os.path.splitext(filename)[0]
                    
                    # Load image and get face encoding
                    image_path = os.path.join(self.images_dir, filename)
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings (might be more than one face in the image)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    # If at least one face is found, use the first one
                    if len(face_encodings) > 0:
                        self.known_faces.append(face_encodings[0])
                        self.known_names.append(name)
                        print(f"Loaded face for: {name}")
                    else:
                        print(f"No face found in image: {filename}")
        
        print(f"Loaded {len(self.known_faces)} faces.")
    
    def capture_new_image(self, name):
        """Capture a new face image for enrollment"""
        if not name:
            print("Name cannot be empty")
            return False
            
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return False
            
        print("Press SPACE to capture an image or ESC to cancel...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Display the frame
            cv2.imshow("Capture New Face - Press SPACE to capture", frame)
            
            # Check for keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("Canceled image capture")
                cap.release()
                cv2.destroyAllWindows()
                return False
                
            elif key == 32:  # SPACE key
                # Check if face is detected in the frame
                face_locations = face_recognition.face_locations(frame)
                
                if len(face_locations) == 0:
                    print("No face detected! Please try again.")
                    continue
                elif len(face_locations) > 1:
                    print("Multiple faces detected! Please ensure only one person is in frame.")
                    continue
                    
                # Save the image with the person's name
                filename = os.path.join(self.images_dir, f"/images/person.jpg")
                cv2.imwrite(filename, frame)
                print(f"Image saved as {filename}")
                
                # Update known faces
                self.load_known_faces()
                
                cap.release()
                cv2.destroyAllWindows()
                return True
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def recognize_faces(self):
        """Recognize faces in real-time and mark attendance"""
        # Check if we have any known faces
        if len(self.known_faces) == 0:
            print("No known faces found. Please enroll a user first.")
            return
            
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        # Track who has already been marked to avoid duplicates
        marked_attendance = set()
        
        print("Starting face recognition... Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
                
            # Find all faces in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            # Process each face found in the frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Try to match with known faces
                matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)
                name = "Unknown"
                
                # Find the best match
                face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        
                        # Mark attendance if not already marked
                        if name not in marked_attendance:
                            self.mark_attendance(name)
                            marked_attendance.add(name)
                            print(f"Attendance marked for {name}")
                
                # Draw a box around the face and display the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # Display the resulting image
            cv2.imshow('Face Recognition', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def mark_attendance(self, name):
        """Mark attendance for a recognized person"""
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")
        
        # Check if user already has attendance for today
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and row[0] == name and row[1] == date_string:
                        print(f"{name} already has attendance marked for today.")
                        return
        
        # Append attendance record to CSV
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_string, time_string])
    
    def list_users(self):
        """List all enrolled users"""
        if os.path.exists(self.images_dir):
            users = [os.path.splitext(f)[0] for f in os.listdir(self.images_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if users:
                print("\nEnrolled users:")
                for idx, user in enumerate(users, 1):
                    print(f"{idx}. {user}")
            else:
                print("\nNo users enrolled yet.")
        else:
            print("\nNo users enrolled yet.")
    
    def delete_user(self, name):
        """Delete a user from the system"""
        # Check all image files in the directory
        if os.path.exists(self.images_dir):
            for filename in os.listdir(self.images_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    user_name = os.path.splitext(filename)[0]
                    if user_name.lower() == name.lower():
                        # Remove the file
                        os.remove(os.path.join(self.images_dir, filename))
                        print(f"User {name} has been deleted.")
                        
                        # Reload known faces
                        self.load_known_faces()
                        return True
        
        print(f"User {name} not found.")
        return False
    
    def view_attendance(self):
        """View the attendance records"""
        if not os.path.exists(self.attendance_file):
            print("No attendance records found.")
            return
            
        print("\nAttendance Records:")
        print("-------------------")
        try:
            with open(self.attendance_file, 'r') as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if idx == 0:  # Header
                        print(f"{row[0]:<20} {row[1]:<12} {row[2]:<10}")
                        print("-" * 42)
                    else:
                        if len(row) >= 3:
                            print(f"{row[0]:<20} {row[1]:<12} {row[2]:<10}")
        except Exception as e:
            print(f"Error reading attendance file: {e}")
    
def main():
    system = FaceAttendanceSystem()
    
    while True:
        print("\n===== Face Recognition Attendance System =====")
        print("1. Enroll new user")
        print("2. Take attendance")
        print("3. List enrolled users")
        print("4. Delete a user")
        print("5. View attendance records")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            name = input("Enter the name for the new user: ")
            system.capture_new_image(name)
        
        elif choice == '2':
            system.recognize_faces()
        
        elif choice == '3':
            system.list_users()
        
        elif choice == '4':
            system.list_users()
            name = input("Enter the name of the user to delete: ")
            system.delete_user(name)
        
        elif choice == '5':
            system.view_attendance()
        
        elif choice == '0':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()