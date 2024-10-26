import customtkinter as ctk
from realtime_detection import update_frame, release_video  

import warnings
warnings.filterwarnings("ignore")

# Initialize customtkinter with a theme and color mode
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Create the main window
root = ctk.CTk()
root.title("Webcam Viewer with Prediction")

# Set window size and position
w, h = 550, 670
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - w) // 2
y = (screen_height - h) // 2
root.geometry(f"{w}x{h}+{x}+{y}")

# Add a label to display the webcam feed within the application window
video_label = ctk.CTkLabel(root, text="")
video_label.pack(pady=10)

# Add a text area for displaying the predicted character sentence
text_area = ctk.CTkTextbox(root, width=530, height=200, font=("Arial", 40, "bold"))
text_area.pack(pady=10)

# Start the video stream within the application
update_frame(video_label, text_area)

# Start the main application loop
root.mainloop()

# Release the video capture when the window is closed
release_video()
