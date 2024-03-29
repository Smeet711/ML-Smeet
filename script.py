import cv2
import numpy as np
import time
import os
from moviepy.editor import VideoFileClip, VideoClip, AudioFileClip
from rembg import remove
from moviepy.audio.AudioClip import CompositeAudioClip



def merge_audio_with_video(video_path, person_audio_path, background_audio_path, output_path):
    try:
        # Load video clip
        video_clip = VideoFileClip(video_path)

        # Load person audio clip
        person_audio_clip = AudioFileClip(person_audio_path)

        # Load background audio clip
        background_audio_clip = AudioFileClip(background_audio_path)

        # Set the audio of the video clip
        video_clip = video_clip.set_audio(person_audio_clip.set_duration(video_clip.duration))

        # Mix background audio with person audio
        final_audio_clip = CompositeAudioClip([person_audio_clip, background_audio_clip.set_duration(video_clip.duration)])

        # Set the audio of the video clip with both person and background audio
        video_clip = video_clip.set_audio(final_audio_clip)

        # Write the final video with merged audio
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=video_clip.fps)

        print(f"Video and audio merged and saved as {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def preprocess_frame(frame):
    # Apply any additional preprocessing steps here
    # Example: Increase contrast and saturation
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

    # Convert BGR frame to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Multiply the saturation channel by a factor
    frame_hsv[:, :, 1] = frame_hsv[:, :, 1] * 1.2

    # Clip the values to the valid range
    frame_hsv = np.clip(frame_hsv, 0, 255)

    # Convert HSV frame back to BGR
    frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

    return frame

def process_and_overlay_videos(person_video_path, background_path, output_path):
    # Remove background from the person video
    cap_person = cv2.VideoCapture(person_video_path)
    fps_person = cap_person.get(cv2.CAP_PROP_FPS)
    width_person = int(cap_person.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_person = int(cap_person.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc_person = cv2.VideoWriter_fourcc(*'mp4v')
    processed_person_output_path = "processed_person_video.mp4"
    writer_person = cv2.VideoWriter(processed_person_output_path, fourcc_person, fps_person, (width_person, height_person), isColor=True)
    
    

    while cap_person.isOpened():
        ret_person, frame_person = cap_person.read()

        if not ret_person:
            break

        # Convert BGR frame to RGBA
        frame_rgba_person = cv2.cvtColor(frame_person, cv2.COLOR_BGR2RGBA)

        # Use rembg to remove the background
        result_person = remove(frame_rgba_person)

        # Convert back to BGR
        result_bgr_person = cv2.cvtColor(result_person, cv2.COLOR_RGBA2BGR)
        
       
        
        
        
        
        

        writer_person.write(result_bgr_person)

    cap_person.release()
    writer_person.release()

    # Overlay videos
    cap_background = cv2.VideoCapture(background_path)
    if not cap_background.isOpened():
        print("Error: Could not open the background video file.")
        return

    cap_processed_person = cv2.VideoCapture(processed_person_output_path)
    if not cap_processed_person.isOpened():
        print("Error: Could not open the processed person video file.")
        cap_background.release()
        return

    background_width = int(cap_background.get(cv2.CAP_PROP_FRAME_WIDTH))
    background_height = int(cap_background.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_fps = cap_background.get(cv2.CAP_PROP_FPS)

    person_width = int(cap_processed_person.get(cv2.CAP_PROP_FRAME_WIDTH))
    person_height = int(cap_processed_person.get(cv2.CAP_PROP_FRAME_HEIGHT))
    person_fps = cap_processed_person.get(cv2.CAP_PROP_FPS)

    overlay_width = int(background_width / 4)
    overlay_height = int((overlay_width / person_width) * person_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, background_fps, (background_width, background_height))

    start_time = time.time()
    change_interval = 60  # Change position every 60 seconds

    # Choose an initial corner randomly
    initial_corner = np.random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])

    # Initialize initial_position_x and initial_position_y outside the loop
    if initial_corner == "top_left":
        initial_position_x = 0
        initial_position_y = 0
    elif initial_corner == "top_right":
        initial_position_x = background_width - overlay_width
        initial_position_y = 0
    elif initial_corner == "bottom_left":
        initial_position_x = 0
        initial_position_y = background_height - overlay_height
    elif initial_corner == "bottom_right":
        initial_position_x = background_width - overlay_width
        initial_position_y = background_height - overlay_height

    while True:
        ret_background, background_frame = cap_background.read()
        ret_person, person_frame = cap_processed_person.read()

        if not ret_background or not ret_person:
            break

        elapsed_time = time.time() - start_time

        # Change position every change_interval seconds
        if elapsed_time >= change_interval:
            start_time = time.time()
            # Determine the diagonal corner based on the initial corner
            if initial_corner == "top_left":
                target_corner = "bottom_right"
            elif initial_corner == "top_right":
                target_corner = "bottom_left"
            elif initial_corner == "bottom_left":
                target_corner = "top_right"
            elif initial_corner == "bottom_right":
                target_corner = "top_left"

            # Set the initial position based on the new target corner
            if target_corner == "top_left":
                initial_position_x = 0
                initial_position_y = 0
            elif target_corner == "top_right":
                initial_position_x = background_width - overlay_width
                initial_position_y = 0
            elif target_corner == "bottom_left":
                initial_position_x = 0
                initial_position_y = background_height - overlay_height
            elif target_corner == "bottom_right":
                initial_position_x = background_width - overlay_width
                initial_position_y = background_height - overlay_height

            initial_corner = target_corner  # Update the initial corner

        person_frame = cv2.resize(person_frame, (overlay_width, overlay_height))

        mask = np.zeros((overlay_height, overlay_width), dtype=np.uint8)
        cv2.circle(mask, (overlay_width // 2, overlay_height // 2), min(overlay_width, overlay_height) // 2, 255, -1)

        roi = background_frame[initial_position_y:initial_position_y + overlay_height, initial_position_x:initial_position_x + overlay_width]
        cv2.bitwise_and(person_frame, person_frame, mask=mask, dst=roi)

        output_video.write(background_frame)

    cap_background.release()
    cap_processed_person.release()
    output_video.release()
    cv2.destroyAllWindows()

    # Merge audio with the final video
    output_with_audio_path = "output_with_audio.mp4"
    merge_audio_with_video(output_path, person_video_path,background_path, output_with_audio_path)

    # Cleanup temporary files
    os.remove(processed_person_output_path)

# Example usage:
person_video_path = "Videom.mp4"
background_path = "Video11.mp4"
output_path = "Joined_video2.mp4"

process_and_overlay_videos(person_video_path, background_path,output_path)