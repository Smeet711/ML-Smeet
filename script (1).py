import cv2
import numpy as np
import time
import os
from moviepy.editor import VideoFileClip, AudioFileClip
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

def process_and_overlay_videos(person_video_path, background_path, output_path, start_time_seconds):
    try:
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

        # Calculate start and end frames for overlaying person video
        background_duration = cap_background.get(cv2.CAP_PROP_FRAME_COUNT) / background_fps
        person_duration = cap_processed_person.get(cv2.CAP_PROP_FRAME_COUNT) / person_fps
        start_frame = int(background_fps * start_time_seconds)
        end_frame = start_frame + int(person_fps * person_duration)

        frame_number = 0

        while True:
            ret_background, background_frame = cap_background.read()

            if not ret_background:
                break

            frame_number += 1

            if frame_number >= start_frame:
                ret_person, person_frame = cap_processed_person.read()

                if not ret_person:
                    break

                person_frame = cv2.resize(person_frame, (overlay_width, overlay_height))

                mask = np.zeros((overlay_height, overlay_width), dtype=np.uint8)
                cv2.circle(mask, (overlay_width // 2, overlay_height // 2), min(overlay_width, overlay_height) // 2, 255, -1)

                roi = background_frame[0:overlay_height, 0:overlay_width]
                cv2.bitwise_and(person_frame, person_frame, mask=mask, dst=roi)

            output_video.write(background_frame)

        cap_background.release()
        cap_processed_person.release()
        output_video.release()
        cv2.destroyAllWindows()

        # Merge audio with the final video
        output_with_audio_path = "output_with_audio.mp4"
        merge_audio_with_video(output_path, person_video_path, background_path, output_with_audio_path)

        # Cleanup temporary files
        os.remove(processed_person_output_path)

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
person_video_path = "Videom.mp4"
background_path = "Video90.mp4"
output_path = "Joined_video2.mp4"
start_time_seconds = 2  # Start overlaying person video at 1 minute into the background video

process_and_overlay_videos(person_video_path, background_path, output_path, start_time_seconds)
