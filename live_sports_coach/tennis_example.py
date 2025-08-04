#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import traceback
from uuid import uuid4
import aiortc
import av
from dotenv import load_dotenv
from aiortc.contrib.media import MediaPlayer
from PIL import Image
from google.genai.types import Modality, MediaResolution, TurnCoverage, ActivityHandling

from utils import create_user, open_browser
from getstream.stream import Stream
from getstream.video import rtc

import asyncio
import os
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types
import websockets

from getstream.video.call import Call
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig, TrackType

from ultralytics import YOLO
import cv2

g_session = None

# Configure logging for the Stream SDK
logging.basicConfig(level=logging.ERROR)

# Audio playback/recording constants
RECEIVE_SAMPLE_RATE = 24000

# Performance constants
GEMINI_FPS = 1      # Send frames to Gemini at 1fps
MAX_QUEUE_SIZE = 30  # Prevent memory buildup

INPUT_FILE = ""

class PoseVideoTrack(aiortc.VideoStreamTrack):
    """Simplified video track with real-time pose detection"""
    
    def __init__(self):
        super().__init__()
        self.frame_q = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.last_frame = Image.new('RGB', (1920, 1080), color='black')
        # Load only pose model
        self.pose_model = YOLO('yolo11n-pose.pt')
        self.pose_model.to('cpu')  # Ensure CPU inference for stability
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pose_processor")
        self._shutdown = False
    
    def cleanup(self):
        """Clean up resources"""
        self._shutdown = True
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    async def recv(self) -> av.frame.Frame:
        try:
            frame = await asyncio.wait_for(self.frame_q.get(), timeout=0.02)
            if frame:
                self.last_frame = frame
        except asyncio.TimeoutError:
            pass
        pts, time_base = await self.next_timestamp()
        av_frame = av.VideoFrame.from_image(self.last_frame)
        av_frame.pts = pts
        av_frame.time_base = time_base
        return av_frame

    def _process_pose_sync(self, frame_array: np.ndarray) -> np.ndarray:
        """Synchronous pose processing for thread pool"""
        try:
            if self._shutdown:
                return frame_array
            
            # Run pose detection on every frame
            pose_results = self.pose_model(
                frame_array, 
                verbose=False, 
                imgsz=416,  # Smaller image size for speed
                conf=0.5,   # Higher confidence threshold
                device='cpu'
            )
            
            if not pose_results:
                return frame_array
                
            # Apply pose results to current frame
            annotated_frame = frame_array.copy()
            
            # Process each detected person
            for person_idx, result in enumerate(pose_results):
                if not result.keypoints:
                    continue
                    
                keypoints = result.keypoints
                if keypoints is not None and len(keypoints.data) > 0:
                    kpts = keypoints.data[0].cpu().numpy()  # Get person's keypoints
                    

                    
                    # Draw keypoints
                    for i, (x, y, conf) in enumerate(kpts):
                        if conf > 0.5:  # Only draw confident keypoints
                            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    
                    # Draw skeleton connections with enhanced hand/wrist tracking
                    connections = [
                        (0, 1), (0, 2), (1, 3), (2, 4),  # Head connections
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arm connections
                        (5, 11), (6, 12), (11, 12),  # Torso connections
                        (11, 13), (13, 15), (12, 14), (14, 16),  # Leg connections
                        # Enhanced hand and wrist connections for tennis
                        (9, 15), (15, 16), (16, 17), (17, 18), (18, 19),  # Right hand thumb
                        (9, 20), (20, 21), (21, 22), (22, 23), (23, 24),  # Right hand index
                        (9, 25), (25, 26), (26, 27), (27, 28), (28, 29),  # Right hand middle
                        (9, 30), (30, 31), (31, 32), (32, 33), (33, 34),  # Right hand ring
                        (9, 35), (35, 36), (36, 37), (37, 38), (38, 39),  # Right hand pinky
                        # Left hand connections (if available)
                        (8, 45), (45, 46), (46, 47), (47, 48), (48, 49),  # Left hand thumb
                        (8, 50), (50, 51), (51, 52), (52, 53), (53, 54),  # Left hand index
                        (8, 55), (55, 56), (56, 57), (57, 58), (58, 59),  # Left hand middle
                        (8, 60), (60, 61), (61, 62), (62, 63), (63, 64),  # Left hand ring
                        (8, 65), (65, 66), (66, 67), (67, 68), (68, 69),  # Left hand pinky
                    ]
                    
                    for start_idx, end_idx in connections:
                        if start_idx < len(kpts) and end_idx < len(kpts):
                            x1, y1, c1 = kpts[start_idx]
                            x2, y2, c2 = kpts[end_idx]
                            if c1 > 0.5 and c2 > 0.5:
                                # Use different colors for different body parts
                                if start_idx >= 9 and start_idx <= 39:  # Right hand
                                    color = (0, 255, 255)  # Cyan for right hand
                                elif start_idx >= 40 and start_idx <= 69:  # Left hand
                                    color = (255, 255, 0)  # Yellow for left hand
                                else:  # Main body
                                    color = (255, 0, 0)  # Blue for main skeleton
                                cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Highlight wrist positions specifically for tennis analysis
                    wrist_keypoints = [9, 10]  # Right and left wrists
                    for wrist_idx in wrist_keypoints:
                        if wrist_idx < len(kpts):
                            x, y, conf = kpts[wrist_idx]
                            if conf > 0.5:
                                # Draw larger, more visible wrist markers
                                cv2.circle(annotated_frame, (int(x), int(y)), 8, (0, 0, 255), -1)  # Red wrist markers
                                cv2.circle(annotated_frame, (int(x), int(y)), 10, (255, 255, 255), 2)  # White outline
                                
                                # Add wrist labels
                                wrist_label = "R Wrist" if wrist_idx == 9 else "L Wrist"
                                cv2.putText(annotated_frame, wrist_label, (int(x) + 15, int(y) - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.putText(annotated_frame, wrist_label, (int(x) + 15, int(y) - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            return annotated_frame
        except Exception as e:
            print(f"Error in pose processing: {e}")
            return frame_array
    
    async def process_pose_async(self, frame: Image.Image) -> Image.Image:
        """Async wrapper for pose processing"""
        frame_array = np.array(frame)
        loop = asyncio.get_event_loop()
        
        try:
            processed_array = await loop.run_in_executor(
                self.executor, self._process_pose_sync, frame_array
            )
            return Image.fromarray(processed_array)
        except Exception as e:
            print(f"Error in async pose processing: {e}")
            return frame

async def play_audio(audio_in_queue, audio_track):
    """Play audio responses from Gemini Live"""
    while True:
        bytestream = await audio_in_queue.get()
        await audio_track.write(bytestream)

async def gather_responses(session: "genai.aio.live.Session", output: Path, audio_in_queue):
    """Collect model responses and append parsed JSON to output list."""
    output_buffer = ""
    input_buffer = ""
    try:
        while True:
            turn = session.receive()
            async for response in turn:
                if data := response.data:
                    audio_in_queue.put_nowait(data)
                    continue
                if response.server_content.input_transcription:
                    input_buffer += response.server_content.input_transcription.text
                if response.server_content.output_transcription:
                    output_buffer += response.server_content.output_transcription.text
            print(f"\nInput transcript: {input_buffer}")
            print(f"\nOutput transcript: {output_buffer}")
            if args.debug:
                with open(output, "a") as f:
                    f.write(output_buffer)
                    f.write("\n")
            output_buffer = ""
    except websockets.exceptions.ConnectionClosedOK:
        print("Connection closed OK")
    except asyncio.CancelledError:
        print("Connection cancelled")
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    except Exception as e:
        print(f"Error: {e}")
        raise e

async def send_pose_frames_to_gemini(session, pose_frame_queue):
    """Send pose-annotated frames to Gemini Live at controlled rate"""
    last_send_time = 0
    
    while True:
        try:
            current_time = time.time()
            
            # Only send if enough time has passed (1fps rate limiting)
            if current_time - last_send_time >= (1.0 / GEMINI_FPS):
                try:
                    # Get the most recent frame, discard older ones
                    pose_frame = None
                    while not pose_frame_queue.empty():
                        try:
                            pose_frame = pose_frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    
                    if pose_frame and session:
                        await session.send_realtime_input(media=pose_frame)
                        print(f"Sent pose frame to Gemini Live (queue size: {pose_frame_queue.qsize()})")
                        last_send_time = current_time
                        
                except asyncio.QueueEmpty:
                    pass
            
            await asyncio.sleep(0.1)  # Short sleep to prevent busy waiting
            
        except Exception as e:
            print(f"Error sending pose frame to Gemini: {e}")
            await asyncio.sleep(1.0)

async def on_track_added(track_id, track_type, user, target_user_id, ai_connection, audio_in_queue, pose_track, audio_track):
    """Handle a new track being added to the ai connection."""
    global g_session
    print(f"Track added: {track_id} for user {user} of type {track_type}")
    
    if track_type != "video":
        return
    if user.user_id != target_user_id:
        print(f"User {target_user_id} does not belong to user {user.user_id}")
        return

    track = ai_connection.subscriber_pc.add_track_subscriber(track_id)

    if track:
        if not g_session:
            client = genai.Client(http_options={"api_version": "v1beta"}, api_key=os.getenv("GOOGLE_API_KEY"))
            PROMPT = """
            You are a professional tennis coach and expert.
            You will be given video frames from a tennis practice session showing a player's serve technique with pose detection overlay at 1 frame per second.
            The pose detection shows key body points and skeletal structure to help analyze form.
            
            Analyze the player's tennis serve technique focusing on:
            - Stance and foot positioning
            - Racket grip and preparation
            - Ball toss technique and timing
            - Body rotation and weight transfer
            - Follow-through and finish position
            - Overall serve mechanics and form
            
            Provide specific, actionable coaching feedback to help improve their tennis serve.
            Be encouraging but direct about areas for improvement.
            
            DO NOT repeat the same feedback unless the player makes changes.
            DO NOT provide general tennis information - focus only on what you observe.
            DO NOT greet the user before they speak.
            """

            gemini_config: types.LiveConnectConfig = types.LiveConnectConfig(
                response_modalities=[Modality.AUDIO],
                speech_config=types. SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Aoede",
                        )
                    ),
                ),
                temperature=0.1,
                system_instruction=PROMPT,
                media_resolution=MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                context_window_compression=types.ContextWindowCompressionConfig(
                    trigger_tokens=25600,
                    sliding_window=types.SlidingWindow(
                        target_tokens=12800
                    )
                ),
                input_audio_transcription=types.AudioTranscriptionConfig(),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                realtime_input_config=types.RealtimeInputConfig(
                    turn_coverage=TurnCoverage.TURN_INCLUDES_ALL_INPUT,
                ),
            )

            async with client.aio.live.connect(
                    model="models/gemini-live-2.5-flash-preview",
                    config=gemini_config,
            ) as session:
                g_session = session
                
                # Create queue for pose-annotated frames to send to Gemini
                pose_frame_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
                
                # Start tasks
                asyncio.create_task(gather_responses(session, Path("debug/tennis_analysis.txt"), audio_in_queue))
                asyncio.create_task(play_audio(audio_in_queue, audio_track))
                asyncio.create_task(send_pose_frames_to_gemini(session, pose_frame_queue))

                frame_count = 0
                
                while True:
                    try:
                        video_frame: aiortc.mediastreams.VideoFrame = await track.recv()
                        if video_frame:
                            img = video_frame.to_image()
                            
                            # Process every frame with pose detection
                            pose_annotated_frame = await pose_track.process_pose_async(img)
                            
                            # Update display queue with processed frame
                            try:
                                pose_track.frame_q.put_nowait(pose_annotated_frame)
                            except asyncio.QueueFull:
                                try:
                                    pose_track.frame_q.get_nowait()
                                    pose_track.frame_q.put_nowait(pose_annotated_frame)
                                except asyncio.QueueEmpty:
                                    pass
                            
                            # Add to Gemini queue for analysis
                            try:
                                pose_frame_queue.put_nowait(pose_annotated_frame)
                            except asyncio.QueueFull:
                                # Remove oldest frame and add new one
                                try:
                                    pose_frame_queue.get_nowait()
                                    pose_frame_queue.put_nowait(pose_annotated_frame)
                                except asyncio.QueueEmpty:
                                    pass
                            
                            if args.debug:
                                pose_annotated_frame.save(f"debug/tennis_frame_{frame_count}.png")
                            
                            frame_count += 1
                            
                    except Exception as e:
                        print(f"Error receiving track: {e} - {type(e)}")
                        break
                
                # Clean up pose track resources
                pose_track.cleanup()
    else:
        print(f"Track not found: {track_id}")

async def publish_media(call: Call, user_id: str, player: MediaPlayer):
    """Publish media from input file"""
    try:
        async with await rtc.join(call, user_id) as connection:
            await connection.add_tracks(audio=player.audio, video=player.video)
            await connection.wait()
    except Exception as e:
        print(f"Error: {e} - stacktrace: {traceback.format_exc()}")

async def main():
    print(f"Tennis Coach AI Example")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    input_media_player = None

    if INPUT_FILE != "":
        if not os.path.exists(INPUT_FILE):
            print(f"Input file not found: {INPUT_FILE}")
            return None
        input_media_player = MediaPlayer(INPUT_FILE, loop=False)
        if not (input_media_player.audio or input_media_player.video):
            print("No audio/video track found in input file")
            return None

    # Initialize Stream client
    client = Stream.from_env()

    # Create a unique call
    call_id = f"tennis-ai-example-{str(uuid4())}"
    call = client.video.call("default", call_id)
    print(f"Call ID: {call_id}")

    viewer_user_id = f"viewer-{uuid4()}"
    player_user_id = f"player-{str(uuid4())[:8]}"
    ai_user_id = f"ai-{str(uuid4())[:8]}"

    create_user(client, player_user_id, "Tennis Player")
    if input_media_player:
        create_user(client, viewer_user_id, "Viewer")
        token = client.create_token(viewer_user_id, expiration=3600)
    else:
        token = client.create_token(player_user_id, expiration=3600)
    create_user(client, ai_user_id, "Tennis Coach AI")

    # Create the call
    call.get_or_create(data={"created_by_id": "tennis-ai-example"})

    try:
        # Join all bots first and add their tracks
        async with await rtc.join(
            call,
            ai_user_id,
            subscription_config=SubscriptionConfig(
                default=TrackSubscriptionConfig(track_types=[
                    TrackType.TRACK_TYPE_VIDEO,
                ]
            ))) as ai_connection:
            
            # Create pose video track for displaying pose-annotated video
            pose_track = PoseVideoTrack()
            
            # Create audio queue for playback and model responses
            audio_in_queue = asyncio.Queue()
            
            # Initialize audio track for Gemini responses
            audio_track = AudioStreamTrack(framerate=RECEIVE_SAMPLE_RATE, stereo=False, format="s16")
            await ai_connection.add_tracks(video=pose_track, audio=audio_track)
            
            # Define event handler with access to pose_track and audio_track
            async def handle_track_added(track_id, track_type, user):
                await on_track_added(track_id, track_type, user, player_user_id, ai_connection, audio_in_queue, pose_track, audio_track)
            
            ai_connection.on(
                "track_added",
                lambda track_id, track_type, user: asyncio.create_task(
                    handle_track_added(track_id, track_type, user)
                )
            )

            # Handle audio from player for Gemini Live
            @ai_connection.on("audio")
            async def on_audio(pcm: PcmData, user):
                if user.user_id == player_user_id and g_session:
                    await g_session.send_realtime_input(
                        audio=types.Blob(
                            data=pcm.samples.astype(np.int16).tobytes(), 
                            mime_type="audio/pcm;rate=48000"
                        )
                    )
            
            open_browser(client.api_key, token, call_id)

            await asyncio.sleep(3)

            if input_media_player:
                asyncio.create_task(publish_media(call, player_user_id, input_media_player))

            await ai_connection.wait()
    except Exception as e:
        print(f"Error: {e} - stacktrace: {traceback.format_exc()}")
    finally:
        # Delete created users
        print("Deleting created users...")
        client.delete_users([player_user_id, ai_user_id, viewer_user_id])

    return None

if __name__ == "__main__":
    # Parse command line arguments
    args_parser = argparse.ArgumentParser(description="Tennis Coach AI Example")
    args_parser.add_argument(
        "-i", "--input-file",
        required=False,
        help="Input file with video and audio tracks to publish. " \
        "If an input file is specified, it will be used. Otherwise, " \
        "the bot will wait till a video track is published",
    )
    args_parser.add_argument(
        "-d", '--debug',
        action='store_true',
        help="Enable debug mode"
    )
    args = args_parser.parse_args()
    if args.input_file and args.input_file != "" and os.path.exists(args.input_file):
        INPUT_FILE = args.input_file
    if args.debug:
        os.makedirs("debug", exist_ok=True)
    asyncio.run(main())