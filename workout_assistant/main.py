import argparse
import logging
import traceback
from uuid import uuid4
import aiortc
import av
from dotenv import load_dotenv
from aiortc.contrib.media import MediaPlayer
import numpy as np
from PIL import Image

from utils import create_user, open_browser
from getstream.stream import Stream
from getstream.video import rtc

import asyncio
import os

from getstream.video.call import Call
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig, TrackType

from ultralytics import solutions

# Configure logging for the Stream SDK
logging.basicConfig(level=logging.ERROR)

INPUT_FILE = ""

class CustomVideoTrack(aiortc.VideoStreamTrack):
    frame_q = asyncio.Queue()
    last_frame = Image.new('RGB', (1920, 1080), color='black')

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

async def process_frame(gym: solutions.AIGym, frame: Image.Image, output_frame_q: asyncio.Queue):
    results = gym.process(np.asarray(frame))
    if results.plot_im.size > 0:
        output_frame_q.put_nowait(Image.fromarray(results.plot_im))

async def analyse_video(input_frame_q: asyncio.Queue, output_frame_q: asyncio.Queue):
    pts = [5,7,9]
    gym = solutions.AIGym(
        model="yolo11n-pose.pt",
        show=False,
        kpts=pts,
        up_angle=165.0,
        down_angle=100.0,
    )

    print(f"Gym: {gym}")

    curr_w = 1280
    curr_h = 720
    frame_count = 0
    while True:
        try:
            frame = await asyncio.wait_for(input_frame_q.get(), timeout=1)
            if frame:
                if frame.width != curr_w or frame.height != curr_h:
                    # create a new instance when resolution changes
                    gym = solutions.AIGym(
                        model="yolo11n-pose.pt",
                        show=False,
                        kpts=pts,
                        up_angle=165.0,
                        down_angle=100.0,
                    )
                    curr_w = frame.width
                    curr_h = frame.height
                if args.debug:
                    with open(f"debug/image_{frame_count}.png", "wb") as f:
                        frame.save(f)
                print(f"Processing frame: {frame.width}x{frame.height}")
                await process_frame(gym, frame, output_frame_q)
                frame_count += 1
        except asyncio.TimeoutError:
            pass

async def on_track_added(track_id, track_type, user, target_user_id, ai_connection, output_frame_q):
    """Handle a new track being added to the ai connection."""
    print(f"Track added: {track_id} for user {user} of type {track_type}")
    if track_type != "video" or user.user_id != target_user_id:
        return

    input_frame_q = asyncio.Queue()
    asyncio.create_task(analyse_video(input_frame_q, output_frame_q))

    track = ai_connection.subscriber_pc.add_track_subscriber(track_id)
    if track:
        while True:
            try:
                video_frame: aiortc.mediastreams.VideoFrame = await track.recv()
                if video_frame:
                    print(f"Video frame received: {video_frame.time} - {video_frame.format}")
                    img = video_frame.to_image()
                    input_frame_q.put_nowait(img)
            except Exception as e:
                print(f"Error receiving track: {e} - {type(e)}")
                break
    else:
        print(f"Track not found: {track_id}")

async def publish_media(call: Call, user_id: str, player: MediaPlayer):
    try:
        async with await rtc.join(call, user_id) as connection:
            await connection.add_tracks(audio=player.audio, video=player.video)

            await connection.wait()
    except Exception as e:
        print(f"Error: {e} - stacktrace: {traceback.format_exc()}")

async def main():

    print(f"Workout Assistant Example")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    input_media_player = None

    if INPUT_FILE != "":
        if not os.path.exists(INPUT_FILE):
            print(f"Input file not found: {INPUT_FILE}")
            return None
        input_media_player = MediaPlayer(INPUT_FILE, loop=False, decode=False)
        if not (input_media_player.audio or input_media_player.video):
            print("No audio/video track found in input file")
            return None

    # Initialize Stream client
    client = Stream.from_env()

    # Create a unique call
    call_id = f"workout-ai-example-{str(uuid4())}"
    call = client.video.call("default", call_id)
    print(f"Call ID: {call_id}")

    viewer_user_id = f"viewer-{uuid4()}"
    player_user_id = f"player-{str(uuid4())[:8]}"
    ai_user_id = f"ai-{str(uuid4())[:8]}"

    create_user(client, player_user_id, "Player")
    if input_media_player:
        create_user(client, viewer_user_id, "Viewer")
        token = client.create_token(viewer_user_id, expiration=3600)
    else:
        token = client.create_token(player_user_id, expiration=3600)
    create_user(client, ai_user_id, "AI Bot")

    # Create the call
    call.get_or_create(data={"created_by_id": "workout-example"})

    try:
        # Join all bots first and add their tracks
        async with await rtc.join(
            call,
            ai_user_id,
            subscription_config=SubscriptionConfig(
                default=TrackSubscriptionConfig(track_types=[
                    TrackType.TRACK_TYPE_VIDEO,
                    # TrackType.TRACK_TYPE_SCREEN_SHARE,
                ]
            ))) as ai_connection:

            video = CustomVideoTrack()
            await ai_connection.add_tracks(video=video)

            ai_connection.on(
                "track_added",
                lambda track_id, track_type, user: asyncio.create_task(
                    on_track_added(track_id, track_type, user, player_user_id, ai_connection, video.frame_q)
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
    args_parser = argparse.ArgumentParser(description="Workout AI Example")
    args_parser.add_argument(
        "-i", "--input-file",
        required = False,
        help = "Input file with video and audio tracks to publish. " \
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
