import av
import argparse
import logging
import numpy as np
from PIL import Image
import traceback
from uuid import uuid4
import aiortc
from dotenv import load_dotenv
from aiortc.contrib.media import MediaPlayer

from utils import create_user, open_browser

import asyncio
import os

from getstream.stream import Stream
from getstream.video import rtc
from getstream.video.call import Call
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig, TrackType

import supervision as sv
from ultralytics import YOLO

g_session = None

# Configure logging for the Stream SDK
logging.basicConfig(level=logging.ERROR)

INPUT_FILE = ""

class CustomVideoTrack(aiortc.VideoStreamTrack):
    frame_q = asyncio.Queue()
    last_frame = Image.new('RGB', (1920, 1080), color='black')

    async def recv(self) -> av.frame.Frame:
        try:
            frame = await asyncio.wait_for(self.frame_q.get(), timeout=5)
            if frame:
                self.last_frame = frame
        except asyncio.TimeoutError:
            pass
        pts, time_base = await self.next_timestamp()
        av_frame = av.VideoFrame.from_image(self.last_frame)
        av_frame.pts = pts
        av_frame.time_base = time_base
        if args.debug:
            with open(f"debug/output_image_{pts}.png", "wb") as f:
                self.last_frame.save(f)
        return av_frame

async def analyse_frames(input_frame_q: asyncio.Queue, output_frame_q: asyncio.Queue):
    model = YOLO("yolo_custom_weights.pt", task="detect")
    tracker = sv.ByteTrack(
        minimum_consecutive_frames=15
    )
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    roi = np.array([[0, 450], [1280, 450], [1280, 720], [0, 720]])
    polygon_zone = sv.PolygonZone(polygon=roi)
    zone_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, thickness=4, text_thickness=4, text_scale=2)
    current_state = "Monitoring"
    new_state = None
    items = {}
    missing_items = []

    frame_count = 0
    while True:
        try:
            frame = await asyncio.wait_for(input_frame_q.get(), timeout=1)
            if frame:
                if args.debug:
                    with open(f"debug/input_image_{frame_count}.png", "wb") as f:
                        frame.save(f)
                results = model.predict(frame, conf=0.7, iou=0.3)
                if len(results) == 0:
                    continue
                results = results[0]
                # Use the original NumPy image from Ultralytics results (BGR)
                scene = results.orig_img.copy()
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[~np.isin(detections.class_id, [4, 5, 12, 20, 21, 24, 25, 31, 36])]
                detections = tracker.update_with_detections(detections)

                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    for tracker_id in detections.tracker_id:
                        print(f"Tracker ID: {tracker_id}")
                        if tracker_id not in items:
                            print(f"New item: {tracker_id}")
                            items[tracker_id] = {
                                "state": "New",
                                "missing_count": 0
                            }
                        elif items[tracker_id]["state"] == "New":
                            print(f"Item already exists: {tracker_id} -> Monitoring")
                            items[tracker_id]["state"] = "Monitoring"
                        elif items[tracker_id]["state"] == "Missing":
                            print(f"Missing item found again: {tracker_id} -> Monitoring")
                            items[tracker_id]["state"] = "Monitoring"
                            items[tracker_id]["missing_count"] -= 1
                for tracker_id, item in items.items():
                    print(f"Item: {tracker_id} -> {item['state']}")
                    if tracker_id not in detections.tracker_id:
                        print(f"Item not found: {tracker_id} -> Missing")
                        item["state"] = "Missing"
                        item["missing_count"] += 1
                    if item["state"] == "Missing" and item["missing_count"] > 10:
                        print(f"Item missing for too long: {tracker_id} -> Missing")
                        missing_items.append(tracker_id)

                if len(missing_items) > 0:
                    items = {k: v for k, v in items.items() if k not in missing_items}
                    print(f"Items: {items}")
                    print(f"Missing items: {missing_items}")
                labels = [
                    f"#{class_id} {tracker_id} {confidence:.2f}"
                    for class_id, tracker_id, confidence
                    in zip(detections.class_id, detections.tracker_id, detections.confidence)
                ]
                try:
                    labels = [
                        f"#{class_id} {tracker_id} {class_name} {confidence:.2f}"
                        for class_id, class_name, tracker_id, confidence
                        in zip(detections.class_id, detections.data["class_name"], detections.tracker_id, detections.confidence)
                    ]
                except Exception:
                    pass
                print(labels)
                annotated_frame = box_annotator.annotate(
                    scene=scene, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                print(polygon_zone.trigger(detections=detections))
                annotated_frame = zone_annotator.annotate(annotated_frame)
                if len(missing_items) > 0:
                    print(f"Items are missing: {missing_items}")
                    text_anchor = sv.Point(x=350, y=50)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame, 
                        text="Items are missing", 
                        text_anchor=text_anchor,
                        text_thickness=4,
                        text_scale=2,
                        text_color=sv.Color.RED,
                    )
                # Convert BGR ndarray back to PIL.Image in RGB for downstream video track
                output_frame_q.put_nowait(Image.fromarray(annotated_frame[:, :, ::-1]))
                frame_count += 1
        except asyncio.TimeoutError:
            pass

async def on_track_added(track_id, track_type, user, target_user_id, ai_connection, frame_q):
    """Handle a new track being added to the ai connection."""
    print(f"Track added: {track_id} for user {user} of type {track_type}")
    if track_type != "video":
        return
    if user.user_id != target_user_id:
        print(f"User {target_user_id} does not belong to user {user.user_id}")
        return

    track = ai_connection.subscriber_pc.add_track_subscriber(track_id)

    if track and track_type == "video":
        frame_count = 0
        input_frame_q = asyncio.Queue()
        asyncio.create_task(analyse_frames(input_frame_q, frame_q))
        while True:
            try:
                video_frame: aiortc.mediastreams.VideoFrame = await track.recv()
                if video_frame:
                    img = video_frame.to_image()
                    if args.debug:
                        with open(f"debug/image_{frame_count}.png", "wb") as f:
                            img.save(f)
                    input_frame_q.put_nowait(img)
                    frame_count += 1
            except Exception as e:
                print(f"Error receiving track: {e} - {type(e)}")
                break
    else:
        print(f"Track not found: {track_id}")

async def publish_media(call: Call, user_id: str, player: MediaPlayer):
    try:
        async with await rtc.join(call, user_id) as connection:
            await asyncio.sleep(3)

            await connection.add_tracks(video=player.video)

            await connection.wait()
    except Exception as e:
        print(f"Error: {e} - stacktrace: {traceback.format_exc()}")

async def main():

    print(f"AI Security Cam Example")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    input_media_player = None

    if INPUT_FILE != "":
        if not os.path.exists(INPUT_FILE):
            print(f"Input file not found: {INPUT_FILE}")
            return None
        input_media_player = MediaPlayer(INPUT_FILE, loop=False, decode=True)
        if not (input_media_player.audio or input_media_player.video):
            print("No audio/video track found in input file")
            return None

    # Initialize Stream client
    client = Stream.from_env()

    # Create a unique call
    call_id = f"ai-cam-example-{str(uuid4())}"
    call = client.video.call("default", call_id)
    print(f"Call ID: {call_id}")

    viewer_user_id = f"viewer-{uuid4()}"
    cam_user_id = f"cam-{str(uuid4())[:8]}"
    ai_user_id = f"ai-{str(uuid4())[:8]}"

    create_user(client, cam_user_id, "Camera")
    if input_media_player:
        create_user(client, viewer_user_id, "Viewer")
        token = client.create_token(viewer_user_id, expiration=3600)
    else:
        token = client.create_token(cam_user_id, expiration=3600)
    create_user(client, ai_user_id, "AI Bot")

    # Create the call
    call.get_or_create(data={"created_by_id": "ai-example"})

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
                    on_track_added(track_id, track_type, user, cam_user_id, ai_connection, video.frame_q)
                )
            )

            open_browser(client.api_key, token, call_id)

            await asyncio.sleep(2)

            if input_media_player:
                asyncio.create_task(publish_media(call, cam_user_id, input_media_player))

            await ai_connection.wait()
    except Exception as e:
        print(f"Error: {e} - stacktrace: {traceback.format_exc()}")
    finally:
        # Delete created users
        print("Deleting created users...")
        client.delete_users([cam_user_id, ai_user_id, viewer_user_id])

    return None

if __name__ == "__main__":
    # Parse command line arguments
    args_parser = argparse.ArgumentParser(description="AI Security Cam Example")
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
