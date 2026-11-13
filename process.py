import whisperx
import textwrap
from vtt_synced_voice import transcribe
from janome.tokenizer import Tokenizer
from jamdict import Jamdict
import ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import subprocess
import cv2
import numpy as np
import os

from fastapi import FastAPI, UploadFile, File
app = FastAPI()

IN_BETWEEN = "assets/in-between"
OUTPUT_DIR = "assets/output"
INPUT_DIR = "assets/input"
os.makedirs(IN_BETWEEN, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

STOP_POS_MAJOR = {
    "助詞",      # particles: は, が, を, に, で
    "助動詞",    # auxiliary verbs: です, ます, た, ない
    "記号",      # symbols and punctuation
    "連体詞",    # pre-noun adjectivals: この, その, ある
    "接続詞",    # conjunctions: そして, しかし
    "感動詞",    # interjections: あっ, えー
    "フィラー",  # fillers: あの, えっと
}

# (major, minor) combos to drop even though the major class is kept
STOP_POS_MINOR = {
    ("名詞", "非自立"),   # こと, もの, ため, わけ
    ("名詞", "代名詞"),   # pronouns: これ, それ, 私
    ("名詞", "数"),       # numbers
    ("名詞", "接尾"),     # suffixes: さん, 的
    ("動詞", "非自立"),   # auxiliary verbs: いる, くる, しまう
    ("動詞", "接尾"),
    ("形容詞", "非自立"),
    ("副詞", "助詞類接続"),
}

# Layer 2: specific words POS filtering misses
STOP_WORDS = {
    "する", "いる", "ある", "なる", "できる", "れる", "られる",
    "ない", "ん", "の", "こと", "もの", "これ", "それ", "あれ",
    "ここ", "そこ", "どこ", "よう", "そう", "ほう", "ところ",
}

@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    input_file = f"assets/input/{file.filename}"
    with open(input_file, "wb") as f: f.write(await file.read())

    stem, _ext = os.path.splitext(input_file)
    input_name = os.path.basename(stem)
    input_name = input_name.replace(" ", "_")

    def extract_audio(input_file, output_file="assets/in-between/extracted_audio.wav"):
        """Extract the default audio stream, normalized for WhisperX (16 kHz mono PCM)."""
        cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-vn",
            "-map", "0:a:0?",   
            "-ac", "1",        
            "-ar", "16000",   
            "-c:a", "pcm_s16le",   
            output_file,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg failed extracting audio from {input_file!r}:\n"
                f"{e.stderr.decode(errors='replace')}"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg not found on PATH.") from e
        return output_file

    audio_file = extract_audio(input_file)

    jam = Jamdict()
    t = Tokenizer()
    content = []
    
    transcribe(
        audio_file=audio_file,
        output_file="assets/in-between/output.vtt",
        language="ja",
        model="base",
        max_gap_seconds=0.4,
        min_cue_chars=50,
        merge_sentences=False
    )

    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    screen_width = int(cap.get(3))
    screen_height = int(cap.get(4))
    size = (screen_width, screen_height)
    # converting transcriptions into dicts for timestamps and content
    
    with open("assets/in-between/output.vtt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        current_segment = {"timestamps":"","text":"","words":[]}
        for line in lines:
            if line == "WEBVTT":
                continue
            elif '-->' in line:
                time_stamps = line.split(" --> ")
                time_stamps = [[float(x) for x in  i.split(":")] for i in time_stamps]
                #loop through each timestamp in list
                for idx, i in enumerate(time_stamps):
                    sum = 0
                    for segment_idx, segment in enumerate(i):
                        sum += segment*(60**(2-segment_idx))
                    time_stamps[idx] = int(sum * fps)

                current_segment["timestamps"] = time_stamps
            else:
                current_segment["text"] = line
                words = []
                # seperate by words
                wordcount = 0
                for token in t.tokenize(line):
                    try:
                        float(token.base_form)
                        continue
                    except ValueError:
                        if (token.part_of_speech.split(",")[0] in STOP_POS_MAJOR
                            or tuple(token.part_of_speech.split(",")[:2]) in STOP_POS_MINOR
                            or token.base_form in STOP_WORDS
                            or token.base_form == "*"):
                            continue
                        jamQ = jam.lookup(token.base_form).entries
                        if jamQ:
                            # dictionary lookup
                            result = jamQ[0]
                            translations = str(result.senses[0].gloss[0])
                            if len(result.senses[0].gloss) > 1:
                                for translation in result.senses[0].gloss[1:]:
                                    translations = translations + ", " + str(translation)
                                wordcount += 1
                        else:
                            continue
                        word = {"word":str(token.base_form),"kana":str(result.kana_forms[0]),"translation":translations}
                        words.append(word)
                    if wordcount >= 7: 
                        break
                current_segment["words"] = words
            if current_segment["timestamps"] != "" and current_segment["text"] != "" and current_segment["words"] != []:
                content.append(current_segment)
                current_segment = {"timestamps":"","text":"","words":""}

    # landscape/portrait support, in landscape we use height in portrait we use width
    base_size = min(screen_width, screen_height)
    dictionary_font_size = int(base_size * 0.04)
    subtitle_font_size = int(base_size * 0.05) * 1.25
    
    def make_fonts(base_size):
        return (
            ImageFont.truetype(FONT_PATH, int(base_size * 0.04)),   # dictionary
            ImageFont.truetype(FONT_PATH, int(base_size * 0.05 * 1.25)),  # subtitle
        )
    
    def composite_roi(frame_bgr, overlay_rgba, bbox):
        if overlay_rgba is None:
            return frame_bgr
        x0, y0, x1, y1 = bbox
        h, w = frame_bgr.shape[:2]
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return frame_bgr

        ov = np.asarray(overlay_rgba)[y0:y1, x0:x1]   # (h, w, 4) RGBA
        alpha = ov[..., 3:4].astype(np.float32) / 255.0
        rgb = ov[..., :3].astype(np.float32)           # RGB
        roi = frame_bgr[y0:y1, x0:x1].astype(np.float32)  # BGR
        # blend: out_bgr = roi*(1-a) + rgb_bgr*a  -> swap RGB<->BGR
        blended = roi * (1.0 - alpha) + rgb[..., ::-1] * alpha
        frame_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)
        return frame_bgr

    def build_overlay(screen_width, screen_height, subtitle, entries, subtitle_font, dict_font):
        overlay = Image.new('RGBA', (screen_width, screen_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x0, y0, x1, y1 = screen_width, screen_height, 0, 0

        def mark(bbox):
            nonlocal x0, y0, x1, y1
            x0 = min(x0, bbox[0]); y0 = min(y0, bbox[1])
            x1 = max(x1, bbox[2]); y1 = max(y1, bbox[3])

        WHITE = (255, 255, 255, 255)
        BLACK_HALF = (0, 0, 0, int(255 * 0.5))
        PAD = 3

        if subtitle:
            char_bbox = draw.textbbox((0, 0), "あ", font=subtitle_font)
            char_width = char_bbox[2] - char_bbox[0]
            lines = textwrap.wrap(subtitle, width=int(screen_width / char_width))

            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=subtitle_font)
                width = bbox[2] - bbox[0]
                textX = (screen_width - width) // 2
                textY = 10 + (subtitle_font.size + 6) * i
                bb = draw.textbbox((textX, textY), line, font=subtitle_font)
                bb = (bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD)
                draw.rectangle(bb, fill=BLACK_HALF)
                draw.text((textX, textY), line, fill=WHITE, font=subtitle_font)
                mark(bb)

        if entries:
            textY = subtitle_font.size * 1.1
            for entry in entries:
                line = entry["word"] + ": " + entry["kana"]
                textY += dict_font.size + 6
                bb = draw.textbbox((20, textY), line, font=dict_font)
                bb = (bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD)
                draw.rectangle(bb, fill=BLACK_HALF)
                draw.text((20, textY), line, fill=WHITE, font=dict_font)
                mark(bb)

                textY += dict_font.size
                bb = draw.textbbox((20, textY), entry["translation"], font=dict_font)
                bb = (bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD)
                draw.rectangle(bb, fill=BLACK_HALF)
                draw.text((20, textY), entry["translation"], fill=WHITE, font=dict_font)
                mark(bb)
                textY += dict_font.size + 3

        if x1 <= x0 or y1 <= y0:
            return None, None
        return overlay, (int(x0), int(y0), int(x1), int(y1))
        
    FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

    dict_font, sub_font = make_fonts(base_size)

    cues = []
    for seg in content:
        ov, bb = build_overlay(screen_width, screen_height, seg["text"], seg["words"],
                            sub_font, dict_font)
        cues.append({
            "start": seg["timestamps"][0],
            "end":   seg["timestamps"][1],
            "overlay": None if ov is None else np.asarray(ov),   # RGBA ndarray
            "bbox": bb,
        })
    cues.sort(key=lambda c: c["start"])

    # Pre-place logo bounds
    logo_roi = None
    thumb = cv2.imread("assets/images/logo.png")
    if thumb is not None:
        thumb_w = int(screen_width * 0.15)
        thumb_h = int(thumb.shape[0] * (thumb_w / thumb.shape[1]))
        thumb = cv2.resize(thumb, (thumb_w, thumb_h))
        x_offset = screen_width - thumb_w - 20
        y_offset = screen_height - thumb_h - 20
        logo_roi = (x_offset, y_offset, x_offset + thumb_w, y_offset + thumb_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(f"assets/in-between/{input_name}_output_temp.mp4",
                            fourcc, fps, (screen_width, screen_height))

    frame_number = 0
    cue_idx = 0
    active = None 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # advance cue pointer (cues are sorted by start frame)
        while cue_idx < len(cues) and frame_number > cues[cue_idx]["end"]:
            cue_idx += 1

        if cue_idx < len(cues) and cues[cue_idx]["start"] <= frame_number <= cues[cue_idx]["end"]:
            cue = cues[cue_idx]
            if cue["overlay"] is not None:
                composite_roi(frame, cue["overlay"], cue["bbox"])

        if logo_roi is not None:
            lx0, ly0, lx1, ly1 = logo_roi
            frame[ly0:ly1, lx0:lx1] = thumb

        output.write(frame)
        frame_number += 1

    output.release()
    cap.release()

    os.makedirs("assets/output", exist_ok=True)

    audio = ffmpeg.input(audio_file)
    video = ffmpeg.input(f"assets/in-between/{input_name}_output_temp.mp4")
    vtt   = ffmpeg.input("assets/in-between/output.vtt")  # The Whisper output

    ori_final_name = f"assets/output/{input_name}_output"
    final_name = ori_final_name + ".mp4"
    count = 0

    while os.path.exists(final_name):
        count += 1
        final_name = ori_final_name + str(count) + ".mp4"

    try:
        final = ffmpeg.output(
            video, 
            audio, 
            vtt,
            final_name,
            vcodec="copy",      # Keep video as-is
            acodec="aac",       # Re-encode audio (as you already do)
            **{"c:s": "mov_text"} # Convert VTT to MP4-compatible subtitles
        )
    except:
        final = ffmpeg.output(
            video,
            audio,
            final_name,
            vcodec="copy",
            acodec="aac",
            shortest=None,
        )

    try:
        final.run(overwrite_output=True)
    except ffmpeg.Error as e:
        print("=== FFmpeg Error ===")
        print("stderr:", e.stderr)
        print("stdout:", e.stdout)
        # Fallback: re-run with cmd string to see real output
        print("cmd:", final.compile())
        raise

    if os.path.exists(audio_file):
        os.remove(audio_file)
        
    if os.path.exists(f"assets/in-between/{input_name}_output_temp.mp4"):
        os.remove(f"assets/in-between/{input_name}_output_temp.mp4")
    
    return {"status": "success", "output_file": final_name}