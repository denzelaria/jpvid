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
    
    def draw_entries(img, entries, font_size, color):    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        font = ImageFont.truetype(font_path, font_size)
        
        textY = subtitle_font_size * 1.2
        for i, entry in enumerate(entries):
            line = entry["word"] + ": " + entry["kana"]
            textY += font_size + 6
            bbox = draw.textbbox((20, textY), line, font=font)
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=(0,0,0,int(255*0.5)), width=3)
            draw.text((20, textY), line, fill=color[::-1], font=font) 
            
            textY += font_size
            bbox = draw.textbbox((20, textY), entry["translation"], font=font)
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=(0,0,0,int(255*0.5)), width=3)
            draw.text((20, textY), entry["translation"], fill=color[::-1], font=font)
            
            textY += font_size + 3
        
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay)
    
        return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    def draw_subtitle(img, text, font_size, color):    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        overlay = Image.new('RGBA', img_pil.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        font = ImageFont.truetype(font_path, font_size)
        char_bbox = draw.textbbox((0,0), "あ", font=font)
        char_width = char_bbox[2] - char_bbox[0]
        lines = textwrap.wrap(text, width=int(screen_width / (char_width))) 
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0,0), line, font=font)
            width = bbox[2] - bbox[0]
            textX = (screen_width - (width)) / 2
            textY = 10 + (font_size+6)*i
            bbox = draw.textbbox((textX,textY), line, font=font)
            draw.rectangle([bbox[0],bbox[1],bbox[2],bbox[3]], fill=(0,0,0,int(255*0.5)), width=3)
            draw.text((textX, textY), line, fill=color[::-1], font=font)
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay)
        return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    frame_number = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(f"assets/in-between/{input_name}_output_temp.mp4", fourcc, fps, (screen_width, screen_height))
    thumb = cv2.imread("assets/images/logo.png")
    if thumb is not None:
        # Resize it so it's 15% of the screen width
        thumb_w = int(screen_width * 0.15)
        thumb_h = int(thumb.shape[0] * (thumb_w / thumb.shape[1]))
        thumb = cv2.resize(thumb, (thumb_w, thumb_h))
        
        x_offset = screen_width - thumb_w - 20  # 20px from the right edge
        y_offset = screen_height - thumb_h - 20 # 20px from the bottom edge

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            current_subtitle = None
            current_info = None
            for segment in content:
                timestamps = segment["timestamps"]
                if frame_number >= timestamps[0] and frame_number <= timestamps[1]:
                    current_subtitle = segment["text"]
                    current_info = segment["words"]
            if current_subtitle:
                frame = draw_subtitle(frame, current_subtitle, subtitle_font_size, (255,255,255))
            if current_info:
                frame = draw_entries(frame, current_info, dictionary_font_size, (255,255,255))
            frame_number += 1
            if thumb is not None:
                frame[y_offset:y_offset+thumb_h, x_offset:x_offset+thumb_w] = thumb
            output.write(frame)
        else:
            break

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