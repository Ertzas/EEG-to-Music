import pygame
import random
import sys
import time
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog
from moviepy.editor import VideoFileClip
import os

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
FPS = 60

LANE_COUNT = 4
LANE_WIDTH = SCREEN_WIDTH // LANE_COUNT
NOTE_WIDTH = LANE_WIDTH - 20
NOTE_HEIGHT = 20
HIT_LINE_Y = SCREEN_HEIGHT - 150
NOTE_SPEED = 300  

KEYS = [pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f]  

BG_COLOR = (30, 30, 30)
LANE_COLOR = (50, 50, 50)
NOTE_COLOR = (200, 80, 80)
HIT_LINE_COLOR = (200, 200, 200)
TEXT_COLOR = (240, 240, 240)


class Note:
    def _init_(self, lane):
        self.lane = lane
        self.y = -NOTE_HEIGHT
        self.hit = False

    def update(self, dt):
        self.y += NOTE_SPEED * dt

    def draw(self, surf):
        x = self.lane * LANE_WIDTH + 10
        pygame.draw.rect(surf, NOTE_COLOR, (x, self.y, NOTE_WIDTH, NOTE_HEIGHT))

    def is_hittable(self):
        return abs(self.y + NOTE_HEIGHT - HIT_LINE_Y) < 30

    def is_missed(self):
        return self.y > HIT_LINE_Y + NOTE_HEIGHT


def main():
    pygame.init()

    # 1) File dialog for video/audio selection
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select video/audio file",
        filetypes=[("Video/Audio Files", "*.mp4 *.mp3 *.wav")]
    )
    if not file_path:
        print("No file selected, exiting.")
        return

    # 2) Extract audio from MP4 if needed
    if file_path.endswith(".mp4"):
        print("🎞 Extracting audio from MP4...")
        video = VideoFileClip(file_path)
        extracted_audio_path = "extracted_audio.wav"
        video.audio.write_audiofile(extracted_audio_path, verbose=False, logger=None)
        analysis_path = extracted_audio_path
    else:
        analysis_path = file_path

    # 3) Audio analysis
    print("Analyzing audio onsets and spectral centroids...")
    y, sr = librosa.load(analysis_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = np.arange(len(centroids))
    centroid_times = librosa.frames_to_time(frames, sr=sr)

    if len(onset_times) == 0:
        print("⚠  No onsets detected.")
        return

    travel_distance = HIT_LINE_Y + NOTE_HEIGHT
    travel_time = travel_distance / NOTE_SPEED

    min_c, max_c = centroids.min(), centroids.max()
    if max_c == min_c:
        max_c = min_c + 1

    spawn_schedule = []
    for t in onset_times:
        spawn_time = max(0.0, t - travel_time)
        idx = np.argmin(np.abs(centroid_times - t))
        c = centroids[idx]
        norm = (c - min_c) / (max_c - min_c)
        lane = int(norm * (LANE_COUNT - 1) + 0.5)
        lane = max(0, min(LANE_COUNT - 1, lane))
        spawn_schedule.append((spawn_time, lane))
    spawn_schedule.sort(key=lambda x: x[0])

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Guitar Hero ▶ " + os.path.basename(file_path))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    pygame.mixer.init()
    pygame.mixer.music.load(analysis_path)
    pygame.mixer.music.play()
    start_time = time.time()

    notes = []
    next_idx = 0
    score = 0
    running = True

    feedback_texts = [""] * LANE_COUNT
    feedback_times = [0] * LANE_COUNT

    while running:
        dt = clock.tick(FPS) / 1000.0
        current_time = time.time() - start_time

        while next_idx < len(spawn_schedule) and spawn_schedule[next_idx][0] <= current_time:
            _, lane = spawn_schedule[next_idx]
            notes.append(Note(lane))
            next_idx += 1

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False
            elif evt.type == pygame.KEYDOWN and evt.key in KEYS:
                lane = KEYS.index(evt.key)
                hit_success = False
                for note in notes:
                    if note.lane == lane and not note.hit and note.is_hittable():
                        note.hit = True
                        score += 10
                        hit_success = True
                        break
                if hit_success:
                    feedback_texts[lane] = "Hit!"
                else:
                    feedback_texts[lane] = "Miss!"
                feedback_times[lane] = current_time

        for note in notes:
            note.update(dt)

        notes = [n for n in notes if not (n.hit or n.is_missed())]

        screen.fill(BG_COLOR)

        for i in range(1, LANE_COUNT):
            pygame.draw.line(screen, LANE_COLOR, (i * LANE_WIDTH, 0), (i * LANE_WIDTH, SCREEN_HEIGHT), 2)

        pygame.draw.line(screen, HIT_LINE_COLOR, (0, HIT_LINE_Y), (SCREEN_WIDTH, HIT_LINE_Y), 4)

        for note in notes:
            note.draw(screen)

        key_labels = ['A', 'S', 'D', 'F']
        for i, label in enumerate(key_labels):
            text = font.render(label, True, TEXT_COLOR)
            x = i * LANE_WIDTH + (LANE_WIDTH - text.get_width()) // 2
            y = HIT_LINE_Y + 10
            screen.blit(text, (x, y))

        for i in range(LANE_COUNT):
            if current_time - feedback_times[i] < 0.5 and feedback_texts[i]:
                fb_surf = font.render(feedback_texts[i], True, TEXT_COLOR)
                fb_x = i * LANE_WIDTH + (LANE_WIDTH - fb_surf.get_width()) // 2
                fb_y = HIT_LINE_Y - 40
                screen.blit(fb_surf, (fb_x, fb_y))

        score_surf = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_surf, ((SCREEN_WIDTH - score_surf.get_width()) // 2, 10))

        pygame.display.flip()

        if not pygame.mixer.music.get_busy() and next_idx >= len(spawn_schedule):
            running = False

    screen.fill(BG_COLOR)
    final_score_text = font.render(f"Final Score: {score}", True, TEXT_COLOR)
    screen.blit(final_score_text, ((SCREEN_WIDTH - final_score_text.get_width()) // 2, SCREEN_HEIGHT // 2))

    pygame.display.flip()

    
    time.sleep(3)

    print(f"🏁 Final Score: {score}")
    pygame.quit()

    if os.path.exists("extracted_audio.wav"):
        os.remove("extracted_audio.wav")

    sys.exit()


if __name__ == "_main_":
    main()