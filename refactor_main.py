import re

with open("src/sign_language/api/main.py", "r") as f:
    code = f.read()

# Define the helper class
helper_class = """
class SessionState:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.mode = "guided"
        self.confidence_threshold = 0.28
        self.debounce_frames = 2
        self.pause_threshold_sec = 0.5
        self.min_sequence_frames = 10

        self.sequence_buffer: collections.deque = collections.deque(maxlen=self.seq_len)
        self.mask_buffer: collections.deque = collections.deque(maxlen=self.seq_len)
        self.candidate_history: collections.deque = collections.deque(maxlen=self.debounce_frames)
        self.recent_replay_frames: collections.deque = collections.deque(maxlen=64)
        self.wrist_history: collections.deque = collections.deque(maxlen=5)
        self.speed_history: collections.deque = collections.deque(maxlen=10)

        self.last_brightness: float | None = None
        self.last_blur_score: float | None = None

        self.current_candidate: str | None = None
        self.current_confidence: float = 0.0
        self.last_committed_word: str | None = None
        self.state: str = "WAITING"  # "WAITING" | "CAPTURING" | "CANDIDATE" | "COMMITTED"
        
        import time
        self.last_hand_activity_time = time.time()
        self.frame_counter = 0
        self.start_time = time.time()

    def update_config(self, payload: dict):
        if "mode" in payload and payload["mode"] in {"guided", "continuous"}:
            self.mode = payload["mode"]
        if "confidence_threshold" in payload:
            self.confidence_threshold = float(payload["confidence_threshold"])
        if "debounce_frames" in payload:
            self.debounce_frames = max(2, int(payload["debounce_frames"]))
            self.candidate_history = collections.deque(
                list(self.candidate_history), maxlen=self.debounce_frames
            )
        if "pause_threshold_sec" in payload:
            self.pause_threshold_sec = float(payload["pause_threshold_sec"])

    def full_reset(self):
        self.sequence_buffer.clear()
        self.mask_buffer.clear()
        self.candidate_history.clear()
        self.wrist_history.clear()
        self.speed_history.clear()
        self.current_candidate = None
        self.current_confidence = 0.0
        self.last_committed_word = None
        self.state = "WAITING"

    def transition_to_committed(self, word: str):
        self.last_committed_word = word
        self.state = "COMMITTED"
        self.current_candidate = None
        self.sequence_buffer.clear()
        self.mask_buffer.clear()
        self.candidate_history.clear()
        self.speed_history.clear()
        self.wrist_history.clear()

    def reject_candidate(self):
        self.current_candidate = None
        self.current_confidence = 0.0
        self.candidate_history.clear()
        self.state = "WAITING"

    def handle_neutral_pause(self):
        if self.state in {"COMMITTED", "CAPTURING", "CANDIDATE"}:
            self.state = "WAITING"
            self.sequence_buffer.clear()
            self.mask_buffer.clear()
            self.wrist_history.clear()
            self.speed_history.clear()
            # Note: last_committed_word is not cleared so it acts as duplicate prevention
            
    def clear_leakage_on_new_sign(self):
        if self.state == "WAITING":
            self.sequence_buffer.clear()
            self.mask_buffer.clear()
            self.wrist_history.clear()
            self.speed_history.clear()
            self.candidate_history.clear()

"""

# Insert helper class right before websocket_live_recognition
code = code.replace('@app.websocket("/ws/live")', helper_class + '\n@app.websocket("/ws/live")')

with open("scratch_main.py", "w") as f:
    f.write(code)
