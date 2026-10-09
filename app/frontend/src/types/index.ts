export type RecognitionMode = 'guided' | 'continuous';

export interface ModelInfo {
  name: string;
  architecture: string;
  status: string;
  checkpoint_path: string | null;
  num_classes: number;
  input_dim: number;
  sequence_length: number;
  device: string;
  feature_representation: string;
  class_names_sample: string[];
  total_classes: number;
  error: string | null;
}

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'error';
  process_alive: boolean;
  backend_initialized: boolean;
  mediapipe_available: boolean;
  trained_model_loaded: boolean;
  inference_ready: boolean;
  checkpoint_path: string | null;
  error: string | null;
  mediapipe_error: string | null;
}

export interface CommittedToken {
  id: string;
  word: string;
  confidence: number;
  timestamp: number;
  mode: RecognitionMode;
  isEdited?: boolean;
}

export interface PerformanceMetrics {
  fps: number;
  extraction_ms: number;
  inference_ms: number;
  total_latency_ms: number;
  frame_count: number;
}

export interface LandmarksStatus {
  hands_detected: boolean;
  left_hand: boolean;
  right_hand: boolean;
  pose: boolean;
  face: boolean;
  valid_points: number;
}

export interface CandidatePrediction {
  class_index: number;
  label: string;
  confidence: number;
}

export interface LivePrediction {
  predicted_label: string;
  confidence: number;
  meets_threshold: boolean;
  top_k: CandidatePrediction[];
  latency_ms: number;
  state: string;
}

export interface WordCandidate {
  word: string | null;
  confidence: number;
  stability_count: number;
  target_count: number;
  progress: number;
}

export interface AppSettings {
  confidenceThreshold: number;
  debounceFrames: number;
  pauseThresholdSec: number;
  targetFps: number;
  autoSpeak: boolean;
  ttsRate: number;
  ttsPitch: number;
  ttsVoice: string;
  mirrorCamera: boolean;
  showLandmarkGuide: boolean;
  wsUrl: string;
}
