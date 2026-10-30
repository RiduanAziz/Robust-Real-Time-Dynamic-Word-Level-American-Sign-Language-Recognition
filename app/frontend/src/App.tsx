import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { CameraPanel } from './components/CameraPanel';
import { LiveRecognitionPanel } from './components/LiveRecognitionPanel';
import { HandQualityCoach } from './components/HandQualityCoach';
import { PostureInspector } from './components/PostureInspector';
import { GestureReplay } from './components/GestureReplay';
import { RobustnessLab } from './components/RobustnessLab';
import { PracticeMode } from './components/PracticeMode';
import { TranscriptWorkspace } from './components/TranscriptWorkspace';
import { SpeechWorkspace } from './components/SpeechWorkspace';
import { SettingsModal } from './components/SettingsModal';
import { HelpModal } from './components/HelpModal';
import {
  HealthStatus,
  ModelInfo,
  RecognitionMode,
  CommittedToken,
  LivePrediction,
  WordCandidate,
  LandmarksStatus,
  PerformanceMetrics,
  AppSettings,
  LandmarkPoint,
  BoundingBox,
  QualityCoachMetrics,
  RecognitionStateCode,
  ReplayFrame,
  AvailableModelItem,
} from './types';
import {
  AlertCircle,
  RefreshCw,
  Radio,
  ExternalLink,
  ShieldCheck,
  Hand,
  Clock,
  FlaskConical,
  GraduationCap,
} from 'lucide-react';
import {
  getBaseApiUrl,
  getDefaultWebSocketUrl,
  resolveEffectiveSettings,
} from './utils/url';

const DEFAULT_SETTINGS: AppSettings = {
  confidenceThreshold: 0.28,
  debounceFrames: 2,
  pauseThresholdSec: 0.5,
  targetFps: 25,
  autoSpeak: false,
  ttsRate: 1.0,
  ttsPitch: 1.0,
  ttsVoice: '',
  mirrorCamera: true,
  showLandmarkGuide: true,
  wsUrl: getDefaultWebSocketUrl(),
  isCustomWsUrl: false,
};

type AdvancedTab = 'coach_posture' | 'replay' | 'robustness' | 'practice';

export const App: React.FC = () => {
  // System & Model Status
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [backendError, setBackendError] = useState<string | null>(null);

  // Candidate Models & Active Model State
  const [availableModels, setAvailableModels] = useState<AvailableModelItem[]>([
    {
      id: 'ensemble_3way',
      name: '3-Way Probability Ensemble (Recommended)',
      accuracy: '76.65% Top-1 / 91.44% Top-5',
      top1: 76.65,
      top5: 91.44,
      ece: 0.0838,
      description: 'Validation-tuned probability fusion across all 3 architectures (w=[0.25, 0.10, 0.65])',
      badge: 'SOTA Benchmark',
      active: true,
      available: true,
    },
    {
      id: 'robust_holistic_fusion',
      name: 'Baseline Holistic BiGRU',
      accuracy: '74.32% Top-1 / 88.72% Top-5',
      top1: 74.32,
      top5: 88.72,
      ece: 0.1598,
      description: 'Recurrent temporal modeling on 553 holistic coordinates',
      badge: 'Historical Baseline',
      active: false,
      available: true,
    },
    {
      id: 'wlasl100_max_accuracy',
      name: 'Deep 2-Layer BiGRU',
      accuracy: '72.76% Top-1 / 89.88% Top-5',
      top1: 72.76,
      top5: 89.88,
      ece: 0.0912,
      description: 'Deep 2-layer BiGRU with label smoothing (hidden 128, emb 96)',
      badge: 'Candidate 2',
      active: false,
      available: true,
    },
    {
      id: 'unisign_pgf',
      name: 'Uni-Sign Prior-Guided Attention',
      accuracy: '70.43% Top-1 / 88.33% Top-5',
      top1: 70.43,
      top5: 88.33,
      ece: 0.0561,
      description: 'ICLR 2025 hand-to-pose cross-attention & score-aware visibility gating',
      badge: 'High Calibration',
      active: false,
      available: true,
    },
  ]);
  const [selectedModelId, setSelectedModelId] = useState<string>('ensemble_3way');
  const [isSwitchingModel, setIsSwitchingModel] = useState<boolean>(false);

  // Video Devices State
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  // Recognition States
  const [recognitionMode, setRecognitionMode] = useState<RecognitionMode>('guided');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [landmarksStatus, setLandmarksStatus] = useState<LandmarksStatus | null>(null);
  const [prediction, setPrediction] = useState<LivePrediction | null>(null);
  const [candidate, setCandidate] = useState<WordCandidate | null>(null);
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);
  const [lastFrameAck, setLastFrameAck] = useState<number>(0);
  const [lastProcessedFrameId, setLastProcessedFrameId] = useState<number | null>(null);

  // Advanced Visuals & Quality States
  const [leftLandmarks, setLeftLandmarks] = useState<LandmarkPoint[]>([]);
  const [rightLandmarks, setRightLandmarks] = useState<LandmarkPoint[]>([]);
  const [leftBox, setLeftBox] = useState<BoundingBox | null>(null);
  const [rightBox, setRightBox] = useState<BoundingBox | null>(null);
  const [quality, setQuality] = useState<QualityCoachMetrics | null>(null);
  const [stateCode, setStateCode] = useState<RecognitionStateCode>('STATE_A_INACTIVE');
  const [replayFrames, setReplayFrames] = useState<ReplayFrame[]>([]);
  const [activeTab, setActiveTab] = useState<AdvancedTab>('coach_posture');

  // Transcript Tokens
  const [tokens, setTokens] = useState<CommittedToken[]>(() => {
    try {
      const saved = localStorage.getItem('signflow_tokens');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [transcriptText, setTranscriptText] = useState<string>('');

  // Settings with auto-migration of stale endpoints
  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      const saved = localStorage.getItem('signflow_settings');
      return resolveEffectiveSettings(saved, DEFAULT_SETTINGS);
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  // Modals
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [isHelpOpen, setIsHelpOpen] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const isMountedRef = useRef<boolean>(true);
  const settingsRef = useRef<AppSettings>(settings);
  settingsRef.current = settings;

  // Persist tokens
  useEffect(() => {
    try {
      localStorage.setItem('signflow_tokens', JSON.stringify(tokens));
    } catch (e) {
      console.warn('Failed to save tokens to localStorage:', e);
    }
  }, [tokens]);

  // Persist settings
  useEffect(() => {
    try {
      localStorage.setItem('signflow_settings', JSON.stringify(settings));
    } catch (e) {
      console.warn('Failed to save settings:', e);
    }
  }, [settings]);

  // Send dynamic configuration update over open WebSocket without reconnecting
  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          mode: recognitionMode,
          confidence_threshold: settings.confidenceThreshold,
          debounce_frames: settings.debounceFrames,
          pause_threshold_sec: settings.pauseThresholdSec,
        })
      );
    }
  }, [settings.confidenceThreshold, settings.debounceFrames, settings.pauseThresholdSec, recognitionMode]);

  // Fetch backend health and model info (environment aware)
  const fetchStatus = useCallback(async () => {
    try {
      const baseUrl = getBaseApiUrl();
      const healthRes = await fetch(`${baseUrl}/health`);
      if (healthRes.ok) {
        const hData = await healthRes.json();
        setHealth(hData);
        setBackendError(null);
      } else {
        setHealth(null);
        setBackendError('Backend health check returned non-200');
      }

      const modelRes = await fetch(`${baseUrl}/model/info`);
      if (modelRes.ok) {
        const mData = await modelRes.json();
        setModelInfo(mData);
      }
    } catch (err: any) {
      setHealth(null);
      setBackendError(
        'Unable to reach FastAPI backend service. Please ensure the backend is running.'
      );
    }
  }, []);

  // Fetch available models from backend
  const fetchAvailableModels = useCallback(async () => {
    try {
      const baseUrl = getBaseApiUrl();
      const res = await fetch(`${baseUrl}/models/available`);
      if (res.ok) {
        const data = await res.json();
        setAvailableModels(data);
        const active = data.find((m: AvailableModelItem) => m.active);
        if (active) {
          setSelectedModelId(active.id);
        }
      }
    } catch (e) {
      console.warn('Could not fetch available models:', e);
    }
  }, []);

  // Handle live model architecture switching
  const handleSelectModel = async (modelId: string) => {
    if (modelId === selectedModelId || isSwitchingModel) return;
    setIsSwitchingModel(true);
    try {
      const baseUrl = getBaseApiUrl();
      const res = await fetch(`${baseUrl}/model/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      });
      if (res.ok) {
        setSelectedModelId(modelId);
        setAvailableModels((prev) =>
          prev.map((m) => ({ ...m, active: m.id === modelId }))
        );
        setPrediction(null);
        setCandidate(null);
        await fetchStatus();
      } else {
        const errData = await res.json();
        console.error('Failed to switch model:', errData);
      }
    } catch (err) {
      console.error('Exception switching model:', err);
    } finally {
      setIsSwitchingModel(false);
    }
  };

  // Enumerate video camera devices
  const enumerateVideoDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const devs = await navigator.mediaDevices.enumerateDevices();
      const videoDevs = devs.filter((d) => d.kind === 'videoinput');
      setVideoDevices(videoDevs);
      if (videoDevs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevs[0].deviceId);
      }
    } catch (err) {
      console.warn('Unable to enumerate video devices:', err);
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    fetchStatus();
    fetchAvailableModels();
    enumerateVideoDevices();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, [fetchStatus, fetchAvailableModels, enumerateVideoDevices]);

  // Auto-speak newly committed token if enabled
  const speakWord = useCallback(
    (word: string) => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
      const utterance = new SpeechSynthesisUtterance(word);
      utterance.rate = settingsRef.current.ttsRate;
      utterance.pitch = settingsRef.current.ttsPitch;
      if (settingsRef.current.ttsVoice) {
        const v = window.speechSynthesis.getVoices().find((x) => x.name === settingsRef.current.ttsVoice);
        if (v) utterance.voice = v;
      }
      window.speechSynthesis.speak(utterance);
    },
    []
  );

  // Connect to WebSocket (only dependent on wsUrl)
  const connectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const ws = new WebSocket(settings.wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        setBackendError(null);
        ws.send(
          JSON.stringify({
            type: 'config',
            mode: recognitionMode,
            confidence_threshold: settingsRef.current.confidenceThreshold,
            debounce_frames: settingsRef.current.debounceFrames,
            pause_threshold_sec: settingsRef.current.pauseThresholdSec,
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          setLastFrameAck(Date.now());
          switch (msg.type) {
            case 'frame_processed':
              if (typeof msg.frame_id === 'number') {
                setLastProcessedFrameId(msg.frame_id);
              }
              setLastFrameAck(Date.now());
              break;

            case 'session_started':
              if (msg.model_ready) {
                fetchStatus();
              }
              break;

            case 'landmarks_status':
              setLandmarksStatus(msg);
              break;

            case 'hand_landmarks':
              if (msg.left_landmarks) setLeftLandmarks(msg.left_landmarks);
              if (msg.right_landmarks) setRightLandmarks(msg.right_landmarks);
              setLeftBox(msg.left_box || null);
              setRightBox(msg.right_box || null);
              if (msg.quality) setQuality(msg.quality);
              if (msg.state_code) setStateCode(msg.state_code);
              setLandmarksStatus(msg);
              break;

            case 'live_prediction':
              setPrediction(msg);
              if (msg.state_code) setStateCode(msg.state_code);
              break;

            case 'word_candidate':
              setCandidate(msg);
              break;

            case 'word_committed':
              const newToken: CommittedToken = {
                id: msg.token_id || String(Date.now()),
                word: msg.word,
                confidence: msg.confidence || 0,
                timestamp: msg.timestamp || Date.now(),
                mode: msg.mode || 'guided',
              };
              setTokens((prev) => [...prev, newToken]);
              setStateCode('STATE_E_COMMITTED');
              if (settings.autoSpeak) {
                speakWord(msg.word);
              }
              // Automatically fetch replay buffer for recently completed gesture
              ws.send(JSON.stringify({ type: 'request_replay' }));
              break;

            case 'replay_buffer':
              if (msg.frames) {
                setReplayFrames(msg.frames);
              }
              break;

            case 'performance':
              setPerformance(msg);
              break;

            case 'error':
              console.warn('Backend WebSocket error:', msg.message);
              break;

            default:
              break;
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (isMountedRef.current) {
          reconnectTimeoutRef.current = window.setTimeout(() => {
            if (isMountedRef.current) {
              connectWebSocket();
            }
          }, 3000);
        }
      };

      ws.onerror = (err) => {
        console.warn('WebSocket connection error:', err);
        setWsConnected(false);
      };
    } catch (err) {
      console.error('WebSocket creation exception:', err);
      setWsConnected(false);
    }
  }, [settings.wsUrl, recognitionMode, fetchStatus, speakWord]);

  // Initial WS connection
  useEffect(() => {
    isMountedRef.current = true;
    connectWebSocket();
    return () => {
      isMountedRef.current = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWebSocket]);

  // Frame transmission
  const handleFrameCaptured = useCallback(
    (imageDataUrl: string, timestampMs: number, frameId?: number) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: 'frame',
            data: imageDataUrl,
            timestamp_ms: timestampMs,
            frame_id: frameId,
          })
        );
      }
    },
    []
  );

  // Manual actions
  const handleConfirmWord = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'confirm_word' }));
    }
  };

  const handleRejectCandidate = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reject_candidate' }));
    }
  };

  const handleModeChange = (newMode: RecognitionMode) => {
    setRecognitionMode(newMode);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'config', mode: newMode }));
    }
  };

  const handleRemoveToken = (id: string) => {
    setTokens((prev) => prev.filter((t) => t.id !== id));
  };

  const handleClearTranscript = () => {
    setTokens([]);
    setTranscriptText('');
  };

  const handleUndo = () => {
    setTokens((prev) => prev.slice(0, -1));
  };

  const handleNewSession = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset' }));
    }
    setTokens([]);
    setTranscriptText('');
    setPrediction(null);
    setCandidate(null);
  };

  const requestReplayBuffer = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'request_replay' }));
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-teal-500/30 selection:text-teal-200 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-slate-950">
      {/* Top Navigation Header with Model & Camera Dropdowns */}
      <Header
        health={health}
        modelInfo={modelInfo}
        wsConnected={wsConnected}
        recognitionMode={recognitionMode}
        onModeChange={handleModeChange}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenHelp={() => setIsHelpOpen(true)}
        availableModels={availableModels}
        selectedModelId={selectedModelId}
        onSelectModel={handleSelectModel}
        isSwitchingModel={isSwitchingModel}
        videoDevices={videoDevices}
        selectedDeviceId={selectedDeviceId}
        onSelectDevice={setSelectedDeviceId}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Connection Error Banner */}
        {backendError && (
          <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center justify-between shadow-xl backdrop-blur-md">
            <div className="flex items-center space-x-2.5">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
              <div>
                <p className="font-bold text-sm text-white">Backend Service Offline</p>
                <p className="mt-0.5 text-rose-300/90">{backendError}</p>
              </div>
            </div>
            <button
              onClick={fetchStatus}
              className="px-3.5 py-1.5 rounded-xl bg-slate-900 border border-rose-500/40 text-rose-300 font-semibold hover:bg-rose-500/10 transition-colors flex items-center space-x-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Checkpoint Missing Advisory */}
        {health && !health.trained_model_loaded && (
          <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start space-x-3 shadow-xl backdrop-blur-md">
            <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-bold text-sm text-white">Trained Model Checkpoint Not Detected</p>
              <p className="leading-relaxed text-amber-200/80">
                The backend is currently running without a trained model checkpoint. Expected{' '}
                <code className="text-amber-300 bg-amber-950/60 px-1 py-0.5 rounded">models/robust_holistic_fusion_best.pt</code>. In accordance with thesis
                integrity requirements, predictions are paused to avoid invented words or random fallbacks.
              </p>
            </div>
          </div>
        )}

        {/* Primary Workspace: Live Camera (Left) + Live Recognition (Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Panel: Camera, 21-pt Skeleton, Hand Focus Inset, Camera Switcher */}
          <div className="lg:col-span-7">
            <CameraPanel
              onFrameCaptured={handleFrameCaptured}
              isStreaming={isStreaming}
              setIsStreaming={setIsStreaming}
              lastFrameAck={lastFrameAck}
              lastProcessedFrameId={lastProcessedFrameId}
              landmarksStatus={landmarksStatus}
              leftLandmarks={leftLandmarks}
              rightLandmarks={rightLandmarks}
              leftBox={leftBox}
              rightBox={rightBox}
              quality={quality}
              stateCode={stateCode}
              candidateWord={candidate?.word || prediction?.predicted_label || null}
              performance={performance}
              targetFps={settings.targetFps}
              mirrorCamera={settings.mirrorCamera}
              onToggleMirror={() =>
                setSettings((s) => ({ ...s, mirrorCamera: !s.mirrorCamera }))
              }
              videoDevices={videoDevices}
              selectedDeviceId={selectedDeviceId}
              onSelectDevice={setSelectedDeviceId}
              onDevicesEnumerated={setVideoDevices}
              onStateCodeChange={(code) => setStateCode(code)}
            />
          </div>

          {/* Right Panel: Recognition Card, Candidate Distribution, Confirm Actions */}
          <div className="lg:col-span-5">
            <LiveRecognitionPanel
              prediction={prediction}
              candidate={candidate}
              mode={recognitionMode}
              stateCode={stateCode}
              onConfirmWord={handleConfirmWord}
              onRejectCandidate={handleRejectCandidate}
              confidenceThreshold={settings.confidenceThreshold}
              lastCommittedWord={tokens.length > 0 ? tokens[tokens.length - 1].word : null}
            />
          </div>
        </div>

        {/* Advanced Features Tab Strip */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2 border-b border-slate-800 pb-2 overflow-x-auto">
            <button
              onClick={() => setActiveTab('coach_posture')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'coach_posture'
                  ? 'bg-teal-500 text-slate-950 font-bold shadow-lg shadow-teal-500/20'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800'
              }`}
            >
              <ShieldCheck className="w-4 h-4" />
              <span>Hand Quality Coach & Posture</span>
            </button>

            <button
              onClick={() => {
                setActiveTab('replay');
                requestReplayBuffer();
              }}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'replay'
                  ? 'bg-teal-500 text-slate-950 font-bold shadow-lg shadow-teal-500/20'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800'
              }`}
            >
              <Clock className="w-4 h-4" />
              <span>Gesture Replay Timeline</span>
            </button>

            <button
              onClick={() => setActiveTab('robustness')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'robustness'
                  ? 'bg-indigo-600 text-white font-bold shadow-lg shadow-indigo-500/20'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800'
              }`}
            >
              <FlaskConical className="w-4 h-4" />
              <span>Robustness Lab (Thesis)</span>
            </button>

            <button
              onClick={() => setActiveTab('practice')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'practice'
                  ? 'bg-teal-500 text-slate-950 font-bold shadow-lg shadow-teal-500/20'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800'
              }`}
            >
              <GraduationCap className="w-4 h-4" />
              <span>Guided Practice Mode</span>
            </button>
          </div>

          {/* Active Tab Content Area */}
          <div>
            {activeTab === 'coach_posture' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <HandQualityCoach
                  quality={quality}
                  handsDetected={Boolean(landmarksStatus?.hands_detected)}
                  stateCode={stateCode}
                />
                <PostureInspector
                  leftLandmarks={leftLandmarks}
                  rightLandmarks={rightLandmarks}
                  currentWord={candidate?.word || prediction?.predicted_label || null}
                />
              </div>
            )}

            {activeTab === 'replay' && (
              <GestureReplay
                frames={replayFrames}
                onClose={() => setActiveTab('coach_posture')}
                candidateWord={candidate?.word || prediction?.predicted_label || null}
                confidence={candidate?.confidence ?? prediction?.confidence ?? 0}
              />
            )}

            {activeTab === 'robustness' && (
              <RobustnessLab
                validPoints={landmarksStatus?.valid_points ?? 0}
                totalPoints={553}
                stability={quality?.stability ?? 1.0}
                bufferCompleteness={quality?.buffer_completeness ?? 0.0}
                confidence={prediction?.confidence ?? 0}
                latencyMs={performance?.total_latency_ms ?? 0}
                recentFeatures={
                  replayFrames
                    .filter((f) => f.features && f.features.length > 0)
                    .map((f) => f.features as number[])
                }
              />
            )}

            {activeTab === 'practice' && (
              <PracticeMode
                currentPrediction={prediction?.predicted_label || null}
                confidence={prediction?.confidence ?? 0}
                qualityScore={quality?.score ?? 0}
              />
            )}
          </div>
        </div>

        {/* Full-Width Transcript Workspace */}
        <TranscriptWorkspace
          tokens={tokens}
          onRemoveToken={handleRemoveToken}
          onClearTranscript={handleClearTranscript}
          onUndo={handleUndo}
          transcriptText={transcriptText}
          setTranscriptText={setTranscriptText}
          onNewSession={handleNewSession}
        />

        {/* Full-Width Text-to-Speech (TTS) Workspace */}
        <SpeechWorkspace
          textToSpeak={transcriptText}
          autoSpeak={settings.autoSpeak}
          rate={settings.ttsRate}
          pitch={settings.ttsPitch}
          voiceName={settings.ttsVoice}
          onRateChange={(rate) => setSettings((s) => ({ ...s, ttsRate: rate }))}
          onPitchChange={(pitch) => setSettings((s) => ({ ...s, ttsPitch: pitch }))}
          onVoiceChange={(voice) => setSettings((s) => ({ ...s, ttsVoice: voice }))}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/80 backdrop-blur-md py-4 px-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>
            <strong className="text-white">SignFlow</strong> — Robust Real-Time Dynamic Word-Level ASL Recognition Thesis Project
          </p>
          <div className="flex items-center space-x-3 text-slate-500 font-mono">
            <span>WLASL Benchmark (100 Classes)</span>
            <span>•</span>
            <span className="text-teal-400">76.65% Top-1 / 91.44% Top-5 SOTA</span>
            <span>•</span>
            <span>Local Inference Only</span>
          </div>
        </div>
      </footer>

      {/* Modals */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onUpdateSettings={(newSettings) => setSettings((s) => ({ ...s, ...newSettings }))}
      />

      <HelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </div>
  );
};

export default App;
