import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { CameraPanel } from './components/CameraPanel';
import { LiveRecognitionPanel } from './components/LiveRecognitionPanel';
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
} from './types';
import { AlertCircle, RefreshCw, Radio, ExternalLink } from 'lucide-react';

const DEFAULT_SETTINGS: AppSettings = {
  confidenceThreshold: 0.40,
  debounceFrames: 5,
  pauseThresholdSec: 0.6,
  targetFps: 12,
  autoSpeak: false,
  ttsRate: 1.0,
  ttsPitch: 1.0,
  ttsVoice: '',
  mirrorCamera: true,
  showLandmarkGuide: true,
  wsUrl:
    typeof window !== 'undefined' && window.location.protocol === 'https:'
      ? `wss://${window.location.hostname}:8000/ws/live`
      : 'ws://127.0.0.1:8000/ws/live',
};

export const App: React.FC = () => {
  // System & Model Status
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [backendError, setBackendError] = useState<string | null>(null);

  // Recognition States
  const [recognitionMode, setRecognitionMode] = useState<RecognitionMode>('guided');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [landmarksStatus, setLandmarksStatus] = useState<LandmarksStatus | null>(null);
  const [prediction, setPrediction] = useState<LivePrediction | null>(null);
  const [candidate, setCandidate] = useState<WordCandidate | null>(null);
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);

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

  // Settings
  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      const saved = localStorage.getItem('signflow_settings');
      return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  // Modals
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [isHelpOpen, setIsHelpOpen] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Persist tokens to localStorage
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

  // Fetch backend health and model info via HTTP
  const fetchStatus = useCallback(async () => {
    try {
      const baseUrl = 'http://127.0.0.1:8000';
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
        'Unable to reach FastAPI backend service at http://127.0.0.1:8000. Please ensure the backend is running.'
      );
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Auto-speak newly committed token if enabled
  const speakWord = useCallback((word: string) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    const utterance = new SpeechSynthesisUtterance(word);
    utterance.rate = settings.ttsRate;
    utterance.pitch = settings.ttsPitch;
    if (settings.ttsVoice) {
      const v = window.speechSynthesis.getVoices().find((x) => x.name === settings.ttsVoice);
      if (v) utterance.voice = v;
    }
    window.speechSynthesis.speak(utterance);
  }, [settings]);

  // Connect to WebSocket
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
        // Send initial configuration
        ws.send(
          JSON.stringify({
            type: 'config',
            mode: recognitionMode,
            confidence_threshold: settings.confidenceThreshold,
            debounce_frames: settings.debounceFrames,
            pause_threshold_sec: settings.pauseThresholdSec,
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          switch (msg.type) {
            case 'session_started':
              if (msg.model_ready) {
                fetchStatus();
              }
              break;

            case 'landmarks_status':
              setLandmarksStatus(msg);
              break;

            case 'live_prediction':
              setPrediction(msg);
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
              if (settings.autoSpeak) {
                speakWord(msg.word);
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
        // Attempt reconnect after delay
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      ws.onerror = (err) => {
        console.warn('WebSocket connection error:', err);
        setWsConnected(false);
      };
    } catch (err) {
      console.error('WebSocket creation exception:', err);
      setWsConnected(false);
    }
  }, [settings, recognitionMode, fetchStatus, speakWord]);

  // Initial WS connection
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWebSocket]);

  // Handle frame capture from camera
  const handleFrameCaptured = useCallback(
    (imageDataUrl: string, timestampMs: number) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: 'frame',
            data: imageDataUrl,
            timestamp_ms: timestampMs,
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

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 selection:bg-teal-100 selection:text-teal-900">
      {/* Top Header */}
      <Header
        health={health}
        modelInfo={modelInfo}
        wsConnected={wsConnected}
        recognitionMode={recognitionMode}
        onModeChange={handleModeChange}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenHelp={() => setIsHelpOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Connection Error Banner */}
        {backendError && (
          <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between shadow-xs">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
              <div>
                <p className="font-bold text-sm">Backend Service Offline</p>
                <p className="mt-0.5 text-rose-700">{backendError}</p>
              </div>
            </div>
            <button
              onClick={fetchStatus}
              className="px-3 py-1.5 rounded-xl bg-white border border-rose-200 text-rose-700 font-semibold hover:bg-rose-50 transition-colors flex items-center space-x-1"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Model Missing Advisory */}
        {health && !health.trained_model_loaded && (
          <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start space-x-3 shadow-xs">
            <AlertCircle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-bold text-sm">Trained Model Checkpoint Not Detected</p>
              <p className="leading-relaxed">
                The backend is currently running without a trained model checkpoint. To enable full
                recognition, verify that <code>models/temporal_transformer_trained.pt</code> exists in
                your workspace.
              </p>
            </div>
          </div>
        )}

        {/* Live Recognition Workspace: 2-Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Live Camera Panel */}
          <div className="lg:col-span-7">
            <CameraPanel
              onFrameCaptured={handleFrameCaptured}
              isStreaming={isStreaming}
              setIsStreaming={setIsStreaming}
              landmarksStatus={landmarksStatus}
              performance={performance}
              targetFps={settings.targetFps}
              mirrorCamera={settings.mirrorCamera}
              onToggleMirror={() =>
                setSettings((s) => ({ ...s, mirrorCamera: !s.mirrorCamera }))
              }
            />
          </div>

          {/* Right Column: Live Recognition & Candidate Card */}
          <div className="lg:col-span-5">
            <LiveRecognitionPanel
              prediction={prediction}
              candidate={candidate}
              mode={recognitionMode}
              onConfirmWord={handleConfirmWord}
              onRejectCandidate={handleRejectCandidate}
              confidenceThreshold={settings.confidenceThreshold}
            />
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
      <footer className="border-t border-slate-200 bg-white py-4 px-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>
            <strong>SignFlow</strong> — Robust Real-Time Dynamic Word-Level ASL Recognition Thesis Project
          </p>
          <div className="flex items-center space-x-3 text-slate-400">
            <span>WLASL Benchmark</span>
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
