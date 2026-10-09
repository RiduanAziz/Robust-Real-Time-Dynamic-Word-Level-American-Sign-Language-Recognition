import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Camera,
  CameraOff,
  FlipHorizontal,
  Pause,
  Play,
  Eye,
  EyeOff,
  Video,
  AlertCircle,
  Activity,
  Layers,
  Sparkles
} from 'lucide-react';
import { LandmarksStatus, PerformanceMetrics } from '../types';

interface CameraPanelProps {
  onFrameCaptured: (imageDataUrl: string, timestampMs: number) => void;
  isStreaming: boolean;
  setIsStreaming: (streaming: boolean) => void;
  landmarksStatus: LandmarksStatus | null;
  performance: PerformanceMetrics | null;
  targetFps: number;
  mirrorCamera: boolean;
  onToggleMirror: () => void;
}

export const CameraPanel: React.FC<CameraPanelProps> = ({
  onFrameCaptured,
  isStreaming,
  setIsStreaming,
  landmarksStatus,
  performance,
  targetFps,
  mirrorCamera,
  onToggleMirror,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);

  // Enumerate video devices
  useEffect(() => {
    const getDevices = async () => {
      try {
        const devs = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devs.filter((d) => d.kind === 'videoinput');
        setDevices(videoDevs);
        if (videoDevs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevs[0].deviceId);
        }
      } catch (err) {
        console.warn('Could not enumerate video devices:', err);
      }
    };
    getDevices();
  }, [selectedDeviceId]);

  // Start Camera Stream
  const startCamera = async () => {
    setPermissionError(null);
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
      setIsStreaming(true);
    } catch (err: any) {
      console.error('Camera access error:', err);
      setCameraActive(false);
      setIsStreaming(false);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setPermissionError('Camera access was denied. Please allow camera permissions in your browser address bar.');
      } else {
        setPermissionError(`Failed to access camera: ${err.message || 'Unknown error'}`);
      }
    }
  };

  // Stop Camera Stream
  const stopCamera = () => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setIsStreaming(false);
  };

  // Capture & transmit frames at targetFps
  useEffect(() => {
    if (!cameraActive || !isStreaming) {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const intervalMs = Math.round(1000 / Math.max(5, Math.min(30, targetFps)));

    intervalRef.current = window.setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;

      // Downscale if necessary for efficient transmission (target 480x360 or 640x480)
      const targetW = 480;
      const targetH = Math.round((height / width) * 480);

      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, targetW, targetH);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      onFrameCaptured(dataUrl, Date.now());
    }, intervalMs);

    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [cameraActive, isStreaming, targetFps, onFrameCaptured]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col space-y-4">
      {/* Panel Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Camera className="w-5 h-5 text-teal-600" />
          <h2 className="text-base font-semibold text-slate-800">Live Camera</h2>
        </div>

        {/* Device selector */}
        {devices.length > 1 && (
          <select
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            disabled={cameraActive}
            className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-slate-700 focus:outline-none focus:ring-1 focus:ring-teal-500"
          >
            {devices.map((d, idx) => (
              <option key={d.deviceId || idx} value={d.deviceId}>
                {d.label || `Camera ${idx + 1}`}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Video Viewport */}
      <div className="relative aspect-[4/3] bg-slate-950 rounded-xl overflow-hidden flex items-center justify-center border border-slate-800 shadow-inner group">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${mirrorCamera ? '-scale-x-100' : ''}`}
        />
        <canvas ref={canvasRef} className="hidden" />

        {/* Empty / Inactive state */}
        {!cameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/90 text-slate-400 p-6 text-center space-y-3">
            <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center text-slate-300 ring-4 ring-slate-800/60">
              <Video className="w-7 h-7" />
            </div>
            <div>
              <p className="font-semibold text-slate-200 text-sm">Camera is Inactive</p>
              <p className="text-xs text-slate-400 mt-1 max-w-xs">
                Click "Start Camera" to begin real-time sign detection and landmark tracking.
              </p>
            </div>
          </div>
        )}

        {/* Landmark Guide Overlay */}
        {cameraActive && showOverlay && (
          <div className="absolute inset-0 pointer-events-none p-4 flex flex-col justify-between">
            {/* Upper Frame Guide */}
            <div className="flex justify-between items-start">
              <div className="bg-slate-900/75 backdrop-blur-xs text-slate-300 text-[11px] px-2.5 py-1 rounded-md border border-slate-700/60 flex items-center space-x-1.5">
                <span className={`w-2 h-2 rounded-full ${landmarksStatus?.hands_detected ? 'bg-emerald-400 animate-ping' : 'bg-amber-400'}`} />
                <span>
                  {landmarksStatus?.hands_detected
                    ? `Hands Active (${landmarksStatus.valid_points} pts)`
                    : 'Hands Neutral / Waiting'}
                </span>
              </div>

              {/* Live FPS & Latency */}
              {performance && (
                <div className="bg-slate-900/75 backdrop-blur-xs text-slate-300 text-[11px] px-2.5 py-1 rounded-md border border-slate-700/60 font-mono">
                  {performance.fps} FPS | {performance.total_latency_ms}ms
                </div>
              )}
            </div>

            {/* Visual Guide Box for Upper Body & Hands */}
            <div className="self-center w-3/4 h-3/4 border-2 border-dashed border-teal-400/30 rounded-2xl flex items-center justify-center pointer-events-none">
              <span className="text-[10px] text-teal-300/40 font-mono uppercase tracking-widest">
                Sign Framing Area
              </span>
            </div>

            {/* Bottom Modality Presence Chips */}
            <div className="flex items-center space-x-2">
              <span
                className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                  landmarksStatus?.left_hand
                    ? 'bg-emerald-500/80 text-white'
                    : 'bg-slate-800/80 text-slate-400'
                }`}
              >
                L-Hand: {landmarksStatus?.left_hand ? 'OK' : '--'}
              </span>
              <span
                className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                  landmarksStatus?.right_hand
                    ? 'bg-emerald-500/80 text-white'
                    : 'bg-slate-800/80 text-slate-400'
                }`}
              >
                R-Hand: {landmarksStatus?.right_hand ? 'OK' : '--'}
              </span>
              <span
                className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                  landmarksStatus?.pose
                    ? 'bg-teal-500/80 text-white'
                    : 'bg-slate-800/80 text-slate-400'
                }`}
              >
                Pose: {landmarksStatus?.pose ? 'OK' : '--'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Permission Error Banner */}
      {permissionError && (
        <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-start space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Camera Access Error</p>
            <p className="mt-0.5">{permissionError}</p>
          </div>
        </div>
      )}

      {/* Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
        <div className="flex items-center space-x-2">
          {!cameraActive ? (
            <button
              onClick={startCamera}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold shadow-xs transition-colors focus:ring-2 focus:ring-teal-500 focus:outline-none"
            >
              <Camera className="w-4 h-4" />
              <span>Start Camera</span>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold shadow-xs transition-colors focus:ring-2 focus:ring-rose-500 focus:outline-none"
            >
              <CameraOff className="w-4 h-4" />
              <span>Stop Camera</span>
            </button>
          )}

          {cameraActive && (
            <button
              onClick={() => setIsStreaming(!isStreaming)}
              className={`inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl border text-xs font-semibold transition-colors ${
                isStreaming
                  ? 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100'
                  : 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100'
              }`}
            >
              {isStreaming ? (
                <>
                  <Pause className="w-3.5 h-3.5" />
                  <span>Pause Stream</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span>Resume Stream</span>
                </>
              )}
            </button>
          )}
        </div>

        <div className="flex items-center space-x-1">
          <button
            onClick={onToggleMirror}
            className={`p-2 rounded-xl border transition-colors ${
              mirrorCamera
                ? 'bg-teal-50 border-teal-200 text-teal-700'
                : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
            }`}
            title="Mirror Camera Preview"
          >
            <FlipHorizontal className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`p-2 rounded-xl border transition-colors ${
              showOverlay
                ? 'bg-teal-50 border-teal-200 text-teal-700'
                : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
            }`}
            title="Toggle Landmark Overlay Guides"
          >
            {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  );
};
