import React, { useEffect, useRef, useState } from 'react';
import {
  Camera,
  CameraOff,
  SwitchCamera,
  FlipHorizontal,
  Eye,
  EyeOff,
  Video,
  AlertCircle,
  Activity,
  Layers,
  Sparkles,
  Maximize2,
  ZoomIn,
  Split,
  ShieldAlert,
} from 'lucide-react';
import {
  LandmarksStatus,
  PerformanceMetrics,
  LandmarkPoint,
  BoundingBox,
  RecognitionStateCode,
  QualityCoachMetrics,
  CameraViewMode,
} from '../types';

interface CameraPanelProps {
  onFrameCaptured: (imageDataUrl: string, timestampMs: number, frameId?: number) => void;
  isStreaming: boolean;
  setIsStreaming: (streaming: boolean) => void;
  lastFrameAck?: number;
  lastProcessedFrameId?: number | null;
  landmarksStatus: LandmarksStatus | null;
  leftLandmarks: LandmarkPoint[];
  rightLandmarks: LandmarkPoint[];
  leftBox: BoundingBox | null;
  rightBox: BoundingBox | null;
  quality: QualityCoachMetrics | null;
  stateCode: RecognitionStateCode;
  candidateWord: string | null;
  performance: PerformanceMetrics | null;
  targetFps: number;
  mirrorCamera: boolean;
  onToggleMirror: () => void;
  videoDevices?: MediaDeviceInfo[];
  selectedDeviceId?: string;
  onSelectDevice?: (deviceId: string) => void;
  onDevicesEnumerated?: (devices: MediaDeviceInfo[]) => void;
  onStateCodeChange?: (code: RecognitionStateCode) => void;
}

const HAND_CONNECTIONS = [
  // Thumb
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  // Index
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  // Middle
  [9, 10],
  [10, 11],
  [11, 12],
  // Ring
  [13, 14],
  [14, 15],
  [15, 16],
  // Pinky
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  // Palm knuckles
  [5, 9],
  [9, 13],
  [13, 17],
];

export const CameraPanel: React.FC<CameraPanelProps> = ({
  onFrameCaptured,
  isStreaming,
  setIsStreaming,
  lastFrameAck,
  lastProcessedFrameId,
  landmarksStatus,
  leftLandmarks,
  rightLandmarks,
  leftBox,
  rightBox,
  quality,
  stateCode,
  candidateWord,
  performance,
  targetFps,
  mirrorCamera,
  onToggleMirror,
  videoDevices,
  selectedDeviceId: controlledDeviceId,
  onSelectDevice,
  onDevicesEnumerated,
  onStateCodeChange,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const focusCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);

  const [localDevices, setLocalDevices] = useState<MediaDeviceInfo[]>([]);
  const [localSelectedId, setLocalSelectedId] = useState<string>('');
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<CameraViewMode>('full');

  const devices = videoDevices && videoDevices.length > 0 ? videoDevices : localDevices;
  const currentDeviceId = controlledDeviceId !== undefined ? controlledDeviceId : localSelectedId;

  const handleDeviceChange = (deviceId: string) => {
    setLocalSelectedId(deviceId);
    if (onSelectDevice) {
      onSelectDevice(deviceId);
    }
  };

  // Smoothed bounding box for Hand Focus inset
  const smoothBoxRef = useRef<BoundingBox | null>(null);

  // Enumerate video devices
  useEffect(() => {
    const getDevices = async () => {
      try {
        if (!navigator.mediaDevices?.enumerateDevices) return;
        const devs = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devs.filter((d) => d.kind === 'videoinput');
        setLocalDevices(videoDevs);
        if (onDevicesEnumerated) {
          onDevicesEnumerated(videoDevs);
        }
        if (videoDevs.length > 0 && !currentDeviceId) {
          const frontCam = videoDevs.find((d) =>
            /front|user|selfie/i.test(d.label)
          );
          if (frontCam) {
            handleDeviceChange(frontCam.deviceId);
          }
        }
      } catch (err) {
        console.warn('Could not enumerate video devices:', err);
      }
    };
    getDevices();
  }, [currentDeviceId]);

  // Flip / Cycle Camera
  const handleFlipCamera = () => {
    if (devices.length < 2) return;
    const currentIdx = devices.findIndex((d) => d.deviceId === currentDeviceId);
    const nextIdx = (currentIdx + 1) % devices.length;
    const nextDevice = devices[nextIdx];
    if (nextDevice) {
      handleDeviceChange(nextDevice.deviceId);
    }
  };

  // Start Camera Stream
  const startCamera = async (targetDeviceId?: string) => {
    setPermissionError(null);
    const devId = targetDeviceId !== undefined ? targetDeviceId : currentDeviceId;
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: devId
          ? { deviceId: { exact: devId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
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
      onStateCodeChange?.('STATE_B_NO_HAND');

      // Re-enumerate devices after permission granted so device labels appear
      try {
        const devs = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devs.filter((d) => d.kind === 'videoinput');
        setLocalDevices(videoDevs);
        if (onDevicesEnumerated) {
          onDevicesEnumerated(videoDevs);
        }
      } catch (e) {
        // ignore
      }
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

  // Auto switch camera if active and selected device changes
  const prevDeviceIdRef = useRef<string>(currentDeviceId);
  useEffect(() => {
    if (cameraActive && currentDeviceId && currentDeviceId !== prevDeviceIdRef.current) {
      prevDeviceIdRef.current = currentDeviceId;
      startCamera(currentDeviceId);
    } else {
      prevDeviceIdRef.current = currentDeviceId;
    }
  }, [currentDeviceId, cameraActive]);

  // Stop Camera Stream
  const stopCamera = () => {
    if (intervalRef.current) {
      cancelAnimationFrame(intervalRef.current);
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

  // Monotonic frame ID generator & in-flight tracking
  const frameIdCounterRef = useRef<number>(0);
  const inFlightFrameIdRef = useRef<number>(0);
  const inFlightRef = useRef<boolean>(false);

  // Release in-flight lock strictly when server acknowledges that frame or a newer frame has been processed
  useEffect(() => {
    if (lastProcessedFrameId != null && lastProcessedFrameId >= inFlightFrameIdRef.current) {
      inFlightRef.current = false;
    }
  }, [lastProcessedFrameId]);

  // Frame Capture Loop (Adaptive requestAnimationFrame pipeline with explicit frame acknowledgement)
  useEffect(() => {
    if (!cameraActive || !isStreaming) {
      if (intervalRef.current) {
        cancelAnimationFrame(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const minFrameIntervalMs = Math.round(1000 / Math.max(10, Math.min(30, targetFps)));
    let lastSendTime = 0;

    const streamLoop = () => {
      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      const now = window.performance.now();

      if (video && canvas && video.readyState >= 2) {
        const elapsed = now - lastSendTime;
        // Transmit next frame only when the previous frame has been acknowledged by server
        if (!inFlightRef.current && elapsed >= minFrameIntervalMs) {
          const width = video.videoWidth || 640;
          const height = video.videoHeight || 480;

          // Optimal resolution for Holistic Landmark Tracking (640px max dimension)
          const maxDim = 640;
          let targetW = width;
          let targetH = height;
          if (width >= height) {
            targetW = Math.min(width, maxDim);
            targetH = Math.round((height / width) * targetW);
          } else {
            targetH = Math.min(height, maxDim);
            targetW = Math.round((width / height) * targetH);
          }

          if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
          }

          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(video, 0, 0, targetW, targetH);
            frameIdCounterRef.current += 1;
            const currentFrameId = frameIdCounterRef.current;
            inFlightFrameIdRef.current = currentFrameId;
            inFlightRef.current = true;
            lastSendTime = now;

            const dataUrl = canvas.toDataURL('image/jpeg', 0.60);
            onFrameCaptured(dataUrl, Date.now(), currentFrameId);

            // Watchdog (600ms): Only unblocks if server stalls or WebSocket connection drops
            window.setTimeout(() => {
              if (inFlightFrameIdRef.current === currentFrameId && inFlightRef.current) {
                inFlightRef.current = false;
              }
            }, 600);
          }
        }
      }

      intervalRef.current = requestAnimationFrame(streamLoop);
    };

    intervalRef.current = requestAnimationFrame(streamLoop);

    return () => {
      if (intervalRef.current) {
        cancelAnimationFrame(intervalRef.current);
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

  // Draw 21-point Hand Landmarks Skeleton Overlay
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !cameraActive || !showOverlay) return;

    if (video.videoWidth && video.videoHeight) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const mapX = (x: number) => (mirrorCamera ? (1.0 - x) * w : x * w);
    const mapY = (y: number) => y * h;

    // Helper to draw single hand skeleton
    const renderHandSkeleton = (
      points: LandmarkPoint[],
      box: BoundingBox | null,
      label: string,
      color: string,
      glow: string
    ) => {
      if (!points || points.length < 21) return;
      const hasValid = points.some((p) => p.valid);
      if (!hasValid) return;

      // Draw bones
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.shadowColor = glow;
      ctx.shadowBlur = 6;

      for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const p1 = points[startIdx];
        const p2 = points[endIdx];
        if (p1?.valid && p2?.valid) {
          ctx.beginPath();
          ctx.moveTo(mapX(p1.x), mapY(p1.y));
          ctx.lineTo(mapX(p2.x), mapY(p2.y));
          ctx.stroke();
        }
      }

      // Draw 21 joint nodes
      ctx.shadowBlur = 0;
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (!p.valid) {
          // Visually distinguish missing / unreliable joints
          ctx.strokeStyle = '#f43f5e';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(mapX(p.x), mapY(p.y), 3, 0, 2 * Math.PI);
          ctx.stroke();
          continue;
        }

        const cx = mapX(p.x);
        const cy = mapY(p.y);

        // Fingertip highlights (indices 4, 8, 12, 16, 20)
        const isTip = [4, 8, 12, 16, 20].includes(i);
        const isWrist = i === 0;

        ctx.fillStyle = isTip ? '#ffffff' : color;
        ctx.beginPath();
        ctx.arc(cx, cy, isWrist ? 5.5 : isTip ? 4.5 : 3.5, 0, 2 * Math.PI);
        ctx.fill();

        if (isTip) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }

      // Draw bounding box if present
      if (box) {
        const bx = mirrorCamera ? (1.0 - box.xmax) * w : box.xmin * w;
        const by = box.ymin * h;
        const bw = box.width * w;
        const bh = box.height * h;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(bx, by, bw, bh);
        ctx.setLineDash([]);

        // Handedness label tag
        ctx.fillStyle = color;
        ctx.fillRect(bx, Math.max(0, by - 20), 80, 20);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText(label, bx + 6, Math.max(14, by - 6));
      }
    };

    // Draw Right Hand (Teal) and Left Hand (Indigo)
    renderHandSkeleton(rightLandmarks, rightBox, 'Right Hand', '#14b8a6', '#0d9488');
    renderHandSkeleton(leftLandmarks, leftBox, 'Left Hand', '#6366f1', '#4f46e5');
  }, [cameraActive, showOverlay, leftLandmarks, rightLandmarks, leftBox, rightBox, mirrorCamera]);

  // Hand Focus Inset Viewport Render
  useEffect(() => {
    if (viewMode === 'full' || !cameraActive) return;

    const canvas = focusCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.readyState < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const targetBox = rightBox || leftBox;
    if (!targetBox) {
      // Clear empty state
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#64748b';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No hand detected in focus frame', canvas.width / 2, canvas.height / 2);
      return;
    }

    // Smooth bounding box coordinates using EMA filter
    if (!smoothBoxRef.current) {
      smoothBoxRef.current = { ...targetBox };
    } else {
      const alpha = 0.25;
      smoothBoxRef.current = {
        xmin: smoothBoxRef.current.xmin * (1 - alpha) + targetBox.xmin * alpha,
        ymin: smoothBoxRef.current.ymin * (1 - alpha) + targetBox.ymin * alpha,
        xmax: smoothBoxRef.current.xmax * (1 - alpha) + targetBox.xmax * alpha,
        ymax: smoothBoxRef.current.ymax * (1 - alpha) + targetBox.ymax * alpha,
        width: smoothBoxRef.current.width * (1 - alpha) + targetBox.width * alpha,
        height: smoothBoxRef.current.height * (1 - alpha) + targetBox.height * alpha,
      };
    }

    const sb = smoothBoxRef.current;
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    const sx = Math.max(0, sb.xmin * vw);
    const sy = Math.max(0, sb.ymin * vh);
    const sw = Math.min(vw - sx, sb.width * vw);
    const sh = Math.min(vh - sy, sb.height * vh);

    const cw = canvas.width;
    const ch = canvas.height;

    ctx.clearRect(0, 0, cw, ch);
    ctx.save();
    if (mirrorCamera) {
      ctx.translate(cw, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, cw, ch);
    ctx.restore();

    // Draw hand skeleton on close-up crop
    const pts = rightBox ? rightLandmarks : leftLandmarks;
    if (pts && pts.length >= 21) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#14b8a6';
      for (const [s, e] of HAND_CONNECTIONS) {
        const p1 = pts[s];
        const p2 = pts[e];
        if (p1?.valid && p2?.valid) {
          const x1 = ((p1.x - sb.xmin) / sb.width) * cw;
          const y1 = ((p1.y - sb.ymin) / sb.height) * ch;
          const x2 = ((p2.x - sb.xmin) / sb.width) * cw;
          const y2 = ((p2.y - sb.ymin) / sb.height) * ch;

          const mx1 = mirrorCamera ? cw - x1 : x1;
          const mx2 = mirrorCamera ? cw - x2 : x2;

          ctx.beginPath();
          ctx.moveTo(mx1, y1);
          ctx.lineTo(mx2, y2);
          ctx.stroke();
        }
      }

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        if (!p.valid) continue;
        const x = ((p.x - sb.xmin) / sb.width) * cw;
        const y = ((p.y - sb.ymin) / sb.height) * ch;
        const mx = mirrorCamera ? cw - x : x;

        ctx.fillStyle = [4, 8, 12, 16, 20].includes(i) ? '#ffffff' : '#14b8a6';
        ctx.beginPath();
        ctx.arc(mx, y, i === 0 ? 6 : 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }, [viewMode, cameraActive, leftBox, rightBox, leftLandmarks, rightLandmarks, mirrorCamera]);

  // Distinct Recognition State Banner
  const renderStateBanner = () => {
    switch (stateCode) {
      case 'STATE_A_INACTIVE':
        return (
          <div className="bg-slate-900/80 backdrop-blur-xs text-slate-300 text-xs px-3 py-1.5 rounded-lg border border-slate-700/60 flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            <span>Camera is off</span>
          </div>
        );
      case 'STATE_B_NO_HAND':
        return (
          <div className="bg-slate-900/80 backdrop-blur-xs text-amber-300 text-xs px-3 py-1.5 rounded-lg border border-amber-800/60 flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-amber-400" />
            <span>No hand detected — place hand in center frame</span>
          </div>
        );
      case 'STATE_C_COLLECTING':
        return (
          <div className="bg-teal-950/80 backdrop-blur-xs text-teal-300 text-xs px-3 py-1.5 rounded-lg border border-teal-700/60 flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-teal-400 animate-ping" />
            <span>Hand detected — collecting sign motion...</span>
          </div>
        );
      case 'STATE_D_CANDIDATE':
        return (
          <div className="bg-indigo-950/80 backdrop-blur-xs text-indigo-300 text-xs px-3 py-1.5 rounded-lg border border-indigo-700/60 flex items-center space-x-2">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
            <span>
              Recognizing sign: <strong>{candidateWord?.toUpperCase()}</strong>
            </span>
          </div>
        );
      case 'STATE_E_COMMITTED':
        return (
          <div className="bg-emerald-950/80 backdrop-blur-xs text-emerald-300 text-xs px-3 py-1.5 rounded-lg border border-emerald-600/60 flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>Word recognized and committed to transcript!</span>
          </div>
        );
      case 'STATE_F_UNCERTAIN':
        return (
          <div className="bg-rose-950/80 backdrop-blur-xs text-rose-300 text-xs px-3 py-1.5 rounded-lg border border-rose-800/60 flex items-center space-x-2">
            <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
            <span>Sign uncertain — improve hand visibility or try again</span>
          </div>
        );
      case 'STATE_G_MODEL_UNAVAILABLE':
        return (
          <div className="bg-rose-950/80 backdrop-blur-xs text-rose-300 text-xs px-3 py-1.5 rounded-lg border border-rose-800/60 flex items-center space-x-2">
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
            <span>Recognition model unavailable (checkpoint missing)</span>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-slate-900/90 backdrop-blur-md rounded-2xl border border-slate-800 shadow-2xl p-4 sm:p-5 flex flex-col space-y-4 text-slate-100">
      {/* Panel Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-teal-500/15 border border-teal-500/30 flex items-center justify-center text-teal-400">
            <Camera className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm sm:text-base font-bold text-white tracking-tight">Live Camera Workspace</h2>
            <p className="text-[11px] text-slate-400">Holistic 553 landmarks tracking & 21-pt hand pose</p>
          </div>
        </div>

        {/* View Mode & Overlay Controls */}
        <div className="flex items-center space-x-2">
          <div className="bg-slate-800/90 p-0.5 rounded-xl border border-slate-700 flex items-center text-xs">
            <button
              onClick={() => setViewMode('full')}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                viewMode === 'full'
                  ? 'bg-teal-500 text-slate-950 shadow-md font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Full Camera View"
            >
              Full
            </button>
            <button
              onClick={() => setViewMode('focus')}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                viewMode === 'focus'
                  ? 'bg-teal-500 text-slate-950 shadow-md font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Hand Focus Zoom"
            >
              Focus
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                viewMode === 'split'
                  ? 'bg-teal-500 text-slate-950 shadow-md font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Split View"
            >
              Split
            </button>
          </div>

          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`p-2 rounded-xl border transition-all ${
              showOverlay
                ? 'bg-teal-500/20 border-teal-500/50 text-teal-300 shadow-sm'
                : 'bg-slate-800 border-slate-700 text-slate-500 hover:text-slate-300'
            }`}
            title={showOverlay ? 'Hide 21-pt Skeleton' : 'Show 21-pt Skeleton'}
          >
            {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Viewport Grid */}
      <div
        className={`grid gap-3 ${
          viewMode === 'split' ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'
        }`}
      >
        {/* Primary Camera Viewport */}
        {viewMode !== 'focus' && (
          <div className="relative aspect-[4/3] bg-slate-950 rounded-xl overflow-hidden flex items-center justify-center border border-slate-800 shadow-inner group">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover ${mirrorCamera ? '-scale-x-100' : ''}`}
            />
            <canvas ref={captureCanvasRef} className="hidden" />

            {/* Skeleton Overlay Canvas */}
            <canvas
              ref={overlayCanvasRef}
              width={640}
              height={480}
              className="absolute inset-0 w-full h-full pointer-events-none object-cover"
            />

            {/* Camera Inactive State */}
            {!cameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/85 text-slate-400 p-6 text-center space-y-3 backdrop-blur-xs">
                <div className="w-14 h-14 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center text-teal-400 shadow-xl">
                  <Video className="w-7 h-7" />
                </div>
                <div>
                  <p className="font-bold text-white text-sm">Camera is Standby</p>
                  <p className="text-xs text-slate-400 mt-1 max-w-xs">
                    Click "Start Camera" to initiate live video capture, 553-point tracking, and word-level recognition.
                  </p>
                </div>
                <button
                  onClick={() => startCamera()}
                  className="mt-2 inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 text-slate-950 text-xs font-bold shadow-lg shadow-teal-500/20 active:scale-95 transition-all"
                >
                  <Camera className="w-4 h-4" />
                  <span>Start Camera</span>
                </button>
              </div>
            )}

            {/* In-Frame Status & Framing Guide */}
            {cameraActive && (
              <div className="absolute inset-0 pointer-events-none p-3 flex flex-col justify-between">
                <div className="flex justify-between items-start gap-2">
                  {renderStateBanner()}
                  {performance && (
                    <div className="bg-slate-900/85 backdrop-blur-md text-slate-300 text-[11px] px-2.5 py-1 rounded-lg border border-slate-700/80 font-mono shadow-md">
                      <span className="text-teal-400 font-bold">{performance.fps} FPS</span> | {performance.total_latency_ms}ms
                    </div>
                  )}
                </div>

                {/* Posture Framing Area Box */}
                <div className="self-center w-3/4 h-3/4 border-2 border-dashed border-teal-500/30 rounded-2xl flex flex-col items-center justify-center pointer-events-none p-2 text-center">
                  <span className="text-[10px] text-teal-400/50 font-mono uppercase tracking-widest font-semibold">
                    Sign Framing Guide
                  </span>
                  <span className="text-[9px] text-slate-400/70 mt-1 max-w-[200px]">
                    Position head, shoulders & hands in view
                  </span>
                </div>

                {/* Bottom Modality Status Chips */}
                <div className="flex items-center space-x-2">
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-mono border ${
                      landmarksStatus?.pose
                        ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                        : 'bg-slate-900/80 border-slate-800 text-slate-500'
                    }`}
                  >
                    Pose: {landmarksStatus?.pose ? 'Tracking' : 'Searching'}
                  </span>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-mono border ${
                      landmarksStatus?.left_hand
                        ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-300'
                        : 'bg-slate-900/80 border-slate-800 text-slate-500'
                    }`}
                  >
                    L-Hand: {landmarksStatus?.left_hand ? 'Tracking' : 'None'}
                  </span>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-mono border ${
                      landmarksStatus?.right_hand
                        ? 'bg-teal-500/20 border-teal-500/50 text-teal-300'
                        : 'bg-slate-900/80 border-slate-800 text-slate-500'
                    }`}
                  >
                    R-Hand: {landmarksStatus?.right_hand ? 'Tracking' : 'None'}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Hand Focus Close-Up Viewport */}
        {(viewMode === 'focus' || viewMode === 'split') && (
          <div className="relative aspect-[4/3] bg-slate-950 rounded-xl overflow-hidden flex flex-col items-center justify-center border border-slate-800 shadow-inner">
            <canvas ref={focusCanvasRef} width={480} height={360} className="w-full h-full object-contain" />
            <div className="absolute top-2 left-2 bg-slate-900/85 backdrop-blur-md text-slate-300 text-[10px] px-2.5 py-1 rounded-lg font-mono border border-slate-700/80 flex items-center space-x-1.5 shadow-md">
              <ZoomIn className="w-3.5 h-3.5 text-teal-400" />
              <span>Hand Focus Close-Up</span>
            </div>
          </div>
        )}
      </div>

      {/* Permission Error Banner */}
      {permissionError && (
        <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-rose-200">Camera Access Error</p>
            <p className="mt-0.5">{permissionError}</p>
          </div>
        </div>
      )}

      {/* Bottom Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800/80">
        <div className="flex items-center space-x-2">
          {!cameraActive ? (
            <button
              onClick={() => startCamera()}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 text-slate-950 text-xs font-bold shadow-lg shadow-teal-500/20 active:scale-95 transition-all"
            >
              <Camera className="w-4 h-4" />
              <span>Start Camera</span>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-gradient-to-r from-rose-600 to-rose-700 hover:from-rose-500 hover:to-rose-600 text-white text-xs font-bold shadow-lg shadow-rose-500/20 active:scale-95 transition-all"
            >
              <CameraOff className="w-4 h-4" />
              <span>Stop Camera</span>
            </button>
          )}

          {cameraActive && (
            <button
              onClick={() => setIsStreaming(!isStreaming)}
              className={`inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl border text-xs font-semibold transition-all ${
                isStreaming
                  ? 'bg-amber-500/15 border-amber-500/40 text-amber-300 hover:bg-amber-500/25'
                  : 'bg-teal-500/15 border-teal-500/40 text-teal-300 hover:bg-teal-500/25'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>{isStreaming ? 'Pause Stream' : 'Resume'}</span>
            </button>
          )}
        </div>

        {/* Mirror and Device Selection */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onToggleMirror}
            className={`p-2 rounded-xl border text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              mirrorCamera
                ? 'bg-teal-500/15 border-teal-500/40 text-teal-300'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200'
            }`}
            title="Toggle Mirror Camera"
          >
            <FlipHorizontal className="w-4 h-4" />
            <span className="hidden sm:inline">Mirror</span>
          </button>

          {devices.length > 1 && (
            <button
              onClick={handleFlipCamera}
              className="p-2 rounded-xl border text-xs font-semibold bg-slate-800 border-slate-700 text-slate-300 hover:text-white flex items-center space-x-1.5 transition-all"
              title="Flip / Switch Front and Back Camera"
            >
              <SwitchCamera className="w-4 h-4 text-teal-400" />
              <span className="hidden sm:inline">Flip</span>
            </button>
          )}

          {/* Camera Selection Dropdown */}
          <div className="relative flex items-center">
            <Video className="w-3.5 h-3.5 text-cyan-400 absolute left-2.5 pointer-events-none" />
            <select
              value={currentDeviceId}
              onChange={(e) => handleDeviceChange(e.target.value)}
              className="text-xs bg-slate-800/90 hover:bg-slate-800 border border-slate-700 rounded-xl pl-8 pr-3 py-2 text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500/50 cursor-pointer transition-all max-w-[180px] sm:max-w-[220px] truncate"
              title="Switch Camera Input Device"
            >
              {devices.length === 0 ? (
                <option value="">Default Camera</option>
              ) : (
                devices.map((d, idx) => (
                  <option key={d.deviceId || idx} value={d.deviceId} className="bg-slate-900 text-slate-200">
                    {d.label || `Camera ${idx + 1}`}
                  </option>
                ))
              )}
            </select>
          </div>
        </div>
      </div>
    </div>
  );
};
