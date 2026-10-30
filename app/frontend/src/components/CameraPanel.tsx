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
  User,
  Sliders,
  CheckCircle2,
  Info,
  Zap,
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
  poseLandmarks?: LandmarkPoint[];
  faceLandmarks?: LandmarkPoint[];
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

// 21 Hand connections (MediaPipe Hands standard topology)
export const HAND_CONNECTIONS: [number, number][] = [
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

// Upper-body pose connections (MediaPipe Pose 33-point standard topology)
export const POSE_UPPER_BODY_CONNECTIONS: [number, number][] = [
  // Shoulders & upper torso
  [11, 12], // left shoulder to right shoulder
  [11, 23], // left shoulder to left hip
  [12, 24], // right shoulder to right hip
  [23, 24], // left hip to right hip
  // Left arm
  [11, 13], // left shoulder to left elbow
  [13, 15], // left elbow to left wrist
  // Right arm
  [12, 14], // right shoulder to right elbow
  [14, 16], // right elbow to right wrist
  // Neck & Head anchors
  [11, 0],  // left shoulder to nose
  [12, 0],  // right shoulder to nose
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],   // left eye to left ear
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],   // right eye to right ear
  [9, 10],  // mouth
];

// Curated MediaPipe Face Mesh Contours (Indices from canonical 478-point topology)
export const FACE_OVAL: number[] = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109, 10,
];

export const RIGHT_EYEBROW: number[] = [70, 63, 105, 66, 107];
export const LEFT_EYEBROW: number[] = [336, 296, 334, 293, 300];

export const RIGHT_EYE: number[] = [
  33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33,
];

export const LEFT_EYE: number[] = [
  362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362,
];

export const LIPS_OUTER: number[] = [
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
  37, 39, 40, 185, 61,
];

export const LIPS_INNER: number[] = [
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
  82, 81, 80, 191, 78,
];

export const NOSE_BRIDGE: number[] = [168, 6, 197, 195, 5, 4, 1];
export const NOSE_BOTTOM: number[] = [98, 97, 2, 326, 327];

export const FACE_CONTOUR_CHAINS: number[][] = [
  FACE_OVAL,
  RIGHT_EYEBROW,
  LEFT_EYEBROW,
  RIGHT_EYE,
  LEFT_EYE,
  LIPS_OUTER,
  LIPS_INNER,
  NOSE_BRIDGE,
  NOSE_BOTTOM,
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
  poseLandmarks = [],
  faceLandmarks = [],
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

  // View mode defaults to upper_body (preserves full camera frame without crop)
  const [viewMode, setViewMode] = useState<CameraViewMode>('upper_body');

  // Visualization overlay toggles
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [showHands, setShowHands] = useState<boolean>(true);
  const [showPose, setShowPose] = useState<boolean>(true);
  const [showFace, setShowFace] = useState<boolean>(true);
  const [showDebugPoints, setShowDebugPoints] = useState<boolean>(false);

  // Hardware zoom support inspection
  const [hardwareZoomAvailable, setHardwareZoomAvailable] = useState<boolean>(false);
  const [zoomRange, setZoomRange] = useState<{ min: number; max: number; step: number }>({ min: 1, max: 1, step: 0.1 });
  const [currentZoom, setCurrentZoom] = useState<number>(1);

  const devices = videoDevices && videoDevices.length > 0 ? videoDevices : localDevices;
  const currentDeviceId = controlledDeviceId !== undefined ? controlledDeviceId : localSelectedId;

  const handleDeviceChange = (deviceId: string) => {
    setLocalSelectedId(deviceId);
    if (onSelectDevice) {
      onSelectDevice(deviceId);
    }
  };

  // Smoothed bounding box for Hands Detail inset
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

  // Inspect hardware zoom capability safely on stream
  const inspectZoomCapabilities = (stream: MediaStream) => {
    try {
      const track = stream.getVideoTracks()[0];
      if (track && typeof track.getCapabilities === 'function') {
        const caps = track.getCapabilities() as any;
        if (caps.zoom && typeof caps.zoom.min === 'number' && typeof caps.zoom.max === 'number') {
          setHardwareZoomAvailable(true);
          setZoomRange({
            min: caps.zoom.min,
            max: caps.zoom.max,
            step: caps.zoom.step || 0.1,
          });
          const settings = (typeof track.getSettings === 'function' ? track.getSettings() : {}) as any;
          if (typeof settings.zoom === 'number') {
            setCurrentZoom(settings.zoom);
          } else {
            setCurrentZoom(caps.zoom.min);
          }
          return;
        }
      }
    } catch {
      // Zoom inspection unsupported or rejected safely
    }
    setHardwareZoomAvailable(false);
  };

  // Apply optical hardware zoom if supported by track capabilities
  const handleHardwareZoom = async (newZoom: number) => {
    if (!hardwareZoomAvailable || !streamRef.current) return;
    const clamped = Math.max(zoomRange.min, Math.min(zoomRange.max, newZoom));
    setCurrentZoom(clamped);
    try {
      const track = streamRef.current.getVideoTracks()[0];
      if (track && typeof track.applyConstraints === 'function') {
        await track.applyConstraints({
          advanced: [{ zoom: clamped }] as any,
        });
      }
    } catch (err) {
      console.debug('Camera hardware zoom constraint ignored:', err);
    }
  };

  // Start Camera Stream with HD constraints for razor-sharp camera quality
  const startCamera = async (targetDeviceId?: string) => {
    setPermissionError(null);
    const devId = targetDeviceId !== undefined ? targetDeviceId : currentDeviceId;
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      // Request crisp HD 720p/1080p stream for preview quality; downsampled on hidden capture canvas
      const constraints: MediaStreamConstraints = {
        video: devId
          ? {
              deviceId: { exact: devId },
              width: { ideal: 1280, max: 1920 },
              height: { ideal: 720, max: 1080 },
              frameRate: { ideal: 30, max: 60 },
            }
          : {
              facingMode: 'user',
              width: { ideal: 1280, max: 1920 },
              height: { ideal: 720, max: 1080 },
              frameRate: { ideal: 30, max: 60 },
            },
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

      // Inspect camera capabilities for physical zoom
      inspectZoomCapabilities(stream);

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

          const ctx = canvas.getContext('2d', { willReadFrequently: false });
          if (ctx) {
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(video, 0, 0, targetW, targetH);
            frameIdCounterRef.current += 1;
            const currentFrameId = frameIdCounterRef.current;
            inFlightFrameIdRef.current = currentFrameId;
            inFlightRef.current = true;
            lastSendTime = now;

            const dataUrl = canvas.toDataURL('image/jpeg', 0.65);
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

  // Naik123-Style Multi-Modality Overlay Renderer
  // (Radial glow + linear gradient bones + depth-based point sizing + white-rim fingertips)
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !cameraActive || !showOverlay) {
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    if (video.videoWidth && video.videoHeight) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Canonical source-to-canvas mapping function
    // Mirroring is applied strictly once to coordinates when preview is mirrored
    const mapX = (x: number) => (mirrorCamera ? (1.0 - x) * w : x * w);
    const mapY = (y: number) => y * h;

    // 1. Render Upper-Body Pose Skeleton (MediaPipe 33-pt Schema) with Cyan/SkyBlue Gradients
    if (showPose && poseLandmarks && poseLandmarks.length >= 33) {
      const hasValidPose = poseLandmarks.some((p) => p.valid);
      if (hasValidPose) {
        ctx.lineWidth = 2.8;
        ctx.lineCap = 'round';
        ctx.shadowColor = '#00f5d4';
        ctx.shadowBlur = 6;

        // Draw pose bones with linear gradients (strictly only when BOTH endpoints are valid)
        for (const [startIdx, endIdx] of POSE_UPPER_BODY_CONNECTIONS) {
          const p1 = poseLandmarks[startIdx];
          const p2 = poseLandmarks[endIdx];
          if (p1?.valid && p2?.valid) {
            const x1 = mapX(p1.x);
            const y1 = mapY(p1.y);
            const x2 = mapX(p2.x);
            const y2 = mapY(p2.y);

            const grad = ctx.createLinearGradient(x1, y1, x2, y2);
            grad.addColorStop(0, 'rgba(56, 189, 248, 0.9)'); // Sky blue
            grad.addColorStop(1, 'rgba(0, 245, 212, 0.9)');  // Cyan
            ctx.strokeStyle = grad;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }

        // Draw pose joints with depth sizing
        ctx.shadowBlur = 0;
        for (let i = 0; i < 33; i++) {
          const p = poseLandmarks[i];
          if (!p?.valid) {
            if (showDebugPoints) {
              ctx.strokeStyle = '#f43f5e';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.arc(mapX(p.x), mapY(p.y), 2.5, 0, 2 * Math.PI);
              ctx.stroke();
            }
            continue;
          }

          const cx = mapX(p.x);
          const cy = mapY(p.y);
          const depth = Math.max(0.6, Math.min(1.8, 1 - (p.z || 0) * 3));
          const isMajorJoint = [11, 12, 13, 14, 15, 16, 23, 24].includes(i);
          const r = (isMajorJoint ? 5.0 : 3.5) * depth;

          // Outer halo
          ctx.beginPath();
          ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(56, 189, 248, 0.2)';
          ctx.fill();

          // Core dot
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.fillStyle = isMajorJoint ? '#38bdf8' : '#00f5d4';
          ctx.fill();

          ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      }
    }

    // 2. Render Curated Face Contours (MediaPipe 478-pt Schema) with Glowing Rose Pink
    if (showFace && faceLandmarks && faceLandmarks.length >= 468) {
      const hasValidFace = faceLandmarks.some((p) => p.valid);
      if (hasValidFace) {
        ctx.lineWidth = 1.6;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'rgba(255, 107, 157, 0.85)'; // Naik123 pink
        ctx.shadowColor = '#ff6b9d';
        ctx.shadowBlur = 6;

        // Draw contour paths (strictly only when BOTH consecutive endpoints are valid)
        for (const chain of FACE_CONTOUR_CHAINS) {
          for (let i = 0; i < chain.length - 1; i++) {
            const idx1 = chain[i];
            const idx2 = chain[i + 1];
            const p1 = faceLandmarks[idx1];
            const p2 = faceLandmarks[idx2];
            if (p1?.valid && p2?.valid) {
              ctx.beginPath();
              ctx.moveTo(mapX(p1.x), mapY(p1.y));
              ctx.lineTo(mapX(p2.x), mapY(p2.y));
              ctx.stroke();
            }
          }
        }

        // Optional debug points
        ctx.shadowBlur = 0;
        if (showDebugPoints) {
          ctx.fillStyle = '#ff6b9d';
          for (let i = 0; i < faceLandmarks.length; i += 4) {
            const p = faceLandmarks[i];
            if (p?.valid) {
              ctx.fillRect(mapX(p.x) - 1, mapY(p.y) - 1, 2, 2);
            }
          }
        }
      }
    }

    // 3. Render Hand Skeletons with Naik123-Style Aesthetic
    if (showHands) {
      const renderHandSkeleton = (
        points: LandmarkPoint[],
        box: BoundingBox | null,
        label: string,
        isRight: boolean
      ) => {
        if (!points || points.length < 21) return;
        const validPoints = points.filter((p) => p.valid);
        if (validPoints.length === 0) return;

        // A. Ambient radial glow behind the hand (Naik123 signature effect)
        const sumX = validPoints.reduce((acc, p) => acc + p.x, 0);
        const sumY = validPoints.reduce((acc, p) => acc + p.y, 0);
        const centroidX = mapX(sumX / validPoints.length);
        const centroidY = mapY(sumY / validPoints.length);
        const glowRadius = Math.max(90, Math.min(180, box ? Math.max(box.width * w, box.height * h) : 130));

        const glow = ctx.createRadialGradient(centroidX, centroidY, 10, centroidX, centroidY, glowRadius);
        if (isRight) {
          glow.addColorStop(0, 'rgba(0, 245, 212, 0.16)');
          glow.addColorStop(1, 'rgba(0, 245, 212, 0)');
        } else {
          glow.addColorStop(0, 'rgba(123, 97, 255, 0.16)');
          glow.addColorStop(1, 'rgba(123, 97, 255, 0)');
        }
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, w, h);

        // B. Cyan-to-Purple Gradient Bones (Naik123 signature style)
        ctx.lineWidth = 3.2;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.shadowColor = isRight ? '#00f5d4' : '#7b61ff';
        ctx.shadowBlur = 8;

        for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
          const p1 = points[startIdx];
          const p2 = points[endIdx];
          if (p1?.valid && p2?.valid) {
            const x1 = mapX(p1.x);
            const y1 = mapY(p1.y);
            const x2 = mapX(p2.x);
            const y2 = mapY(p2.y);

            const grad = ctx.createLinearGradient(x1, y1, x2, y2);
            if (isRight) {
              grad.addColorStop(0, 'rgba(0, 245, 212, 0.95)');  // Neon Cyan
              grad.addColorStop(1, 'rgba(123, 97, 255, 0.95)'); // Purple
            } else {
              grad.addColorStop(0, 'rgba(123, 97, 255, 0.95)'); // Purple
              grad.addColorStop(1, 'rgba(255, 107, 157, 0.95)'); // Pink
            }
            ctx.strokeStyle = grad;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }

        // C. 21 Joints with depth-based sizing (z-depth) & fingertip highlight (Naik123 style)
        ctx.shadowBlur = 0;
        for (let i = 0; i < points.length; i++) {
          const p = points[i];
          if (!p.valid) {
            if (showDebugPoints) {
              ctx.strokeStyle = '#f43f5e';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.arc(mapX(p.x), mapY(p.y), 3, 0, 2 * Math.PI);
              ctx.stroke();
            }
            continue;
          }

          const cx = mapX(p.x);
          const cy = mapY(p.y);
          const depth = Math.max(0.6, Math.min(1.8, 1 - (p.z || 0) * 3));
          const isTip = [4, 8, 12, 16, 20].includes(i);
          const isWrist = i === 0;
          const baseR = isWrist ? 6.0 : isTip ? 5.5 : 4.0;
          const r = baseR * depth;

          // Outer soft halo
          ctx.beginPath();
          ctx.arc(cx, cy, r + 5, 0, Math.PI * 2);
          ctx.fillStyle = isTip
            ? 'rgba(255, 107, 157, 0.28)'
            : isRight
            ? 'rgba(0, 245, 212, 0.20)'
            : 'rgba(123, 97, 255, 0.20)';
          ctx.fill();

          // Core dot
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.fillStyle = isTip
            ? '#ff6b9d' // Naik123 fingertip pink
            : isRight
            ? '#00f5d4' // Naik123 cyan
            : '#7b61ff'; // Naik123 purple
          ctx.fill();

          // Crisp white rim stroke
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
          ctx.stroke();
        }

        // D. Sleek Glassmorphic Bounding Box
        if (box) {
          const bx = mirrorCamera ? (1.0 - box.xmax) * w : box.xmin * w;
          const by = box.ymin * h;
          const bw = box.width * w;
          const bh = box.height * h;

          const boxColor = isRight ? '#00f5d4' : '#7b61ff';
          ctx.strokeStyle = boxColor;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect(bx, by, bw, bh);
          ctx.setLineDash([]);

          // Tag
          ctx.fillStyle = isRight ? 'rgba(0, 245, 212, 0.85)' : 'rgba(123, 97, 255, 0.85)';
          ctx.fillRect(bx, Math.max(0, by - 22), 86, 20);
          ctx.fillStyle = '#0f172a';
          ctx.font = 'bold 10px monospace';
          ctx.fillText(label.toUpperCase(), bx + 6, Math.max(14, by - 8));
        }
      };

      // Both hands rendered simultaneously
      renderHandSkeleton(rightLandmarks, rightBox, 'Right Hand', true);
      renderHandSkeleton(leftLandmarks, leftBox, 'Left Hand', false);
    }
  }, [
    cameraActive,
    showOverlay,
    showHands,
    showPose,
    showFace,
    showDebugPoints,
    leftLandmarks,
    rightLandmarks,
    poseLandmarks,
    faceLandmarks,
    leftBox,
    rightBox,
    mirrorCamera,
  ]);

  // Hands Detail Inset Viewport Render (Computes union of both hand boxes with Naik123 gradients)
  useEffect(() => {
    if (viewMode === 'full' || viewMode === 'upper_body' || !cameraActive) return;

    const canvas = focusCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.readyState < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Union bounding box computation to never discard either hand
    let targetBox: BoundingBox | null = null;
    if (leftBox && rightBox) {
      const pad = 0.04;
      const xmin = Math.max(0, Math.min(leftBox.xmin, rightBox.xmin) - pad);
      const ymin = Math.max(0, Math.min(leftBox.ymin, rightBox.ymin) - pad);
      const xmax = Math.min(1, Math.max(leftBox.xmax, rightBox.xmax) + pad);
      const ymax = Math.min(1, Math.max(leftBox.ymax, rightBox.ymax) + pad);
      targetBox = {
        xmin,
        ymin,
        xmax,
        ymax,
        width: Math.max(0.01, xmax - xmin),
        height: Math.max(0.01, ymax - ymin),
      };
    } else {
      targetBox = rightBox || leftBox;
    }

    if (!targetBox) {
      // Clear empty state
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#64748b';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No hands detected in detail frame', canvas.width / 2, canvas.height / 2);
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
    ctx.drawImage(video, sx, sy, Math.max(1, sw), Math.max(1, sh), 0, 0, cw, ch);
    ctx.restore();

    // Helper to render skeleton on detail crop with Naik123 gradients
    const renderDetailSkeleton = (pts: LandmarkPoint[], isRight: boolean) => {
      if (!pts || pts.length < 21) return;
      ctx.lineWidth = 3.0;
      ctx.lineCap = 'round';

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

          const grad = ctx.createLinearGradient(mx1, y1, mx2, y2);
          if (isRight) {
            grad.addColorStop(0, 'rgba(0, 245, 212, 0.9)');
            grad.addColorStop(1, 'rgba(123, 97, 255, 0.9)');
          } else {
            grad.addColorStop(0, 'rgba(123, 97, 255, 0.9)');
            grad.addColorStop(1, 'rgba(255, 107, 157, 0.9)');
          }
          ctx.strokeStyle = grad;

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

        const isTip = [4, 8, 12, 16, 20].includes(i);
        const depth = Math.max(0.6, Math.min(1.8, 1 - (p.z || 0) * 3));
        const r = (isTip ? 5.5 : 4.0) * depth;

        ctx.beginPath();
        ctx.arc(mx, y, r + 4, 0, Math.PI * 2);
        ctx.fillStyle = isTip ? 'rgba(255, 107, 157, 0.25)' : 'rgba(0, 245, 212, 0.2)';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(mx, y, r, 0, Math.PI * 2);
        ctx.fillStyle = isTip ? '#ff6b9d' : isRight ? '#00f5d4' : '#7b61ff';
        ctx.fill();

        ctx.lineWidth = 1.5;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.stroke();
      }
    };

    if (rightLandmarks && rightLandmarks.length >= 21) {
      renderDetailSkeleton(rightLandmarks, true);
    }
    if (leftLandmarks && leftLandmarks.length >= 21) {
      renderDetailSkeleton(leftLandmarks, false);
    }
  }, [viewMode, cameraActive, leftBox, rightBox, leftLandmarks, rightLandmarks, mirrorCamera]);

  // Derive dynamic framing guidance from landmark validity
  const getFramingGuidance = () => {
    if (!cameraActive) return null;
    if (!landmarksStatus?.pose) {
      return {
        message: 'Torso/shoulders not detected — position upper body in frame',
        type: 'warning',
      };
    }
    if (!landmarksStatus?.face) {
      return {
        message: 'Face not detected — position head in upper center frame',
        type: 'info',
      };
    }
    const allHands = [...leftLandmarks, ...rightLandmarks].filter((p) => p.valid);
    const isNearEdge = allHands.some((p) => p.x < 0.05 || p.x > 0.95 || p.y < 0.05 || p.y > 0.95);
    if (isNearEdge) {
      return {
        message: 'Hands near frame edge — step back slightly (~1.5m) for full reach',
        type: 'warning',
      };
    }
    if (landmarksStatus.left_hand && landmarksStatus.right_hand) {
      return {
        message: 'Both hands and upper-body tracking — framing optimal',
        type: 'success',
      };
    }
    if (landmarksStatus.left_hand || landmarksStatus.right_hand) {
      const activeHand = landmarksStatus.left_hand ? 'Left hand' : 'Right hand';
      return {
        message: `${activeHand} active — bring second hand in frame if signing two-handed`,
        type: 'info',
      };
    }
    return {
      message: 'Upper body aligned — raise hands into view to begin signing',
      type: 'neutral',
    };
  };

  const guidance = getFramingGuidance();

  // Distinct Recognition State Banner
  const renderStateBanner = () => {
    switch (stateCode) {
      case 'STATE_A_INACTIVE':
        return (
          <div className="bg-slate-900/85 backdrop-blur-xs text-slate-300 text-xs px-3 py-1.5 rounded-lg border border-slate-700/60 flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            <span>Camera is off</span>
          </div>
        );
      case 'STATE_B_NO_HAND':
        return (
          <div className="bg-slate-900/85 backdrop-blur-xs text-amber-300 text-xs px-3 py-1.5 rounded-lg border border-amber-800/60 flex items-center space-x-2 shadow-lg">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            <span>Ready — raise hands inside upper-body frame</span>
          </div>
        );
      case 'STATE_C_COLLECTING':
        return (
          <div className="bg-teal-950/85 backdrop-blur-xs text-teal-300 text-xs px-3 py-1.5 rounded-lg border border-teal-700/60 flex items-center space-x-2 shadow-lg shadow-teal-500/10">
            <span className="w-2 h-2 rounded-full bg-teal-400 animate-ping" />
            <span>Motion detected — collecting sign frames...</span>
          </div>
        );
      case 'STATE_D_CANDIDATE':
        return (
          <div className="bg-indigo-950/85 backdrop-blur-xs text-indigo-300 text-xs px-3 py-1.5 rounded-lg border border-indigo-700/60 flex items-center space-x-2 shadow-lg shadow-indigo-500/10">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
            <span>
              Evaluating sign: <strong>{candidateWord?.toUpperCase()}</strong>
            </span>
          </div>
        );
      case 'STATE_E_COMMITTED':
        return (
          <div className="bg-emerald-950/85 backdrop-blur-xs text-emerald-300 text-xs px-3 py-1.5 rounded-lg border border-emerald-600/60 flex items-center space-x-2 shadow-lg shadow-emerald-500/10">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>Word committed to transcript!</span>
          </div>
        );
      case 'STATE_F_UNCERTAIN':
        return (
          <div className="bg-rose-950/85 backdrop-blur-xs text-rose-300 text-xs px-3 py-1.5 rounded-lg border border-rose-800/60 flex items-center space-x-2">
            <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
            <span>Sign uncertain — check lighting and hand visibility</span>
          </div>
        );
      case 'STATE_G_MODEL_UNAVAILABLE':
        return (
          <div className="bg-rose-950/85 backdrop-blur-xs text-rose-300 text-xs px-3 py-1.5 rounded-lg border border-rose-800/60 flex items-center space-x-2">
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
            <span>Model unavailable (checkpoint not loaded)</span>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div
      className="bg-slate-900/90 backdrop-blur-md rounded-2xl border border-slate-800 p-4 sm:p-5 flex flex-col space-y-4 text-slate-100 transition-all duration-300"
      style={{
        boxShadow: cameraActive
          ? '0 0 35px rgba(0, 245, 212, 0.12), 0 10px 40px rgba(0, 0, 0, 0.6)'
          : '0 10px 30px rgba(0, 0, 0, 0.5)',
      }}
    >
      {/* Panel Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-teal-500/15 border border-teal-500/30 flex items-center justify-center text-teal-400 shadow-sm">
            <Camera className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm sm:text-base font-bold text-white tracking-tight flex items-center space-x-2">
              <span>Live Camera Workspace</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-teal-500/15 border border-teal-500/30 text-teal-300 font-mono font-medium">
                HD 60FPS
              </span>
            </h2>
            <p className="text-[11px] text-slate-400">
              Upper-Body 553-Point Holistic Tracking (Pose 33, Hands 42, Face Mesh)
            </p>
          </div>
        </div>

        {/* View Mode & Overlay Controls */}
        <div className="flex items-center space-x-2">
          {/* View Modes */}
          <div className="bg-slate-800/90 p-0.5 rounded-xl border border-slate-700 flex items-center text-xs">
            <button
              onClick={() => setViewMode('upper_body')}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                viewMode === 'upper_body'
                  ? 'bg-teal-500 text-slate-950 shadow-md font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Upper Body Wide View (Preserves Full Frame Uncropped)"
            >
              Upper Body
            </button>
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
              title="Hands Detail Close-Up"
            >
              Hands Detail
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                viewMode === 'split'
                  ? 'bg-teal-500 text-slate-950 shadow-md font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Split View (Upper Body + Hands Detail)"
            >
              Split
            </button>
          </div>

          {/* Master Overlay Toggle */}
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`p-2 rounded-xl border transition-all ${
              showOverlay
                ? 'bg-teal-500/20 border-teal-500/50 text-teal-300 shadow-sm'
                : 'bg-slate-800 border-slate-700 text-slate-500 hover:text-slate-300'
            }`}
            title={showOverlay ? 'Hide Landmark Overlays' : 'Show Landmark Overlays'}
          >
            {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Layer Toggles & Modality Legend Bar (Naik123 Visual Theme) */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 bg-slate-950/60 rounded-xl border border-slate-800 text-xs">
        {/* Color Legend */}
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-300">
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#7b61ff] shadow-[0_0_8px_#7b61ff] inline-block" />
            <span>Left Hand (21pt)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#00f5d4] shadow-[0_0_8px_#00f5d4] inline-block" />
            <span>Right Hand (21pt)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#38bdf8] shadow-[0_0_8px_#38bdf8] inline-block" />
            <span>Pose (33pt)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ff6b9d] shadow-[0_0_8px_#ff6b9d] inline-block" />
            <span>Face Mesh</span>
          </div>
        </div>

        {/* Layer Visibility Toggles */}
        <div className="flex items-center space-x-1 text-[11px]">
          <span className="text-slate-400 mr-1 text-[10px] uppercase tracking-wider font-semibold">Layers:</span>
          <button
            onClick={() => setShowHands(!showHands)}
            className={`px-2.5 py-0.5 rounded-md border font-medium transition-all ${
              showHands
                ? 'bg-teal-500/20 border-teal-500/40 text-teal-300 shadow-xs'
                : 'bg-slate-900 border-slate-800 text-slate-500'
            }`}
          >
            Hands
          </button>
          <button
            onClick={() => setShowPose(!showPose)}
            className={`px-2.5 py-0.5 rounded-md border font-medium transition-all ${
              showPose
                ? 'bg-sky-500/20 border-sky-500/40 text-sky-300 shadow-xs'
                : 'bg-slate-900 border-slate-800 text-slate-500'
            }`}
          >
            Pose
          </button>
          <button
            onClick={() => setShowFace(!showFace)}
            className={`px-2.5 py-0.5 rounded-md border font-medium transition-all ${
              showFace
                ? 'bg-pink-500/20 border-pink-500/40 text-pink-300 shadow-xs'
                : 'bg-slate-900 border-slate-800 text-slate-500'
            }`}
          >
            Face
          </button>
          <button
            onClick={() => setShowDebugPoints(!showDebugPoints)}
            className={`px-2.5 py-0.5 rounded-md border font-medium transition-all ${
              showDebugPoints
                ? 'bg-amber-500/20 border-amber-500/40 text-amber-300 shadow-xs'
                : 'bg-slate-900 border-slate-800 text-slate-500'
            }`}
            title="Toggle dense point markers"
          >
            Points
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
          <div className="relative aspect-[4/3] bg-slate-950 rounded-2xl overflow-hidden flex items-center justify-center border border-slate-800 shadow-inner group">
            {/*
              In Upper Body mode, object-contain is strictly used to display the full source image
              without center-cropping head, shoulders, or extended hands.
            */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full ${
                viewMode === 'upper_body' ? 'object-contain' : 'object-cover'
              } ${mirrorCamera ? '-scale-x-100' : ''}`}
            />
            <canvas ref={captureCanvasRef} className="hidden" />

            {/* Skeleton Overlay Canvas */}
            <canvas
              ref={overlayCanvasRef}
              width={640}
              height={480}
              className={`absolute inset-0 w-full h-full pointer-events-none ${
                viewMode === 'upper_body' ? 'object-contain' : 'object-cover'
              }`}
            />

            {/* Camera Inactive State */}
            {!cameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/90 text-slate-400 p-6 text-center space-y-3 backdrop-blur-xs">
                <div className="w-14 h-14 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center text-teal-400 shadow-xl">
                  <Video className="w-7 h-7" />
                </div>
                <div>
                  <p className="font-bold text-white text-sm">Camera Standby</p>
                  <p className="text-xs text-slate-400 mt-1 max-w-xs">
                    Start camera to initiate real-time 553-point holistic tracking and word recognition.
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

            {/* Live Status Header & Framing Overlay */}
            {cameraActive && (
              <div className="absolute inset-0 pointer-events-none p-3 flex flex-col justify-between">
                <div className="flex justify-between items-start gap-2">
                  {renderStateBanner()}
                  {performance && (
                    <div className="bg-slate-900/90 backdrop-blur-md text-slate-300 text-[11px] px-3 py-1 rounded-xl border border-slate-700/80 font-mono shadow-xl flex items-center space-x-2">
                      <Zap className="w-3.5 h-3.5 text-teal-400 animate-pulse" />
                      <span className="text-teal-400 font-bold">{performance.fps} FPS</span>
                      <span className="text-slate-500">|</span>
                      <span>Ext: {performance.extraction_ms}ms</span>
                      <span className="text-slate-500">|</span>
                      <span>Inf: {performance.inference_ms}ms</span>
                    </div>
                  )}
                </div>

                {/* Upper Body Framing Guide & Dynamic Assistant */}
                <div className="self-center w-5/6 h-5/6 border-2 border-dashed border-teal-500/25 rounded-3xl flex flex-col items-center justify-between pointer-events-none p-3 text-center relative">
                  {/* Head framing oval indicator */}
                  <div className="w-24 h-28 border border-dashed border-pink-400/30 rounded-full mt-1 flex items-center justify-center">
                    <span className="text-[9px] text-pink-300/50 uppercase font-mono tracking-wider">Head</span>
                  </div>

                  {/* Dynamic Positioning Feedback Banner */}
                  {guidance && (
                    <div
                      className={`px-3 py-1.5 rounded-xl text-xs font-medium border backdrop-blur-md shadow-xl transition-all ${
                        guidance.type === 'warning'
                          ? 'bg-amber-950/85 border-amber-500/40 text-amber-200'
                          : guidance.type === 'success'
                          ? 'bg-emerald-950/85 border-emerald-500/40 text-emerald-200'
                          : guidance.type === 'info'
                          ? 'bg-sky-950/85 border-sky-500/40 text-sky-200'
                          : 'bg-slate-900/85 border-slate-700/60 text-slate-300'
                      }`}
                    >
                      {guidance.message}
                    </div>
                  )}

                  {/* Torso & Hand Signing Region boundary indicator */}
                  <div className="w-full flex justify-between items-end px-4 mb-1">
                    <span className="text-[9px] text-indigo-400/50 font-mono uppercase">Left Hand Zone</span>
                    <span className="text-[9px] text-slate-400/50 font-mono uppercase">Waist / Torso</span>
                    <span className="text-[9px] text-teal-400/50 font-mono uppercase">Right Hand Zone</span>
                  </div>
                </div>

                {/* Bottom Modality Status Badges */}
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-mono border ${
                      landmarksStatus?.pose
                        ? 'bg-sky-500/20 border-sky-500/50 text-sky-300'
                        : 'bg-slate-900/80 border-slate-800 text-slate-500'
                    }`}
                  >
                    Pose: {landmarksStatus?.pose ? 'Tracking (33)' : 'Searching'}
                  </span>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-mono border ${
                      landmarksStatus?.face
                        ? 'bg-pink-500/20 border-pink-500/50 text-pink-300'
                        : 'bg-slate-900/80 border-slate-800 text-slate-500'
                    }`}
                  >
                    Face: {landmarksStatus?.face ? 'Tracking (478)' : 'Searching'}
                  </span>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-mono border ${
                      landmarksStatus?.left_hand
                        ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-300'
                        : 'bg-slate-900/80 border-slate-800 text-slate-500'
                    }`}
                  >
                    L-Hand: {landmarksStatus?.left_hand ? 'Tracking (21)' : 'None'}
                  </span>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-mono border ${
                      landmarksStatus?.right_hand
                        ? 'bg-teal-500/20 border-teal-500/50 text-teal-300'
                        : 'bg-slate-900/80 border-slate-800 text-slate-500'
                    }`}
                  >
                    R-Hand: {landmarksStatus?.right_hand ? 'Tracking (21)' : 'None'}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Hands Detail Close-Up Viewport */}
        {(viewMode === 'focus' || viewMode === 'split') && (
          <div className="relative aspect-[4/3] bg-slate-950 rounded-2xl overflow-hidden flex flex-col items-center justify-center border border-slate-800 shadow-inner">
            <canvas ref={focusCanvasRef} width={480} height={360} className="w-full h-full object-contain" />
            <div className="absolute top-2 left-2 bg-slate-900/85 backdrop-blur-md text-slate-300 text-[10px] px-2.5 py-1 rounded-lg font-mono border border-slate-700/80 flex items-center space-x-1.5 shadow-md">
              <ZoomIn className="w-3.5 h-3.5 text-teal-400" />
              <span>Dual-Hand Detail (Union Crop)</span>
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
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800/80">
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

        {/* Hardware Optical Zoom Controls (Active only when camera hardware supports zoom) */}
        {hardwareZoomAvailable && cameraActive && (
          <div className="flex items-center space-x-2 bg-slate-800/80 px-3 py-1.5 rounded-xl border border-slate-700 text-xs">
            <Sliders className="w-3.5 h-3.5 text-teal-400" />
            <span className="text-[11px] text-slate-300">Hardware Zoom:</span>
            <input
              type="range"
              min={zoomRange.min}
              max={zoomRange.max}
              step={zoomRange.step}
              value={currentZoom}
              onChange={(e) => handleHardwareZoom(parseFloat(e.target.value))}
              className="w-20 accent-teal-500 cursor-pointer"
            />
            <span className="font-mono text-[11px] text-teal-300">{currentZoom.toFixed(1)}x</span>
          </div>
        )}

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
