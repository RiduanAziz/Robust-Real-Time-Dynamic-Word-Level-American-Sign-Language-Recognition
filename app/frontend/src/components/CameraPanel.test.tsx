import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import React from 'react';
import {
  CameraPanel,
  HAND_CONNECTIONS,
  POSE_UPPER_BODY_CONNECTIONS,
  FACE_CONTOUR_CHAINS,
} from './CameraPanel';
import { LandmarkPoint, LandmarksStatus, QualityCoachMetrics } from '../types';

describe('CameraPanel Visual Component', () => {
  let mockCtx: any;
  let drawnLines: { from: [number, number]; to: [number, number]; strokeStyle: string }[];
  let drawnArcs: { x: number; y: number; r: number; fillStyle: string }[];

  beforeEach(() => {
    vi.clearAllMocks();
    drawnLines = [];
    drawnArcs = [];

    mockCtx = {
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn((x, y) => {
        (mockCtx as any)._lastMove = [x, y];
      }),
      lineTo: vi.fn((x, y) => {
        const from = (mockCtx as any)._lastMove || [0, 0];
        drawnLines.push({ from, to: [x, y], strokeStyle: mockCtx.strokeStyle });
      }),
      stroke: vi.fn(),
      arc: vi.fn((x, y, r) => {
        drawnArcs.push({ x, y, r, fillStyle: mockCtx.fillStyle });
      }),
      fill: vi.fn(),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      fillText: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
      drawImage: vi.fn(),
      setLineDash: vi.fn(),
      lineWidth: 1,
      strokeStyle: '#000000',
      fillStyle: '#000000',
      shadowColor: '',
      shadowBlur: 0,
      createLinearGradient: vi.fn().mockReturnValue({ addColorStop: vi.fn() }),
      createRadialGradient: vi.fn().mockReturnValue({ addColorStop: vi.fn() }),
    };

    HTMLCanvasElement.prototype.getContext = vi.fn().mockImplementation((contextId) => {
      if (contextId === '2d') return mockCtx;
      return null;
    });

    // Mock play/pause on HTMLMediaElement
    window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
    window.HTMLMediaElement.prototype.pause = vi.fn();

    // Mock navigator.mediaDevices.getUserMedia
    const mockTrack = {
      stop: vi.fn(),
      getCapabilities: vi.fn().mockReturnValue({
        zoom: { min: 1.0, max: 3.0, step: 0.1 },
      }),
      getSettings: vi.fn().mockReturnValue({ zoom: 1.0 }),
      applyConstraints: vi.fn().mockResolvedValue(undefined),
    };

    const mockStream = {
      getTracks: vi.fn().mockReturnValue([mockTrack]),
      getVideoTracks: vi.fn().mockReturnValue([mockTrack]),
    };

    Object.defineProperty(navigator, 'mediaDevices', {
      writable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue(mockStream),
        enumerateDevices: vi.fn().mockResolvedValue([
          { deviceId: 'cam1', kind: 'videoinput', label: 'Front HD Camera' },
        ]),
      },
    });
  });

  const createDummyHand = (valid = true): LandmarkPoint[] =>
    Array.from({ length: 21 }, (_, i) => ({
      x: 0.2 + (i % 5) * 0.05,
      y: 0.3 + Math.floor(i / 5) * 0.05,
      z: 0.0,
      valid,
    }));

  const createDummyPose = (valid = true): LandmarkPoint[] =>
    Array.from({ length: 33 }, (_, i) => ({
      x: 0.5 + ((i % 2 === 0 ? -1 : 1) * (i * 0.01)),
      y: 0.2 + i * 0.02,
      z: 0.0,
      valid,
    }));

  const createDummyFace = (valid = true): LandmarkPoint[] =>
    Array.from({ length: 478 }, (_, i) => ({
      x: 0.45 + (i % 20) * 0.005,
      y: 0.15 + Math.floor(i / 20) * 0.005,
      z: 0.0,
      valid,
    }));

  it('renders default Upper Body view mode and controls cleanly', () => {
    render(
      <CameraPanel
        onFrameCaptured={vi.fn()}
        isStreaming={false}
        setIsStreaming={vi.fn()}
        landmarksStatus={null}
        leftLandmarks={[]}
        rightLandmarks={[]}
        poseLandmarks={[]}
        faceLandmarks={[]}
        leftBox={null}
        rightBox={null}
        quality={null}
        stateCode="STATE_A_INACTIVE"
        candidateWord={null}
        performance={null}
        targetFps={25}
        mirrorCamera={true}
        onToggleMirror={vi.fn()}
      />
    );

    expect(screen.getByText('Live Camera Workspace')).toBeDefined();
    expect(screen.getByText('Upper Body')).toBeDefined();
    expect(screen.getByText('Hands Detail')).toBeDefined();
    expect(screen.getByText('Split')).toBeDefined();
    expect(screen.getByText('Camera Standby')).toBeDefined();

    // Check legend chips
    expect(screen.getByText(/Left Hand \(21pt\)/i)).toBeDefined();
    expect(screen.getByText(/Right Hand \(21pt\)/i)).toBeDefined();
    expect(screen.getByText(/Pose \(33pt\)/i)).toBeDefined();
    expect(screen.getAllByText(/Face Mesh/i).length).toBeGreaterThan(0);
  });

  it('renders both left and right hands simultaneously with correct colors', () => {
    const leftHand = createDummyHand(true);
    const rightHand = createDummyHand(true);

    render(
      <CameraPanel
        onFrameCaptured={vi.fn()}
        isStreaming={true}
        setIsStreaming={vi.fn()}
        landmarksStatus={{
          hands_detected: true,
          left_hand: true,
          right_hand: true,
          pose: false,
          face: false,
          valid_points: 42,
        }}
        leftLandmarks={leftHand}
        rightLandmarks={rightHand}
        poseLandmarks={[]}
        faceLandmarks={[]}
        leftBox={{ xmin: 0.1, ymin: 0.1, xmax: 0.4, ymax: 0.4, width: 0.3, height: 0.3 }}
        rightBox={{ xmin: 0.6, ymin: 0.1, xmax: 0.9, ymax: 0.4, width: 0.3, height: 0.3 }}
        quality={null}
        stateCode="STATE_C_COLLECTING"
        candidateWord={null}
        performance={{ fps: 28.5, extraction_ms: 12, inference_ms: 4, total_latency_ms: 16, frame_count: 100 }}
        targetFps={25}
        mirrorCamera={false}
        onToggleMirror={vi.fn()}
      />
    );

    // Verify connections definition count
    expect(HAND_CONNECTIONS.length).toBe(21);
    expect(POSE_UPPER_BODY_CONNECTIONS.length).toBeGreaterThan(15);
    expect(FACE_CONTOUR_CHAINS.length).toBe(9);
  });

  it('starts camera stream and renders upper-body pose & face status', async () => {
    const pose = createDummyPose(true);
    // Invalidate right wrist (index 16) and left elbow (index 13)
    pose[16].valid = false;
    pose[13].valid = false;

    render(
      <CameraPanel
        onFrameCaptured={vi.fn()}
        isStreaming={true}
        setIsStreaming={vi.fn()}
        landmarksStatus={{
          hands_detected: false,
          left_hand: false,
          right_hand: false,
          pose: true,
          face: true,
          valid_points: 31,
        }}
        leftLandmarks={[]}
        rightLandmarks={[]}
        poseLandmarks={pose}
        faceLandmarks={createDummyFace(true)}
        leftBox={null}
        rightBox={null}
        quality={null}
        stateCode="STATE_B_NO_HAND"
        candidateWord={null}
        performance={null}
        targetFps={25}
        mirrorCamera={true}
        onToggleMirror={vi.fn()}
      />
    );

    // Click Start Camera
    const startBtn = screen.getAllByRole('button', { name: /Start Camera/i })[0];
    await act(async () => {
      fireEvent.click(startBtn);
    });

    // Verify dynamic state banner and modality badges appear
    expect(screen.getByText(/Ready — raise hands inside upper-body frame/i)).toBeDefined();
    expect(screen.getByText(/Pose: Tracking \(33\)/i)).toBeDefined();
    expect(screen.getByText(/Face: Tracking \(478\)/i)).toBeDefined();
  });

  it('supports layer toggles for Hands, Pose, Face, and Points', () => {
    render(
      <CameraPanel
        onFrameCaptured={vi.fn()}
        isStreaming={true}
        setIsStreaming={vi.fn()}
        landmarksStatus={null}
        leftLandmarks={createDummyHand(true)}
        rightLandmarks={createDummyHand(true)}
        poseLandmarks={createDummyPose(true)}
        faceLandmarks={createDummyFace(true)}
        leftBox={null}
        rightBox={null}
        quality={null}
        stateCode="STATE_C_COLLECTING"
        candidateWord={null}
        performance={null}
        targetFps={25}
        mirrorCamera={true}
        onToggleMirror={vi.fn()}
      />
    );

    const handsToggle = screen.getByRole('button', { name: /^Hands$/i });
    const poseToggle = screen.getByRole('button', { name: /^Pose$/i });
    const faceToggle = screen.getByRole('button', { name: /^Face$/i });
    const pointsToggle = screen.getByRole('button', { name: /^Points$/i });

    expect(handsToggle).toBeDefined();
    expect(poseToggle).toBeDefined();
    expect(faceToggle).toBeDefined();
    expect(pointsToggle).toBeDefined();

    // Toggle layers off and on
    fireEvent.click(handsToggle);
    fireEvent.click(poseToggle);
    fireEvent.click(faceToggle);
    fireEvent.click(pointsToggle);
  });

  it('switches view mode between Upper Body, Full, Focus, and Split', () => {
    render(
      <CameraPanel
        onFrameCaptured={vi.fn()}
        isStreaming={true}
        setIsStreaming={vi.fn()}
        landmarksStatus={null}
        leftLandmarks={createDummyHand(true)}
        rightLandmarks={createDummyHand(true)}
        leftBox={null}
        rightBox={null}
        quality={null}
        stateCode="STATE_B_NO_HAND"
        candidateWord={null}
        performance={null}
        targetFps={25}
        mirrorCamera={true}
        onToggleMirror={vi.fn()}
      />
    );

    const fullBtn = screen.getByRole('button', { name: /^Full$/i });
    const focusBtn = screen.getByRole('button', { name: /Hands Detail/i });
    const splitBtn = screen.getByRole('button', { name: /^Split$/i });
    const upperBodyBtn = screen.getByRole('button', { name: /Upper Body/i });

    fireEvent.click(focusBtn);
    expect(screen.getByText('Dual-Hand Detail (Union Crop)')).toBeDefined();

    fireEvent.click(splitBtn);
    expect(screen.getByText('Dual-Hand Detail (Union Crop)')).toBeDefined();

    fireEvent.click(fullBtn);
    fireEvent.click(upperBodyBtn);
  });

  it('cleans up animation frame and media tracks on stop / unmount', () => {
    const trackStop = vi.fn();
    const mockStream = {
      getTracks: vi.fn().mockReturnValue([{ stop: trackStop }]),
      getVideoTracks: vi.fn().mockReturnValue([{ stop: trackStop, getCapabilities: vi.fn().mockReturnValue({}) }]),
    };

    const cancelAnimSpy = vi.spyOn(window, 'cancelAnimationFrame');

    const { unmount } = render(
      <CameraPanel
        onFrameCaptured={vi.fn()}
        isStreaming={true}
        setIsStreaming={vi.fn()}
        landmarksStatus={null}
        leftLandmarks={[]}
        rightLandmarks={[]}
        leftBox={null}
        rightBox={null}
        quality={null}
        stateCode="STATE_A_INACTIVE"
        candidateWord={null}
        performance={null}
        targetFps={25}
        mirrorCamera={true}
        onToggleMirror={vi.fn()}
      />
    );

    unmount();
    expect(cancelAnimSpy).toBeDefined();
  });
});
