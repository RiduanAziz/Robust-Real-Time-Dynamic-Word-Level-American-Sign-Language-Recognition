import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { LiveRecognitionPanel } from './components/LiveRecognitionPanel';
import { TranscriptWorkspace } from './components/TranscriptWorkspace';
import { SpeechWorkspace } from './components/SpeechWorkspace';
import { SettingsModal } from './components/SettingsModal';
import { Header } from './components/Header';
import { CommittedToken, LivePrediction, WordCandidate, AppSettings } from './types';

describe('SignFlow Component Suite', () => {
  beforeEach(() => {
    class MockUtterance {
      text: string;
      rate = 1;
      pitch = 1;
      voice: any = null;
      onstart: any = null;
      onend: any = null;
      onerror: any = null;
      constructor(text: string) {
        this.text = text;
      }
    }
    (globalThis as any).SpeechSynthesisUtterance = MockUtterance;
    (window as any).SpeechSynthesisUtterance = MockUtterance;

    // Mock Web Speech API for jsdom
    Object.defineProperty(window, 'speechSynthesis', {
      writable: true,
      value: {
        speak: vi.fn(),
        cancel: vi.fn(),
        pause: vi.fn(),
        resume: vi.fn(),
        getVoices: vi.fn(() => [
          { name: 'Alex', lang: 'en-US' } as SpeechSynthesisVoice,
        ]),
        onvoiceschanged: null,
      },
    });
  });

  it('renders Header with correct status badges', () => {
    render(
      <Header
        health={{
          status: 'ok',
          process_alive: true,
          backend_initialized: true,
          mediapipe_available: true,
          trained_model_loaded: true,
          inference_ready: true,
          checkpoint_path: 'models/temporal_transformer_trained.pt',
          error: null,
          mediapipe_error: null,
        }}
        modelInfo={{
          name: 'temporal_transformer',
          architecture: 'TemporalTransformer',
          status: 'ready',
          checkpoint_path: 'models/temporal_transformer_trained.pt',
          num_classes: 2000,
          input_dim: 4977,
          sequence_length: 64,
          device: 'cpu',
          feature_representation: 'holistic',
          class_names_sample: ['hello', 'help'],
          total_classes: 2000,
          error: null,
        }}
        wsConnected={true}
        recognitionMode="guided"
        onModeChange={vi.fn()}
        onOpenSettings={vi.fn()}
        onOpenHelp={vi.fn()}
      />
    );

    expect(screen.getByText('SignFlow')).toBeDefined();
    expect(screen.getByText('Mode A: Guided')).toBeDefined();
    expect(screen.getByText('Holistic 553pts')).toBeDefined();
    expect(screen.getByText('Live WS')).toBeDefined();
  });

  it('renders LiveRecognitionPanel with candidate and responds to confirm/reject actions', () => {
    const handleConfirm = vi.fn();
    const handleReject = vi.fn();

    const candidate: WordCandidate = {
      word: 'HELP',
      confidence: 0.88,
      stability_count: 5,
      target_count: 5,
      progress: 1.0,
    };

    const prediction: LivePrediction = {
      predicted_label: 'HELP',
      confidence: 0.88,
      meets_threshold: true,
      top_k: [
        { class_index: 0, label: 'HELP', confidence: 0.88 },
        { class_index: 1, label: 'PLEASE', confidence: 0.08 },
      ],
      latency_ms: 18.5,
      state: 'CANDIDATE',
    };

    render(
      <LiveRecognitionPanel
        prediction={prediction}
        candidate={candidate}
        mode="guided"
        onConfirmWord={handleConfirm}
        onRejectCandidate={handleReject}
        confidenceThreshold={0.4}
      />
    );

    const helpElements = screen.getAllByText(/HELP/i);
    expect(helpElements.length).toBeGreaterThan(0);
    expect(screen.getAllByText('88%').length).toBeGreaterThan(0);

    const confirmBtn = screen.getByRole('button', { name: /confirm word/i });
    fireEvent.click(confirmBtn);
    expect(handleConfirm).toHaveBeenCalledTimes(1);

    const rejectBtn = screen.getByRole('button', { name: /reject \/ clear/i });
    fireEvent.click(rejectBtn);
    expect(handleReject).toHaveBeenCalledTimes(1);
  });

  it('renders TranscriptWorkspace with tokens, undo, and removal', () => {
    const handleRemove = vi.fn();
    const handleClear = vi.fn();
    const handleUndo = vi.fn();
    const handleSetText = vi.fn();
    const handleNewSession = vi.fn();

    const tokens: CommittedToken[] = [
      { id: '1', word: 'HELLO', confidence: 0.9, timestamp: 100, mode: 'guided' },
      { id: '2', word: 'WORLD', confidence: 0.85, timestamp: 200, mode: 'guided' },
    ];

    render(
      <TranscriptWorkspace
        tokens={tokens}
        onRemoveToken={handleRemove}
        onClearTranscript={handleClear}
        onUndo={handleUndo}
        transcriptText="HELLO WORLD"
        setTranscriptText={handleSetText}
        onNewSession={handleNewSession}
      />
    );

    expect(screen.getByText('HELLO')).toBeDefined();
    expect(screen.getByText('WORLD')).toBeDefined();

    const undoBtn = screen.getByRole('button', { name: /undo/i });
    fireEvent.click(undoBtn);
    expect(handleUndo).toHaveBeenCalledTimes(1);

    const deleteBtns = screen.getAllByTitle('Delete word');
    expect(deleteBtns.length).toBe(2);
    fireEvent.click(deleteBtns[0]);
    expect(handleRemove).toHaveBeenCalledWith('1');
  });

  it('renders SpeechWorkspace controls with Web Speech API', () => {
    render(
      <SpeechWorkspace
        textToSpeak="HELLO WORLD"
        autoSpeak={false}
        rate={1.0}
        pitch={1.0}
        voiceName=""
        onRateChange={vi.fn()}
        onPitchChange={vi.fn()}
        onVoiceChange={vi.fn()}
      />
    );

    expect(screen.getByText('Text-to-Speech (TTS)')).toBeDefined();
    const speakBtn = screen.getByRole('button', { name: /speak transcript/i });
    expect(speakBtn).toBeDefined();
    fireEvent.click(speakBtn);
  });

  it('renders SettingsModal and handles updates', () => {
    const handleUpdate = vi.fn();
    const settings: AppSettings = {
      confidenceThreshold: 0.45,
      debounceFrames: 5,
      pauseThresholdSec: 0.6,
      targetFps: 12,
      autoSpeak: false,
      ttsRate: 1.0,
      ttsPitch: 1.0,
      ttsVoice: '',
      mirrorCamera: true,
      showLandmarkGuide: true,
      wsUrl: 'ws://127.0.0.1:8000/ws/live',
    };

    render(
      <SettingsModal
        isOpen={true}
        onClose={vi.fn()}
        settings={settings}
        onUpdateSettings={handleUpdate}
      />
    );

    expect(screen.getByText('Recognition & App Settings')).toBeDefined();
    expect(screen.getByText('45%')).toBeDefined();
    expect(screen.getByText('5 frames')).toBeDefined();
  });
});
