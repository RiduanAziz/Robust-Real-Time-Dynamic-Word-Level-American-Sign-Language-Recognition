import { describe, it, expect } from 'vitest';
import {
  getDefaultWebSocketUrl,
  getBaseApiUrl,
  resolveEffectiveSettings,
  LocationLike,
} from './url';
import { AppSettings } from '../types';

const BASE_TEST_SETTINGS: AppSettings = {
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
  wsUrl: 'ws://127.0.0.1:8000/ws/live',
  isCustomWsUrl: false,
};

describe('WebSocket & API URL Resolution', () => {
  it('resolves local Vite development URL (:5173)', () => {
    const loc: LocationLike = {
      protocol: 'http:',
      host: '127.0.0.1:5173',
      hostname: '127.0.0.1',
      origin: 'http://127.0.0.1:5173',
    };
    expect(getDefaultWebSocketUrl(loc)).toBe('ws://127.0.0.1:8000/ws/live');
    expect(getBaseApiUrl(loc)).toBe('http://127.0.0.1:8000');
  });

  it('resolves deployed HTTPS origin with wss:// without hardcoded :8000 port', () => {
    const loc: LocationLike = {
      protocol: 'https:',
      host: 'signflow-demo.onrender.com',
      hostname: 'signflow-demo.onrender.com',
      origin: 'https://signflow-demo.onrender.com',
    };
    expect(getDefaultWebSocketUrl(loc)).toBe('wss://signflow-demo.onrender.com/ws/live');
    expect(getBaseApiUrl(loc)).toBe('https://signflow-demo.onrender.com');
  });

  it('resolves local production HTTP with custom port', () => {
    const loc: LocationLike = {
      protocol: 'http:',
      host: '192.168.1.100:8080',
      hostname: '192.168.1.100',
      origin: 'http://192.168.1.100:8080',
    };
    expect(getDefaultWebSocketUrl(loc)).toBe('ws://192.168.1.100:8080/ws/live');
    expect(getBaseApiUrl(loc)).toBe('http://192.168.1.100:8080');
  });

  it('migrates stale localStorage localhost:8000 URL on a deployed HTTPS origin', () => {
    const loc: LocationLike = {
      protocol: 'https:',
      host: 'signflow.example.com',
      hostname: 'signflow.example.com',
      origin: 'https://signflow.example.com',
    };
    // Saved settings from old dev session
    const staleJson = JSON.stringify({
      ...BASE_TEST_SETTINGS,
      wsUrl: 'ws://127.0.0.1:8000/ws/live',
      isCustomWsUrl: true, // Marked custom previously by default
    });

    const resolved = resolveEffectiveSettings(staleJson, BASE_TEST_SETTINGS, loc);
    // Should be migrated to same-origin wss:// because HTTPS page cannot reach unencrypted localhost
    expect(resolved.wsUrl).toBe('wss://signflow.example.com/ws/live');
    expect(resolved.isCustomWsUrl).toBe(false);
  });

  it('preserves valid custom external WebSocket URL if user explicitly configured it', () => {
    const loc: LocationLike = {
      protocol: 'https:',
      host: 'signflow.example.com',
      hostname: 'signflow.example.com',
      origin: 'https://signflow.example.com',
    };
    const customJson = JSON.stringify({
      ...BASE_TEST_SETTINGS,
      wsUrl: 'wss://api.custom-server.com/ws/live',
      isCustomWsUrl: true,
    });

    const resolved = resolveEffectiveSettings(customJson, BASE_TEST_SETTINGS, loc);
    expect(resolved.wsUrl).toBe('wss://api.custom-server.com/ws/live');
    expect(resolved.isCustomWsUrl).toBe(true);
  });

  it('migrates stale settings without isCustomWsUrl flag', () => {
    const loc: LocationLike = {
      protocol: 'https:',
      host: 'my-signflow.onrender.com',
      hostname: 'my-signflow.onrender.com',
      origin: 'https://my-signflow.onrender.com',
    };
    // Legacy settings payload before isCustomWsUrl existed
    const legacyJson = JSON.stringify({
      autoSpeak: true,
      confidenceThreshold: 0.35,
      wsUrl: 'ws://127.0.0.1:8000/ws/live',
    });

    const resolved = resolveEffectiveSettings(legacyJson, BASE_TEST_SETTINGS, loc);
    expect(resolved.wsUrl).toBe('wss://my-signflow.onrender.com/ws/live');
    expect(resolved.autoSpeak).toBe(true);
    expect(resolved.confidenceThreshold).toBe(0.35);
  });
});

