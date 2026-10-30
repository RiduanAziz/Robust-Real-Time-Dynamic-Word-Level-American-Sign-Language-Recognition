import { AppSettings } from '../types';

export interface LocationLike {
  protocol: string;
  host: string;
  hostname: string;
  origin: string;
}

export function getCurrentLocation(): LocationLike {
  if (typeof window !== 'undefined' && window.location) {
    return {
      protocol: window.location.protocol,
      host: window.location.host,
      hostname: window.location.hostname,
      origin: window.location.origin,
    };
  }
  return {
    protocol: 'http:',
    host: '127.0.0.1:8000',
    hostname: '127.0.0.1',
    origin: 'http://127.0.0.1:8000',
  };
}

/**
 * Derive default WebSocket URL based on current page location.
 * - Vite dev server (:5173) -> ws://127.0.0.1:8000/ws/live
 * - HTTPS deployment (e.g. Render, custom domain) -> wss://<host>/ws/live
 * - HTTP deployment (e.g. LAN, localhost:8000) -> ws://<host>/ws/live
 */
export function getDefaultWebSocketUrl(loc: LocationLike = getCurrentLocation()): string {
  if (loc.host.includes(':5173') || loc.origin.includes(':5173')) {
    return 'ws://127.0.0.1:8000/ws/live';
  }
  const wsProto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsProto}//${loc.host}/ws/live`;
}

/**
 * Derive base API REST URL based on current page location.
 * - Vite dev server (:5173) -> http://127.0.0.1:8000
 * - Production / Same-origin -> window.location.origin
 */
export function getBaseApiUrl(loc: LocationLike = getCurrentLocation()): string {
  if (loc.host.includes(':5173') || loc.origin.includes(':5173')) {
    return 'http://127.0.0.1:8000';
  }
  return loc.origin || 'http://127.0.0.1:8000';
}

/**
 * Safely parse settings from localStorage and migrate stale WebSocket endpoints:
 * - Migrates hard-coded :8000 endpoints when running on public HTTPS hosts.
 * - Respects explicit user overrides when isCustomWsUrl is true, unless blocked by Mixed Content.
 */
export function resolveEffectiveSettings(
  savedRaw: string | null,
  defaultSettings: AppSettings,
  loc: LocationLike = getCurrentLocation()
): AppSettings {
  const dynamicDefaultWs = getDefaultWebSocketUrl(loc);
  if (!savedRaw) {
    return {
      ...defaultSettings,
      wsUrl: dynamicDefaultWs,
      isCustomWsUrl: false,
    };
  }

  try {
    const parsed = JSON.parse(savedRaw);
    let wsUrl = parsed.wsUrl;
    let isCustom = Boolean(parsed.isCustomWsUrl);

    // Detect stale or invalid configurations
    if (!isCustom) {
      wsUrl = dynamicDefaultWs;
    } else if (typeof wsUrl === 'string') {
      const isHttpsPage = loc.protocol === 'https:';
      const isLocalWs = wsUrl.includes('127.0.0.1') || wsUrl.includes('localhost');
      // HTTPS browsers will reject unencrypted ws:// or 127.0.0.1 from remote clients
      if (isHttpsPage && isLocalWs) {
        wsUrl = dynamicDefaultWs;
        isCustom = false;
      }
    } else {
      wsUrl = dynamicDefaultWs;
      isCustom = false;
    }

    return {
      ...defaultSettings,
      ...parsed,
      wsUrl: wsUrl || dynamicDefaultWs,
      isCustomWsUrl: isCustom,
    };
  } catch {
    return {
      ...defaultSettings,
      wsUrl: dynamicDefaultWs,
      isCustomWsUrl: false,
    };
  }
}

