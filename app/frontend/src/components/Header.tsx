import React, { useState, useRef, useEffect } from 'react';
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  Settings,
  HelpCircle,
  Radio,
  Sliders,
  Sparkles,
  Camera,
  Layers,
  Cpu,
  ChevronDown,
  Video,
  Award,
  Zap,
  RefreshCw,
  Check
} from 'lucide-react';
import { HealthStatus, ModelInfo, RecognitionMode, AvailableModelItem } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
  modelInfo: ModelInfo | null;
  wsConnected: boolean;
  recognitionMode: RecognitionMode;
  onModeChange: (mode: RecognitionMode) => void;
  onOpenSettings: () => void;
  onOpenHelp: () => void;
  availableModels: AvailableModelItem[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  isSwitchingModel: boolean;
  videoDevices: MediaDeviceInfo[];
  selectedDeviceId: string;
  onSelectDevice: (deviceId: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  modelInfo,
  wsConnected,
  recognitionMode,
  onModeChange,
  onOpenSettings,
  onOpenHelp,
  availableModels,
  selectedModelId,
  onSelectModel,
  isSwitchingModel,
  videoDevices,
  selectedDeviceId,
  onSelectDevice,
}) => {
  const isModelReady = health?.trained_model_loaded ?? false;
  const isMediaPipeReady = health?.mediapipe_available ?? false;

  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [cameraDropdownOpen, setCameraDropdownOpen] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);
  const cameraMenuRef = useRef<HTMLDivElement | null>(null);

  // Close dropdowns on outside click
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setModelDropdownOpen(false);
      }
      if (cameraMenuRef.current && !cameraMenuRef.current.contains(e.target as Node)) {
        setCameraDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  const activeModel = availableModels.find((m) => m.id === selectedModelId) || {
    id: selectedModelId,
    name: modelInfo?.architecture || '3-Way Probability Ensemble',
    accuracy: '76.65% Top-1 / 91.44% Top-5',
    badge: 'SOTA Benchmark',
    top1: 76.65,
    description: 'Optimal probability fusion of 3 distinct architectures',
  };

  const activeCamera = videoDevices.find((d) => d.deviceId === selectedDeviceId) || videoDevices[0];

  return (
    <header className="sticky top-0 z-40 bg-slate-900/95 backdrop-blur-md border-b border-slate-800 shadow-xl px-4 sm:px-6 py-2.5 text-slate-100">
      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
        {/* Brand & Title */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-500 via-cyan-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-teal-500/20 ring-2 ring-teal-400/30">
            <Radio className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-300 bg-clip-text text-transparent">
                SignFlow
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30 flex items-center space-x-1">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-ping" />
                <span>AI SOTA</span>
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium hidden sm:block">
              Real-Time Dynamic Word-Level ASL Recognition
            </p>
          </div>
        </div>

        {/* Center: Dropdown Controls (Model Switcher & Camera Switcher) */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-2.5">
          {/* 1. Model Selection Dropdown */}
          <div className="relative" ref={modelMenuRef}>
            <button
              onClick={() => {
                setModelDropdownOpen(!modelDropdownOpen);
                setCameraDropdownOpen(false);
              }}
              disabled={isSwitchingModel}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-xl border text-xs font-medium transition-all shadow-sm ${
                modelDropdownOpen
                  ? 'bg-slate-800 border-teal-500 ring-2 ring-teal-500/20 text-white'
                  : 'bg-slate-800/80 hover:bg-slate-800 border-slate-700 text-slate-200'
              }`}
              title="Click to switch recognition model architecture"
            >
              <Cpu className="w-3.5 h-3.5 text-teal-400 shrink-0" />
              <div className="text-left flex flex-col">
                <div className="flex items-center space-x-1.5">
                  <span className="font-semibold text-slate-100 truncate max-w-[140px] sm:max-w-[180px]">
                    {activeModel.name}
                  </span>
                  <span className="px-1.5 py-0.2 rounded text-[10px] font-bold bg-teal-500/20 text-teal-300 border border-teal-500/30">
                    {activeModel.accuracy ? activeModel.accuracy.split('/')[0].trim() : '76.65%'}
                  </span>
                </div>
              </div>
              {isSwitchingModel ? (
                <RefreshCw className="w-3.5 h-3.5 text-teal-400 animate-spin" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
              )}
            </button>

            {/* Model Dropdown Menu */}
            {modelDropdownOpen && (
              <div className="absolute left-0 mt-2 w-80 sm:w-96 rounded-2xl bg-slate-900 border border-slate-700/80 shadow-2xl p-2 z-50 backdrop-blur-xl animate-in fade-in slide-in-from-top-2 duration-150">
                <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between">
                  <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-300">
                    <Sparkles className="w-3.5 h-3.5 text-teal-400" />
                    <span>Select Recognition Model</span>
                  </div>
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">
                    Protocol A (Official)
                  </span>
                </div>

                <div className="py-1.5 space-y-1">
                  {availableModels.map((m) => {
                    const isSelected = m.id === selectedModelId;
                    return (
                      <button
                        key={m.id}
                        onClick={() => {
                          onSelectModel(m.id);
                          setModelDropdownOpen(false);
                        }}
                        className={`w-full text-left p-2.5 rounded-xl transition-all flex items-start justify-between group ${
                          isSelected
                            ? 'bg-teal-500/15 border border-teal-500/40 text-white'
                            : 'hover:bg-slate-800/80 border border-transparent text-slate-300'
                        }`}
                      >
                        <div className="space-y-1">
                          <div className="flex items-center space-x-2">
                            <span className="font-semibold text-xs text-white group-hover:text-teal-300 transition-colors">
                              {m.name}
                            </span>
                            <span
                              className={`text-[9px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider ${
                                m.badge === 'SOTA Benchmark'
                                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                  : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                              }`}
                            >
                              {m.badge}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 leading-tight">
                            {m.description}
                          </p>
                          <div className="flex items-center space-x-3 text-[10px] text-teal-400/90 font-mono pt-0.5">
                            <span className="font-bold text-teal-300">Top-1: {m.top1}%</span>
                            <span className="text-slate-400">Top-5: {m.top5}%</span>
                            {m.ece != null && (
                              <span className="text-slate-400">ECE: {m.ece.toFixed(4)}</span>
                            )}
                          </div>
                        </div>

                        {isSelected && (
                          <div className="w-5 h-5 rounded-full bg-teal-500 text-slate-950 flex items-center justify-center shrink-0 ml-2 mt-0.5">
                            <Check className="w-3 h-3 stroke-[3]" />
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* 2. Video Camera Dropdown */}
          <div className="relative" ref={cameraMenuRef}>
            <button
              onClick={() => {
                setCameraDropdownOpen(!cameraDropdownOpen);
                setModelDropdownOpen(false);
              }}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-xl border text-xs font-medium transition-all shadow-sm ${
                cameraDropdownOpen
                  ? 'bg-slate-800 border-cyan-500 ring-2 ring-cyan-500/20 text-white'
                  : 'bg-slate-800/80 hover:bg-slate-800 border-slate-700 text-slate-200'
              }`}
              title="Select Video Camera Input"
            >
              <Video className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
              <span className="truncate max-w-[120px] sm:max-w-[150px]">
                {activeCamera?.label || (videoDevices.length > 0 ? `Camera 1` : 'Default Camera')}
              </span>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
            </button>

            {/* Camera Dropdown Menu */}
            {cameraDropdownOpen && (
              <div className="absolute left-0 mt-2 w-72 rounded-2xl bg-slate-900 border border-slate-700/80 shadow-2xl p-2 z-50 backdrop-blur-xl animate-in fade-in slide-in-from-top-2 duration-150">
                <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between">
                  <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-300">
                    <Camera className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Video Input Devices</span>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono">
                    {videoDevices.length} available
                  </span>
                </div>

                <div className="py-1.5 space-y-1">
                  {videoDevices.length === 0 ? (
                    <div className="p-3 text-center text-xs text-slate-400">
                      No external cameras enumerated yet. Allow camera access in browser to list devices.
                    </div>
                  ) : (
                    videoDevices.map((d, idx) => {
                      const isSelected = (d.deviceId === selectedDeviceId) || (!selectedDeviceId && idx === 0);
                      return (
                        <button
                          key={d.deviceId || idx}
                          onClick={() => {
                            onSelectDevice(d.deviceId);
                            setCameraDropdownOpen(false);
                          }}
                          className={`w-full text-left p-2.5 rounded-xl transition-all flex items-center justify-between text-xs group ${
                            isSelected
                              ? 'bg-cyan-500/15 border border-cyan-500/40 text-white'
                              : 'hover:bg-slate-800/80 border border-transparent text-slate-300'
                          }`}
                        >
                          <div className="flex items-center space-x-2 truncate">
                            <Video className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                            <span className="truncate">{d.label || `Camera ${idx + 1}`}</span>
                          </div>
                          {isSelected && (
                            <Check className="w-4 h-4 text-cyan-400 shrink-0 ml-2" />
                          )}
                        </button>
                      );
                    })
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: Mode Switcher & System Telemetry */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3 text-xs">
          {/* Mode Switcher */}
          <div className="flex items-center bg-slate-800/90 p-0.5 rounded-xl border border-slate-700/80">
            <button
              onClick={() => onModeChange('guided')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                recognitionMode === 'guided'
                  ? 'bg-teal-500 text-slate-950 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Mode A: Guided Word Accumulation (Detects pauses & commits words)"
            >
              Mode A: Guided
            </button>
            <button
              onClick={() => onModeChange('continuous')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all flex items-center space-x-1 ${
                recognitionMode === 'continuous'
                  ? 'bg-amber-400 text-slate-950 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Mode B: Continuous Sign Recognition (Sliding window)"
            >
              <span>Mode B</span>
              <span className="text-[9px] px-1 py-0.2 rounded font-mono bg-slate-900/40 text-slate-900 uppercase">
                Exp
              </span>
            </button>
          </div>

          {/* Telemetry Pills */}
          <div className="hidden sm:flex items-center space-x-2">
            {/* MediaPipe Status */}
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full border text-[11px] ${
                isMediaPipeReady
                  ? 'bg-teal-500/10 text-teal-300 border-teal-500/30'
                  : 'bg-slate-800 text-slate-500 border-slate-700'
              }`}
              title="MediaPipe Holistic 553 Landmark Tracker"
            >
              <Layers className="w-3 h-3 text-teal-400" />
              <span>Holistic 553pts</span>
            </div>

            {/* WebSocket Connection */}
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full border text-[11px] ${
                wsConnected
                  ? 'bg-indigo-500/10 text-indigo-300 border-indigo-500/30'
                  : 'bg-rose-500/10 text-rose-300 border-rose-500/30'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  wsConnected ? 'bg-indigo-400 animate-pulse' : 'bg-rose-500'
                }`}
              />
              <span>{wsConnected ? 'Live WS' : 'Offline'}</span>
            </div>
          </div>

          {/* Action Settings / Help */}
          <div className="flex items-center space-x-1">
            <button
              onClick={onOpenSettings}
              className="p-1.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Audio & Sensitivity Settings"
            >
              <Settings className="w-4 h-4" />
            </button>
            <button
              onClick={onOpenHelp}
              className="p-1.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Benchmark & Help Documentation"
            >
              <HelpCircle className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
