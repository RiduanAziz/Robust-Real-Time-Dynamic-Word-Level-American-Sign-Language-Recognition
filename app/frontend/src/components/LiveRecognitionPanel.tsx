import React from 'react';
import {
  Sparkles,
  Check,
  X,
  Clock,
  Layers,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  ShieldAlert,
  HelpCircle,
} from 'lucide-react';
import { LivePrediction, WordCandidate, RecognitionMode, RecognitionStateCode } from '../types';

interface LiveRecognitionPanelProps {
  prediction: LivePrediction | null;
  candidate: WordCandidate | null;
  mode: RecognitionMode;
  stateCode: RecognitionStateCode;
  onConfirmWord: () => void;
  onRejectCandidate: () => void;
  confidenceThreshold: number;
  lastCommittedWord?: string | null;
}

export const LiveRecognitionPanel: React.FC<LiveRecognitionPanelProps> = ({
  prediction,
  candidate,
  mode,
  stateCode,
  onConfirmWord,
  onRejectCandidate,
  confidenceThreshold,
  lastCommittedWord,
}) => {
  const activeWord = candidate?.word || (prediction?.meets_threshold ? prediction.predicted_label : null);
  const confidence = candidate?.confidence ?? prediction?.confidence ?? 0;
  const confidencePct = Math.round(confidence * 100);

  const stateConfigs: Record<
    RecognitionStateCode,
    { label: string; bg: string; text: string; dot: string }
  > = {
    STATE_A_INACTIVE: { label: 'Camera Off', bg: 'bg-slate-800/80 border border-slate-700/60', text: 'text-slate-400', dot: 'bg-slate-500' },
    STATE_B_NO_HAND: { label: 'No Hand Detected', bg: 'bg-slate-800/80 border border-slate-700/60', text: 'text-slate-400', dot: 'bg-slate-500' },
    STATE_C_COLLECTING: { label: 'Collecting Motion', bg: 'bg-teal-500/15 border border-teal-500/30', text: 'text-teal-300', dot: 'bg-teal-400 animate-pulse' },
    STATE_D_CANDIDATE: { label: 'Candidate Active', bg: 'bg-indigo-500/20 border border-indigo-500/40', text: 'text-indigo-200', dot: 'bg-indigo-400 animate-ping' },
    STATE_E_COMMITTED: { label: 'Word Committed', bg: 'bg-emerald-500/20 border border-emerald-500/40', text: 'text-emerald-300', dot: 'bg-emerald-400' },
    STATE_F_UNCERTAIN: { label: 'Sign Uncertain', bg: 'bg-amber-500/15 border border-amber-500/30', text: 'text-amber-300', dot: 'bg-amber-400' },
    STATE_G_MODEL_UNAVAILABLE: { label: 'Model Missing', bg: 'bg-rose-500/15 border border-rose-500/30', text: 'text-rose-300', dot: 'bg-rose-400' },
  };

  const currentCfg = stateConfigs[stateCode] || stateConfigs.STATE_B_NO_HAND;

  return (
    <div className="bg-slate-900/90 backdrop-blur-md rounded-2xl border border-slate-800 shadow-2xl p-4 sm:p-5 flex flex-col justify-between space-y-4 text-slate-100">
      {/* Header with Recognition State */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm sm:text-base font-bold text-white tracking-tight">Live Recognition</h2>
            <p className="text-[11px] text-slate-400">Continuous sequence classification</p>
          </div>
        </div>

        {/* State Badge */}
        <div
          className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${currentCfg.bg} ${currentCfg.text}`}
        >
          <span className={`w-2 h-2 rounded-full ${currentCfg.dot}`} />
          <span>{currentCfg.label}</span>
        </div>
      </div>

      {/* Main Status & Candidate Card */}
      <div className="bg-slate-950/80 rounded-xl border border-slate-800/80 p-5 flex flex-col items-center justify-center text-center space-y-3 relative overflow-hidden shadow-inner min-h-[185px]">
        {/* Ambient background glow for active candidates */}
        {activeWord && (
          <div className="absolute inset-0 bg-radial from-teal-500/10 via-transparent to-transparent pointer-events-none" />
        )}

        {/* State G: Model Unavailable */}
        {stateCode === 'STATE_G_MODEL_UNAVAILABLE' ? (
          <div className="flex flex-col items-center justify-center text-rose-400 space-y-2 py-2">
            <ShieldAlert className="w-9 h-9 text-rose-500 animate-bounce" />
            <p className="font-bold text-sm text-white">Recognition Model Standby</p>
            <p className="text-xs text-rose-300 max-w-xs">
              PyTorch model checkpoint not detected. Please verify checkpoint path in models directory.
            </p>
          </div>
        ) : stateCode === 'STATE_F_UNCERTAIN' ? (
          /* State F: Uncertain Sign */
          <div className="flex flex-col items-center justify-center text-amber-300 space-y-2 py-2">
            <AlertCircle className="w-9 h-9 text-amber-400 animate-pulse" />
            <p className="font-bold text-sm text-white">Sign Uncertain</p>
            <p className="text-xs text-slate-400 max-w-xs">
              Hand motion did not exceed confidence threshold ({Math.round(confidenceThreshold * 100)}%). Repeat gesture clearly.
            </p>
          </div>
        ) : activeWord ? (
          /* State D or E: Candidate or Confirmed */
          <>
            <div className="text-[10px] uppercase font-mono tracking-widest text-teal-400/80">
              {stateCode === 'STATE_E_COMMITTED' ? 'RECOGNIZED SIGN (COMMITTED)' : 'PROVISIONAL CANDIDATE'}
            </div>
            <div
              className={`text-3xl sm:text-4xl font-black tracking-tight capitalize ${
                stateCode === 'STATE_E_COMMITTED'
                  ? 'text-emerald-400'
                  : 'bg-gradient-to-r from-teal-200 via-white to-cyan-200 bg-clip-text text-transparent'
              }`}
            >
              {activeWord}
            </div>

            {/* Confidence & Stability Bar */}
            <div className="w-full max-w-xs space-y-1.5 pt-1">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-slate-400">Probability Confidence</span>
                <span
                  className={
                    confidencePct >= 70
                      ? 'text-emerald-400 font-bold'
                      : confidencePct >= 45
                      ? 'text-amber-400 font-bold'
                      : 'text-slate-400'
                  }
                >
                  {confidencePct}%
                </span>
              </div>

              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
                <div
                  className={`h-full rounded-full transition-all duration-200 shadow-sm ${
                    confidencePct >= 70
                      ? 'bg-gradient-to-r from-teal-400 to-emerald-400 shadow-emerald-500/50'
                      : confidencePct >= 45
                      ? 'bg-gradient-to-r from-amber-400 to-orange-400'
                      : 'bg-slate-500'
                  }`}
                  style={{ width: `${Math.min(100, confidencePct)}%` }}
                />
              </div>

              {mode === 'guided' && candidate && (
                <div className="flex items-center justify-between text-[11px] text-slate-400 pt-0.5 font-mono">
                  <span>Stability frames</span>
                  <span className="text-teal-400">
                    {candidate.stability_count} / {candidate.target_count}
                  </span>
                </div>
              )}
            </div>
          </>
        ) : (
          /* State B or C: Waiting or Collecting */
          <div className="flex flex-col items-center justify-center text-slate-400 space-y-2 py-4">
            <Clock className="w-8 h-8 text-slate-600 stroke-1" />
            <p className="text-sm font-medium text-slate-300">
              {stateCode === 'STATE_C_COLLECTING'
                ? 'Collecting sign motion frames...'
                : 'Waiting for sign gesture...'}
            </p>
            <p className="text-xs text-slate-500 max-w-xs">
              Perform dynamic sign in front of camera. System tracks 553 landmarks and classifies sign.
            </p>
          </div>
        )}
      </div>

      {/* Manual Actions: Confirm Word / Reject */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={onConfirmWord}
          disabled={!activeWord}
          className={`flex items-center justify-center space-x-1.5 py-2.5 px-3 rounded-xl text-xs font-bold transition-all ${
            activeWord
              ? 'bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-lg shadow-emerald-500/20 active:scale-95 cursor-pointer'
              : 'bg-slate-800 text-slate-600 border border-slate-800 cursor-not-allowed'
          }`}
          title="Manually confirm and commit word to transcript"
        >
          <Check className="w-4 h-4 stroke-[2.5]" />
          <span>Confirm Word</span>
        </button>

        <button
          onClick={onRejectCandidate}
          disabled={!activeWord}
          className={`flex items-center justify-center space-x-1.5 py-2.5 px-3 rounded-xl border text-xs font-semibold transition-all ${
            activeWord
              ? 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700 active:scale-95 cursor-pointer'
              : 'bg-slate-800/40 text-slate-600 border-slate-800 cursor-not-allowed'
          }`}
          title="Reject candidate and reset gesture buffer"
        >
          <X className="w-4 h-4" />
          <span>Reject / Reset</span>
        </button>
      </div>

      {/* Probability Distribution */}
      <div className="space-y-2 pt-2 border-t border-slate-800/80">
        <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
          <span className="flex items-center space-x-1.5">
            <TrendingUp className="w-3.5 h-3.5 text-teal-400" />
            <span className="text-slate-300">Top Candidates Distribution</span>
          </span>
          <span className="text-[10px] text-slate-500 font-mono uppercase">Softmax</span>
        </div>

        {prediction?.top_k && prediction.top_k.length > 0 ? (
          <div className="space-y-1.5">
            {prediction.top_k.slice(0, 4).map((cand, idx) => {
              const pct = Math.round(cand.confidence * 100);
              return (
                <div key={idx} className="flex items-center text-xs space-x-2">
                  <span className="w-24 truncate font-medium text-slate-300 capitalize">
                    {cand.label}
                  </span>
                  <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        idx === 0
                          ? 'bg-teal-400 shadow-xs shadow-teal-400/50'
                          : 'bg-slate-600'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-8 text-right font-mono text-[11px] text-slate-400">
                    {pct}%
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-2 text-xs text-slate-500 italic">
            No active candidates
          </div>
        )}
      </div>

      {/* Mode Advisory Notice */}
      <div
        className={`p-3 rounded-xl text-xs flex items-start space-x-2.5 border ${
          mode === 'guided'
            ? 'bg-slate-950/60 border-slate-800 text-slate-400'
            : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
        }`}
      >
        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-teal-400" />
        <div className="text-[11px] leading-relaxed">
          {mode === 'guided' ? (
            <>
              <strong className="text-slate-200">Mode A Guided:</strong> Sign one word at a time, then pause. Confirmed words are safely committed to your transcript.
            </>
          ) : (
            <>
              <strong className="text-amber-200">Mode B Continuous (Exp):</strong> Heuristic temporal sliding window segmentation.
            </>
          )}
        </div>
      </div>
    </div>
  );
};
