import React from 'react';
import {
  Sparkles,
  Check,
  X,
  Clock,
  Layers,
  ChevronRight,
  TrendingUp,
  AlertCircle,
  HelpCircle,
  CheckCircle2
} from 'lucide-react';
import { LivePrediction, WordCandidate, RecognitionMode } from '../types';

interface LiveRecognitionPanelProps {
  prediction: LivePrediction | null;
  candidate: WordCandidate | null;
  mode: RecognitionMode;
  onConfirmWord: () => void;
  onRejectCandidate: () => void;
  confidenceThreshold: number;
}

export const LiveRecognitionPanel: React.FC<LiveRecognitionPanelProps> = ({
  prediction,
  candidate,
  mode,
  onConfirmWord,
  onRejectCandidate,
  confidenceThreshold,
}) => {
  const activeWord = candidate?.word || (prediction?.meets_threshold ? prediction.predicted_label : null);
  const confidence = candidate?.confidence ?? prediction?.confidence ?? 0;
  const confidencePct = Math.round(confidence * 100);
  const meetsThreshold = confidence >= confidenceThreshold;

  // Determine state label & style
  const currentState = prediction?.state || (activeWord ? 'CANDIDATE' : 'WAITING');

  const stateColors: Record<string, { bg: string; text: string; dot: string }> = {
    WAITING: { bg: 'bg-slate-100', text: 'text-slate-600', dot: 'bg-slate-400' },
    CAPTURING: { bg: 'bg-indigo-50', text: 'text-indigo-700', dot: 'bg-indigo-500 animate-pulse' },
    CANDIDATE: { bg: 'bg-amber-50', text: 'text-amber-700', dot: 'bg-amber-500' },
    COMMITTED: { bg: 'bg-emerald-50', text: 'text-emerald-700', dot: 'bg-emerald-500' },
  };

  const stateStyle = stateColors[currentState] || stateColors.WAITING;

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5 flex flex-col justify-between space-y-4">
      {/* Header with Boundary State */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-5 h-5 text-indigo-600" />
          <h2 className="text-base font-semibold text-slate-800">Live Recognition</h2>
        </div>

        {/* State Badge */}
        <div
          className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${stateStyle.bg} ${stateStyle.text}`}
        >
          <span className={`w-2 h-2 rounded-full ${stateStyle.dot}`} />
          <span>{currentState}</span>
        </div>
      </div>

      {/* Main Candidate Card */}
      <div className="bg-gradient-to-b from-slate-50 to-white rounded-xl border border-slate-200/90 p-5 flex flex-col items-center justify-center text-center space-y-3 relative overflow-hidden shadow-xs min-h-[170px]">
        {activeWord ? (
          <>
            <div className="text-xs uppercase font-mono tracking-widest text-slate-400">
              Provisional Candidate
            </div>
            <div className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 capitalize">
              {activeWord}
            </div>

            {/* Confidence & Stability Meter */}
            <div className="w-full max-w-xs space-y-1.5 pt-1">
              <div className="flex justify-between items-center text-xs font-medium">
                <span className="text-slate-500">Confidence</span>
                <span
                  className={
                    confidencePct >= 70
                      ? 'text-emerald-600 font-bold'
                      : confidencePct >= 45
                      ? 'text-amber-600 font-bold'
                      : 'text-slate-500'
                  }
                >
                  {confidencePct}%
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-200 ${
                    confidencePct >= 70
                      ? 'bg-emerald-500'
                      : confidencePct >= 45
                      ? 'bg-amber-500'
                      : 'bg-slate-400'
                  }`}
                  style={{ width: `${Math.min(100, confidencePct)}%` }}
                />
              </div>

              {/* Stability frames progress (Mode A) */}
              {mode === 'guided' && candidate && (
                <div className="flex items-center justify-between text-[11px] text-slate-400 pt-0.5">
                  <span>Stability filter</span>
                  <span className="font-mono">
                    {candidate.stability_count} / {candidate.target_count} frames
                  </span>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-400 space-y-2 py-4">
            <Clock className="w-8 h-8 text-slate-300 stroke-1" />
            <p className="text-sm font-medium text-slate-500">Waiting for sign gesture...</p>
            <p className="text-xs text-slate-400 max-w-xs">
              Perform a dynamic ASL sign in front of the camera. The system will detect your gesture.
            </p>
          </div>
        )}
      </div>

      {/* Manual Actions: Confirm Word / Reject */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={onConfirmWord}
          disabled={!activeWord}
          className={`flex items-center justify-center space-x-1.5 py-2 px-3 rounded-xl text-xs font-semibold shadow-xs transition-all ${
            activeWord
              ? 'bg-emerald-600 hover:bg-emerald-700 text-white cursor-pointer active:scale-98'
              : 'bg-slate-100 text-slate-400 cursor-not-allowed'
          }`}
          title="Manually confirm and commit active word to transcript"
        >
          <Check className="w-4 h-4" />
          <span>Confirm Word</span>
        </button>

        <button
          onClick={onRejectCandidate}
          disabled={!activeWord}
          className={`flex items-center justify-center space-x-1.5 py-2 px-3 rounded-xl border text-xs font-semibold transition-all ${
            activeWord
              ? 'bg-white hover:bg-slate-50 text-slate-700 border-slate-200 cursor-pointer active:scale-98'
              : 'bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed'
          }`}
          title="Reject active candidate and reset buffer"
        >
          <X className="w-4 h-4" />
          <span>Reject / Clear</span>
        </button>
      </div>

      {/* Top Candidates Probability Distribution */}
      <div className="space-y-2 pt-1 border-t border-slate-100">
        <div className="flex items-center justify-between text-xs text-slate-600 font-medium">
          <span className="flex items-center space-x-1">
            <TrendingUp className="w-3.5 h-3.5 text-slate-400" />
            <span>Candidate Distribution</span>
          </span>
          <span className="text-[11px] text-slate-400 font-mono">Top 5</span>
        </div>

        {prediction?.top_k && prediction.top_k.length > 0 ? (
          <div className="space-y-1.5">
            {prediction.top_k.slice(0, 4).map((cand, idx) => {
              const pct = Math.round(cand.confidence * 100);
              return (
                <div key={idx} className="flex items-center text-xs space-x-2">
                  <span className="w-20 truncate font-medium text-slate-700 capitalize">
                    {cand.label}
                  </span>
                  <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        idx === 0 ? 'bg-teal-500' : 'bg-slate-300'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-8 text-right font-mono text-[11px] text-slate-500">
                    {pct}%
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-2 text-xs text-slate-400 italic">
            No active candidates
          </div>
        )}
      </div>

      {/* Mode Advisory Notice */}
      <div
        className={`p-3 rounded-xl text-xs flex items-start space-x-2 border ${
          mode === 'guided'
            ? 'bg-slate-50 border-slate-200 text-slate-600'
            : 'bg-amber-50/60 border-amber-200/80 text-amber-800'
        }`}
      >
        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
        <div className="text-[11px] leading-relaxed">
          {mode === 'guided' ? (
            <>
              <strong>Mode A (Guided Accumulation):</strong> Sign one word at a time, then pause.
              The system confirms stable words and commits them to your transcript.
            </>
          ) : (
            <>
              <strong>Mode B (Continuous - Experimental):</strong> Continuous sliding window active.
              Boundaries are heuristic and based on isolated-model features.
            </>
          )}
        </div>
      </div>
    </div>
  );
};
