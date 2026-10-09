import React from 'react';
import { X, BookOpen, Layers, ShieldCheck, Cpu, GitBranch } from 'lucide-react';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xl max-w-lg w-full p-6 space-y-4 max-h-[90vh] overflow-y-auto animate-in zoom-in-95 duration-150 text-xs">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center space-x-2">
            <BookOpen className="w-5 h-5 text-indigo-600" />
            <h3 className="text-base font-bold text-slate-800">About SignFlow & Thesis</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="space-y-4 text-slate-600 leading-relaxed">
          <div>
            <h4 className="font-bold text-slate-900 text-sm">Thesis Research Title</h4>
            <p className="mt-1 text-slate-700 font-medium">
              <em>
                Robust Real-Time Dynamic Word-Level American Sign Language Recognition: Mitigating
                Spatial-Temporal Noise via Holistic Feature Extraction
              </em>
            </p>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded-xl space-y-2">
            <div className="flex items-center space-x-2 font-bold text-slate-800">
              <Cpu className="w-4 h-4 text-teal-600" />
              <span>Feature Extraction Pipeline</span>
            </div>
            <ul className="list-disc list-inside space-y-1 text-slate-600 text-[11px]">
              <li>Canonical 553-point Holistic Landmark schema (Hands: 42, Pose: 33, Face: 478).</li>
              <li>Per-frame structured geometric normalization with visibility mask preservation.</li>
              <li>Boundary-safe velocity and acceleration derivatives yielding 4,977-dim temporal sequences.</li>
              <li>Trained models: Temporal Transformer Baseline & Robust Holistic Fusion Classifier.</li>
            </ul>
          </div>

          <div className="p-3 bg-indigo-50/60 border border-indigo-100 rounded-xl space-y-2">
            <div className="flex items-center space-x-2 font-bold text-indigo-900">
              <Layers className="w-4 h-4 text-indigo-600" />
              <span>Operating Modes & Research Integrity</span>
            </div>
            <div className="space-y-1.5 text-[11px] text-indigo-950">
              <p>
                <strong>Mode A (Guided Word Accumulation):</strong> Primary operational mode. Evaluates
                isolated sign gestures using calibrated confidence thresholds and debounce stability.
                Allows accumulating verified tokens into persistent sentences and speech.
              </p>
              <p>
                <strong>Mode B (Continuous Sign Recognition - Experimental):</strong> Continuous sliding
                window prototype. Word boundaries and temporal segmentation are active as experimental
                heuristics. Genuine continuous signing requires specialized continuous datasets (e.g. How2Sign).
              </p>
            </div>
          </div>

          <div>
            <h4 className="font-bold text-slate-900">Academic Citation & Dataset Agreement</h4>
            <p className="mt-1 text-[11px] text-slate-500">
              Built upon the WLASL (Word-Level American Sign Language) dataset conforming to the
              Computational Use of Data Agreement. Trained on disjoint signer splits for robust
              held-out generalization.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end pt-3 border-t border-slate-100">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white font-semibold text-xs transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
