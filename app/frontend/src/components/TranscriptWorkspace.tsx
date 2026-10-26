import React, { useState, useEffect } from 'react';
import {
  FileText,
  Copy,
  Download,
  RotateCcw,
  Trash2,
  Check,
  Plus,
  X,
  Sparkles,
  BookOpen,
  Volume2
} from 'lucide-react';
import { CommittedToken } from '../types';

interface TranscriptWorkspaceProps {
  tokens: CommittedToken[];
  onRemoveToken: (id: string) => void;
  onClearTranscript: () => void;
  onUndo: () => void;
  transcriptText: string;
  setTranscriptText: (text: string) => void;
  onNewSession: () => void;
}

export const TranscriptWorkspace: React.FC<TranscriptWorkspaceProps> = ({
  tokens,
  onRemoveToken,
  onClearTranscript,
  onUndo,
  transcriptText,
  setTranscriptText,
  onNewSession,
}) => {
  const [activeTab, setActiveTab] = useState<'gloss' | 'draft'>('gloss');
  const [copied, setCopied] = useState<boolean>(false);
  const [confirmClearOpen, setConfirmClearOpen] = useState<boolean>(false);

  // Compute formatted gloss draft assistance from tokens
  const generateEnglishDraft = (words: string[]): string => {
    if (words.length === 0) return '';
    const joined = words.join(' ').toLowerCase();
    // Capitalize first letter of sentences
    const capitalized = joined.replace(/(^\w|\.\s+\w)/g, (c) => c.toUpperCase());
    return capitalized.endsWith('.') ? capitalized : `${capitalized}.`;
  };

  const prevTokensLengthRef = React.useRef<number>(tokens.length);

  // Preserve user edits: append new tokens when they arrive without clobbering existing edits
  useEffect(() => {
    const prevLen = prevTokensLengthRef.current;
    if (tokens.length === 0) {
      if (prevLen > 0) {
        setTranscriptText('');
      }
    } else if (tokens.length > prevLen) {
      // Append newly committed word(s)
      const newTokens = tokens.slice(prevLen);
      const newWords = newTokens.map((t) => t.word.toUpperCase()).join(' ');
      const updated = transcriptText.trim() ? `${transcriptText.trim()} ${newWords}` : newWords;
      setTranscriptText(updated);
    }
    prevTokensLengthRef.current = tokens.length;
  }, [tokens, transcriptText, setTranscriptText]);

  // Copy to clipboard
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(transcriptText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.warn('Copy failed:', err);
    }
  };

  // Download transcript as .txt
  const handleDownload = () => {
    const element = document.createElement('a');
    const file = new Blob([transcriptText], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `SignFlow_Transcript_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Quick punctuation inserters
  const insertPunctuation = (punct: string) => {
    if (punct === ' ') {
      setTranscriptText(transcriptText + ' ');
    } else {
      const trimmed = transcriptText.trimEnd();
      setTranscriptText(trimmed + punct + ' ');
    }
  };

  const wordCount = transcriptText.trim() ? transcriptText.trim().split(/\s+/).length : 0;
  const charCount = transcriptText.length;

  return (
    <div className="bg-slate-900/90 backdrop-blur-md rounded-2xl border border-slate-800 shadow-2xl p-4 sm:p-6 flex flex-col space-y-4 text-slate-100">
      {/* Workspace Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
            <FileText className="w-4 h-4" />
          </div>
          <div className="flex items-center space-x-2">
            <h2 className="text-sm sm:text-base font-bold text-white tracking-tight">Transcript Workspace</h2>
            <span className="text-[11px] bg-slate-800 text-teal-300 border border-slate-700 px-2.5 py-0.5 rounded-full font-mono">
              {tokens.length} {tokens.length === 1 ? 'word' : 'words'}
            </span>
          </div>
        </div>

        {/* View Tabs: Gloss Tokens vs English Draft */}
        <div className="flex items-center bg-slate-800/90 p-1 rounded-xl border border-slate-700 text-xs">
          <button
            onClick={() => setActiveTab('gloss')}
            className={`px-3 py-1 rounded-lg font-medium transition-all ${
              activeTab === 'gloss'
                ? 'bg-teal-500 text-slate-950 font-bold shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Recognized Glosses
          </button>
          <button
            onClick={() => setActiveTab('draft')}
            className={`px-3 py-1 rounded-lg font-medium transition-all flex items-center space-x-1.5 ${
              activeTab === 'draft'
                ? 'bg-indigo-600 text-white font-bold shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>English Draft</span>
          </button>
        </div>
      </div>

      {/* Gloss Chips View */}
      {activeTab === 'gloss' ? (
        <div className="space-y-3">
          <div className="min-h-[56px] p-3 bg-slate-950/80 border border-slate-800/80 rounded-xl flex flex-wrap gap-2 items-center">
            {tokens.length > 0 ? (
              tokens.map((token, index) => (
                <div
                  key={token.id || index}
                  className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 shadow-xs text-xs font-semibold text-slate-200 animate-in fade-in zoom-in-95 duration-150"
                >
                  <span className="uppercase tracking-wide text-teal-300 font-bold">{token.word}</span>
                  {token.confidence > 0 && (
                    <span className="text-[10px] text-slate-400 font-mono">
                      {Math.round(token.confidence * 100)}%
                    </span>
                  )}
                  <button
                    onClick={() => onRemoveToken(token.id)}
                    className="p-0.5 text-slate-400 hover:text-rose-400 rounded-full hover:bg-slate-700 transition-colors"
                    title="Delete word"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))
            ) : (
              <span className="text-xs text-slate-500 italic">
                Confirmed sign words accumulate here in real-time as vocabulary chips...
              </span>
            )}
          </div>
        </div>
      ) : (
        /* Formatted Gloss Draft View Notice */
        <div className="p-3.5 bg-indigo-500/10 border border-indigo-500/30 rounded-xl text-xs text-indigo-200 space-y-1">
          <div className="font-semibold flex items-center space-x-1.5 text-indigo-300">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <span>Formatted Gloss Draft (Assisted Formatting)</span>
          </div>
          <p className="text-[11px] text-indigo-200/80 leading-relaxed">
            Formatted Draft: <em className="text-white">"{generateEnglishDraft(tokens.map((t) => t.word)) || 'No words accumulated yet.'}"</em>
            <br />
            (Note: Formats recognized glosses with standard sentence capitalization and punctuation. ASL glosses do not have a 1:1 syntax with English sentences.)
          </p>
        </div>
      )}

      {/* Editable Multi-Line Textarea */}
      <div className="space-y-2">
        <textarea
          value={transcriptText}
          onChange={(e) => setTranscriptText(e.target.value)}
          placeholder="Recognized words will accumulate here. You can also type, correct, or format the text directly..."
          rows={3}
          className="w-full p-3.5 text-sm bg-slate-950/90 border border-slate-800 rounded-xl text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500 transition-all shadow-inner leading-relaxed resize-y"
        />

        {/* Counts & Punctuation Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          {/* Quick Punctuation Buttons */}
          <div className="flex items-center space-x-1 bg-slate-800/90 p-1 rounded-xl border border-slate-700">
            <span className="text-[11px] font-medium text-slate-400 px-1.5">Insert:</span>
            {['.', ',', '?', '!', ' '].map((p, idx) => (
              <button
                key={idx}
                onClick={() => insertPunctuation(p)}
                className="w-6 h-6 rounded-lg bg-slate-700 hover:bg-slate-600 border border-slate-600 text-slate-200 font-mono text-xs transition-colors flex items-center justify-center font-bold"
                title={`Insert ${p === ' ' ? 'Space' : p}`}
              >
                {p === ' ' ? '␣' : p}
              </button>
            ))}
          </div>

          <div className="text-slate-500 font-mono text-[11px]">
            {wordCount} words | {charCount} chars
          </div>
        </div>
      </div>

      {/* Transcript Action Controls */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800">
        <div className="flex items-center space-x-2">
          <button
            onClick={onUndo}
            disabled={tokens.length === 0}
            className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all ${
              tokens.length > 0
                ? 'bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700 cursor-pointer active:scale-95'
                : 'bg-slate-800/40 border-slate-800 text-slate-600 cursor-not-allowed'
            }`}
            title="Undo most recent committed word"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Undo</span>
          </button>

          {!confirmClearOpen ? (
            <button
              onClick={() => setConfirmClearOpen(true)}
              disabled={tokens.length === 0 && !transcriptText}
              className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all ${
                tokens.length > 0 || transcriptText
                  ? 'bg-slate-800 border-slate-700 text-rose-400 hover:bg-rose-500/10 cursor-pointer active:scale-95'
                  : 'bg-slate-800/40 border-slate-800 text-slate-600 cursor-not-allowed'
              }`}
              title="Clear transcript"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear</span>
            </button>
          ) : (
            <div className="flex items-center space-x-1.5 bg-rose-500/15 p-1 rounded-xl border border-rose-500/30 text-xs">
              <span className="text-rose-300 text-[11px] font-medium px-1">Clear all?</span>
              <button
                onClick={() => {
                  onClearTranscript();
                  setConfirmClearOpen(false);
                }}
                className="px-2 py-0.5 rounded-lg bg-rose-600 text-white font-bold text-[11px] hover:bg-rose-700 active:scale-95"
              >
                Yes
              </button>
              <button
                onClick={() => setConfirmClearOpen(false)}
                className="px-2 py-0.5 rounded-lg bg-slate-700 text-slate-300 font-medium text-[11px] hover:bg-slate-600"
              >
                No
              </button>
            </div>
          )}

          <button
            onClick={onNewSession}
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-all active:scale-95"
            title="Reset session and start new conversation"
          >
            <span>New Session</span>
          </button>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={handleCopy}
            disabled={!transcriptText}
            className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all ${
              transcriptText
                ? 'bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700 cursor-pointer active:scale-95'
                : 'bg-slate-800/40 border-slate-800 text-slate-600 cursor-not-allowed'
            }`}
            title="Copy transcript to clipboard"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied!' : 'Copy'}</span>
          </button>

          <button
            onClick={handleDownload}
            disabled={!transcriptText}
            className={`inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all ${
              transcriptText
                ? 'bg-gradient-to-r from-teal-500 to-indigo-600 hover:from-teal-400 hover:to-indigo-500 text-slate-950 shadow-md shadow-teal-500/20 active:scale-95 cursor-pointer'
                : 'bg-slate-800/40 text-slate-600 cursor-not-allowed'
            }`}
            title="Download transcript as text file"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export .txt</span>
          </button>
        </div>
      </div>
    </div>
  );
};
