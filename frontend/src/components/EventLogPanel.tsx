import React, { useState } from 'react';
import { Bell, BellOff, Volume2, Trash2, ShieldAlert, AlertTriangle, Info, Terminal } from 'lucide-react';
import { EventItem, api } from '../services/api';

interface EventLogPanelProps {
  events: EventItem[];
  audioEnabled: boolean;
  onToggleAudio: () => void;
  onClearHistory: () => void;
  cooldownRemaining: number;
}

export const EventLogPanel: React.FC<EventLogPanelProps> = ({
  events,
  audioEnabled,
  onToggleAudio,
  onClearHistory,
  cooldownRemaining
}) => {
  const [testingBeep, setTestingBeep] = useState(false);

  const handleTestBeep = async () => {
    setTestingBeep(true);
    try {
      await api.testAlert();
    } catch (e) {
      console.error(e);
    } finally {
      setTimeout(() => setTestingBeep(false), 500);
    }
  };

  const getEventIcon = (level: string) => {
    switch (level) {
      case 'CRITICAL':
        return <ShieldAlert className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />;
      case 'WARNING':
        return <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />;
      default:
        return <Info className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />;
    }
  };

  return (
    <div className="glass-panel rounded-xl p-5 border border-slate-800 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-amber-400" />
          <h2 className="text-sm font-bold text-white tracking-wide uppercase font-sans">
            In-Vehicle Telematics Event Log
          </h2>
        </div>

        {/* Cooldown Status */}
        <div className="flex items-center gap-2">
          {cooldownRemaining > 0 && (
            <span className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-[10px] font-mono text-amber-400 animate-pulse">
              Cooldown: {cooldownRemaining}s
            </span>
          )}
          <button
            onClick={onClearHistory}
            title="Purge Telematics Log"
            className="p-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Action Controls */}
      <div className="flex items-center gap-2 mb-3">
        <button
          onClick={handleTestBeep}
          disabled={testingBeep}
          className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-850 border border-slate-800 text-xs font-mono font-semibold text-slate-200 transition-all"
        >
          <Volume2 className="w-3.5 h-3.5 text-cyan-400" />
          <span>{testingBeep ? 'Transmitting Audio Pulse...' : 'Test Acoustic Alarm'}</span>
        </button>

        <button
          onClick={onToggleAudio}
          className={`flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-semibold border transition-all ${
            audioEnabled
              ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-300'
              : 'bg-slate-900 border-slate-800 text-slate-500'
          }`}
        >
          {audioEnabled ? <Volume2 className="w-3.5 h-3.5" /> : <BellOff className="w-3.5 h-3.5" />}
          <span>{audioEnabled ? 'Alarm Active' : 'Muted'}</span>
        </button>
      </div>

      {/* Event Stream (Blackbox Logger Style) */}
      <div className="flex-1 max-h-48 overflow-y-auto space-y-2 pr-1 font-mono">
        {events.length === 0 ? (
          <div className="text-center py-8 text-xs text-slate-500 font-mono">
            No safety incidents logged. System operating nominally.
          </div>
        ) : (
          events.map((evt) => (
            <div
              key={evt.id}
              className={`p-2.5 rounded-lg border text-xs flex items-start gap-2.5 transition-all ${
                evt.level === 'CRITICAL'
                  ? 'bg-rose-950/20 border-rose-500/30 text-rose-200'
                  : evt.level === 'WARNING'
                  ? 'bg-amber-950/20 border-amber-500/30 text-amber-200'
                  : 'bg-slate-900/60 border-slate-800 text-slate-300'
              }`}
            >
              {getEventIcon(evt.level)}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-[10px] text-slate-400">{evt.timestamp}</span>
                  <span className="font-mono text-[9px] uppercase px-1.5 py-0.2 rounded bg-black/40 text-slate-300 border border-slate-800">
                    {evt.level}
                  </span>
                </div>
                <p className="text-[11px] font-sans mt-0.5 leading-snug break-words">
                  {evt.message}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
