import React, { useState, useEffect } from 'react';
import { Video, VideoOff, Eye, Cpu, ShieldCheck, AlertTriangle, Scan, Camera } from 'lucide-react';
import { TelemetryData, api } from '../services/api';

interface LiveCameraPanelProps {
  telemetry: TelemetryData | null;
  onToggleMonitoring: () => void;
}

export const LiveCameraPanel: React.FC<LiveCameraPanelProps> = ({
  telemetry,
  onToggleMonitoring
}) => {
  const [streamError, setStreamError] = useState(false);
  const [snapshotSrc, setSnapshotSrc] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<number>(Date.now());
  const isRunning = telemetry?.status === 'RUNNING';
  const cnn = telemetry?.cnn;
  const ear = telemetry?.ear;
  const fps = telemetry?.fps || 0.0;
  const latency = telemetry?.total_frame_latency_ms || 0.0;

  useEffect(() => {
    if (isRunning) {
      setStreamError(false);
      setSessionId(Date.now());
    }
  }, [isRunning]);

  // Robust snapshot fallback: If multipart stream drops, fetch high-speed snapshot
  useEffect(() => {
    let active = true;
    let timer: any = null;

    if (isRunning && streamError) {
      const pollSnapshot = () => {
        if (!active || !isRunning) return;
        const img = new Image();
        const url = `http://localhost:5000/api/camera/snapshot?t=${Date.now()}`;
        img.onload = () => {
          if (active) {
            setSnapshotSrc(url);
            timer = setTimeout(pollSnapshot, 60);
          }
        };
        img.onerror = () => {
          if (active) {
            timer = setTimeout(pollSnapshot, 200);
          }
        };
        img.src = url;
      };

      pollSnapshot();
    }

    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [isRunning, streamError]);

  const streamUrl = `${api.getStreamUrl()}?session=${sessionId}`;

  const getConfidenceColor = (prob: number) => {
    if (prob >= 0.6) return 'bg-rose-500';
    if (prob >= 0.4) return 'bg-amber-500';
    return 'bg-emerald-500';
  };

  const confidencePercent = Math.round((cnn?.smoothed_probability || 0) * 100);

  return (
    <div className="glass-panel rounded-xl p-5 border border-slate-800 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
        <div className="flex items-center gap-2">
          <Camera className="w-5 h-5 text-cyan-400" />
          <h2 className="text-sm font-bold text-white tracking-wide uppercase font-sans">
            Driver Monitoring Vision Stream
          </h2>
        </div>
        <div className="flex items-center gap-2">
          {isRunning && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-[11px] font-mono text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
              {fps} FPS
            </span>
          )}
          <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300">
            {latency} ms Pipeline Time
          </span>
        </div>
      </div>

      {/* Video Viewport with Industrial DMS Targeting Overlay */}
      <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
        {isRunning ? (
          !streamError ? (
            <img
              key={sessionId}
              src={streamUrl}
              alt="Live Vision Stream"
              onError={() => {
                console.warn('Switching to high-frequency frame polling buffer.');
                setStreamError(true);
              }}
              className="w-full h-full object-cover"
            />
          ) : snapshotSrc ? (
            <img
              src={snapshotSrc}
              alt="Live Vision Stream"
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="text-center p-6 flex flex-col items-center">
              <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin mb-3"></div>
              <p className="text-xs text-slate-300 font-mono">Synchronizing Optical Stream...</p>
            </div>
          )
        ) : (
          <div className="text-center p-6 flex flex-col items-center">
            <div className="p-4 rounded-full bg-slate-900 border border-slate-800 text-slate-500 mb-3">
              <VideoOff className="w-8 h-8" />
            </div>
            <h3 className="text-sm font-semibold text-slate-200 mb-1">
              Optical Telemetry Standby
            </h3>
            <p className="text-xs text-slate-400 max-w-xs mb-4">
              Engage camera monitoring to begin real-time face landmark tracking and drowsiness inference.
            </p>
            <button
              onClick={onToggleMonitoring}
              className="px-5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-mono font-bold tracking-wider uppercase shadow-lg shadow-emerald-950/40 transition-all"
            >
              Engage Monitoring
            </button>
          </div>
        )}

        {/* Professional Corner Targeting HUD Brackets */}
        {isRunning && (
          <>
            <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-cyan-400 pointer-events-none" />
            <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-cyan-400 pointer-events-none" />
            <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-cyan-400 pointer-events-none" />
            <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-cyan-400 pointer-events-none" />

            {/* Live Telemetry Overlay Badges */}
            <div className="absolute top-3 left-3 flex flex-col gap-1.5 pointer-events-none">
              {telemetry?.face_detected ? (
                <span className="px-2 py-0.5 rounded bg-black/75 backdrop-blur-md border border-emerald-500/50 text-[10px] font-mono text-emerald-400 flex items-center gap-1.5">
                  <ShieldCheck className="w-3 h-3" /> FACE LOCK: 478 MESH PTS
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded bg-black/75 backdrop-blur-md border border-rose-500/50 text-[10px] font-mono text-rose-400 flex items-center gap-1.5 animate-pulse">
                  <AlertTriangle className="w-3 h-3" /> OCCLUSION / NO FACE DETECTED
                </span>
              )}

              {/* Real Eye State Prediction Badge */}
              {telemetry?.eye_state?.is_valid ? (
                <span className={`px-2 py-0.5 rounded bg-black/75 backdrop-blur-md border text-[10px] font-mono flex items-center gap-1.5 ${
                  telemetry.eye_state.is_closed ? 'border-rose-500/60 text-rose-400' : 'border-emerald-500/60 text-emerald-300'
                }`}>
                  <Eye className="w-3 h-3" /> EYE CLASSIFIER: {telemetry.eye_state.state.toUpperCase()} ({Math.round(telemetry.eye_state.confidence * 100)}%)
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded bg-black/75 backdrop-blur-md border border-slate-700 text-[10px] font-mono text-cyan-300 flex items-center gap-1.5">
                  <Eye className="w-3 h-3" /> EYE APERTURE (EAR): {ear?.avg_ear.toFixed(3) || '0.000'} ({ear?.is_closed ? 'CLOSED' : 'OPEN'})
                </span>
              )}

              {/* Microsleep Sustained Closure Warning */}
              {(telemetry?.perclos?.sustained_closure_sec || 0) > 0.5 && (
                <span className="px-2 py-0.5 rounded bg-rose-950/90 border border-rose-500 text-[10px] font-mono text-rose-300 flex items-center gap-1.5 animate-pulse font-bold">
                  <AlertTriangle className="w-3 h-3 text-rose-400" />
                  SUSTAINED CLOSURE: {telemetry?.perclos?.sustained_closure_sec?.toFixed(1)}s
                </span>
              )}
            </div>
          </>
        )}
      </div>

      {/* Deep Learning Classification Bar */}
      <div className="mt-4 pt-3 border-t border-slate-800/80">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <div className="flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400 font-medium">Vision Classifier (MobileNetV2):</span>
            <span className={`font-bold font-mono ${
              cnn?.predicted_class === 'Drowsy' ? 'text-rose-400' : 'text-emerald-400'
            }`}>
              {cnn?.is_valid ? cnn?.predicted_class.toUpperCase() : (telemetry?.model_status === 'LOADED' ? 'ANALYZING...' : 'MODEL STANDBY')}
            </span>
          </div>
          <span className="font-mono text-slate-200 font-semibold text-xs">
            {cnn?.is_valid ? `${confidencePercent}% Fatigue Index` : '—'}
          </span>
        </div>

        {/* Confidence Meter Bar */}
        <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
          <div
            className={`h-full transition-all duration-300 ${getConfidenceColor(cnn?.smoothed_probability || 0)}`}
            style={{ width: `${confidencePercent}%` }}
          />
        </div>

        <div className="flex justify-between items-center text-[10px] text-slate-500 font-mono mt-1">
          <span>Attentive (&lt;40%)</span>
          <span className="text-amber-400/90 font-semibold">Threshold: 60%</span>
          <span>Drowsy (&gt;60%)</span>
        </div>
      </div>
    </div>
  );
};
