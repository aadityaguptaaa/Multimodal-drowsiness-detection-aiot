import React from 'react';
import { Layers, ShieldCheck, AlertTriangle, ShieldAlert, Cpu, Eye, Compass, CheckCircle2 } from 'lucide-react';
import { TelemetryData } from '../services/api';

interface SensorFusionPanelProps {
  telemetry: TelemetryData | null;
}

export const SensorFusionPanel: React.FC<SensorFusionPanelProps> = ({ telemetry }) => {
  const fusion = telemetry?.fusion;
  const overallState = fusion?.overall_state || 'NORMAL';
  const score = fusion?.composite_score || 0.0;
  const factors = fusion?.contributing_factors || ['All primary telemetry channels nominal'];
  const confidence = fusion?.confidence_level || 'OPTIMAL';

  const getStateDetails = () => {
    switch (overallState) {
      case 'CRITICAL':
        return {
          title: 'CRITICAL: DROWSINESS INTERVENE',
          color: 'text-rose-400',
          border: 'border-rose-500/50 shadow-rose-950/40',
          bg: 'bg-rose-950/20',
          icon: <ShieldAlert className="w-5 h-5 text-rose-400 animate-pulse" />,
        };
      case 'WARNING':
        return {
          title: 'WARNING: EARLY FATIGUE DETECTED',
          color: 'text-amber-400',
          border: 'border-amber-500/50 shadow-amber-950/40',
          bg: 'bg-amber-950/20',
          icon: <AlertTriangle className="w-5 h-5 text-amber-400" />,
        };
      default:
        return {
          title: 'DRIVER ATTENTIVE & VIGILANT',
          color: 'text-emerald-400',
          border: 'border-emerald-500/40 shadow-emerald-950/40',
          bg: 'bg-emerald-950/20',
          icon: <ShieldCheck className="w-5 h-5 text-emerald-400" />,
        };
    }
  };

  const details = getStateDetails();

  return (
    <div className={`glass-panel rounded-xl p-5 border ${details.border} transition-all duration-300 flex flex-col justify-between`}>
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-cyan-400" />
          <h2 className="text-sm font-bold text-white tracking-wide uppercase font-sans">
            Multi-Sensor Fusion Matrix
          </h2>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
          Confidence: <span className="text-emerald-400 font-bold">{confidence}</span>
        </span>
      </div>

      {/* Main Status Cockpit Card */}
      <div className={`p-4 rounded-xl border ${details.border} ${details.bg} mb-3.5 flex items-center justify-between shadow-lg`}>
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800">
            {details.icon}
          </div>
          <div>
            <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
              Driver Vigilance Status
            </span>
            <h3 className={`text-base font-extrabold tracking-tight font-sans ${details.color}`}>
              {details.title}
            </h3>
          </div>
        </div>

        {/* Composite Risk Metric */}
        <div className="text-right">
          <span className="text-[10px] font-mono text-slate-400 block">Risk Index</span>
          <span className="text-2xl font-black font-mono text-white">
            {score.toFixed(1)} <span className="text-xs text-slate-500 font-normal">/ 100</span>
          </span>
        </div>
      </div>

      {/* Contributing Evidence Channels (Explainable AI Architecture) */}
      <div className="mb-2">
        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider font-mono block mb-2">
          Sensor Channel Evidence Breakdown:
        </span>
        <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
          {factors.map((factor, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2 p-2 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0" />
              <span className="font-mono text-[11px] leading-tight text-slate-200">{factor}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Subsystem Readiness Matrix (Section 6 Requirement) */}
      <div className="pt-2.5 border-t border-slate-800/80">
        <span className="text-[10px] font-mono text-slate-400 block mb-1.5 uppercase font-bold">
          Subsystem Readiness Matrix:
        </span>
        <div className="grid grid-cols-3 gap-1.5 text-center text-[10px] font-mono">
          {/* 1. Camera Operational */}
          <div className="p-1 rounded bg-slate-900 border border-slate-800">
            <span className="text-slate-500 block text-[9px]">Camera</span>
            <span className={telemetry?.camera_available ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
              {telemetry?.camera_available ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>

          {/* 2. Face & Eyes Tracking */}
          <div className="p-1 rounded bg-slate-900 border border-slate-800">
            <span className="text-slate-500 block text-[9px]">Face / Eyes</span>
            <span className={telemetry?.face_detected ? 'text-emerald-400 font-bold' : 'text-amber-400'}>
              {telemetry?.face_detected ? 'TRACKED' : 'SEARCHING'}
            </span>
          </div>

          {/* 3. Eye State Model */}
          <div className="p-1 rounded bg-slate-900 border border-slate-800">
            <span className="text-slate-500 block text-[9px]">Eye Model</span>
            <span className={telemetry?.eye_model_status === 'LOADED' ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
              {telemetry?.eye_model_status === 'LOADED' ? 'LOADED' : 'NOT READY'}
            </span>
          </div>

          {/* 4. PERCLOS Measurement */}
          <div className="p-1 rounded bg-slate-900 border border-slate-800">
            <span className="text-slate-500 block text-[9px]">PERCLOS</span>
            <span className={telemetry?.perclos?.valid_observation ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
              {telemetry?.perclos?.valid_observation ? 'VALID' : 'WARMING UP'}
            </span>
          </div>

          {/* 5. ESP32 Hardware Link */}
          <div className="p-1 rounded bg-slate-900 border border-slate-800">
            <span className="text-slate-500 block text-[9px]">ESP32 Link</span>
            <span className={telemetry?.sensor?.is_connected ? 'text-emerald-400 font-bold' : 'text-rose-400'}>
              {telemetry?.sensor?.is_connected ? `${telemetry?.sensor?.port || 'COM9'}` : 'OFFLINE'}
            </span>
          </div>

          {/* 6. Motion Sensor Hardware Verification */}
          <div className="p-1 rounded bg-slate-900 border border-slate-800">
            <span className="text-slate-500 block text-[9px]">MPU-6050</span>
            <span className={telemetry?.sensor?.chip_verified ? 'text-emerald-400 font-bold' : (telemetry?.sensor_mode === 'SIMULATED' ? 'text-cyan-400' : 'text-slate-500')}>
              {telemetry?.sensor?.chip_verified ? 'VERIFIED' : (telemetry?.sensor_mode === 'SIMULATED' ? 'SIMULATED' : 'UNVERIFIED')}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
