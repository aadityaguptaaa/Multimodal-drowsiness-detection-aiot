import React, { useState, useEffect } from 'react';
import { Activity, Cpu, Volume2, VolumeX, Play, Square, Database, Wrench, Shield, Radio } from 'lucide-react';
import { TelemetryData } from '../services/api';

interface NavbarProps {
  telemetry: TelemetryData | null;
  backendConnected: boolean;
  onToggleMonitoring: () => void;
  onOpenModelModal: () => void;
  onOpenDatasetModal: () => void;
  onOpenHardwareModal: () => void;
  onToggleAudio: () => void;
  audioEnabled: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  telemetry,
  backendConnected,
  onToggleMonitoring,
  onOpenModelModal,
  onOpenDatasetModal,
  onOpenHardwareModal,
  onToggleAudio,
  audioEnabled
}) => {
  const [timeStr, setTimeStr] = useState<string>('');

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeStr(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const isRunning = telemetry?.status === 'RUNNING';
  const overallState = telemetry?.fusion?.overall_state || 'NORMAL';

  const getStateBadge = () => {
    if (!isRunning) {
      return (
        <span className="px-3 py-1 rounded-md text-xs font-mono font-semibold bg-slate-900 text-slate-400 border border-slate-800 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-slate-600"></span>
          DMS STANDBY
        </span>
      );
    }
    if (overallState === 'CRITICAL') {
      return (
        <span className="px-3 py-1 rounded-md text-xs font-mono font-bold bg-rose-500/15 text-rose-400 border border-rose-500/40 flex items-center gap-2 animate-pulse glow-rose">
          <span className="w-2 h-2 rounded-full bg-rose-500"></span>
          CRITICAL: DROWSINESS INTERVENE
        </span>
      );
    }
    if (overallState === 'WARNING') {
      return (
        <span className="px-3 py-1 rounded-md text-xs font-mono font-bold bg-amber-500/15 text-amber-400 border border-amber-500/40 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-amber-500"></span>
          WARNING: FATIGUE DETECTED
        </span>
      );
    }
    return (
      <span className="px-3 py-1 rounded-md text-xs font-mono font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/40 flex items-center gap-2 glow-emerald">
        <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
        DRIVER ATTENTIVE (OPTIMAL)
      </span>
    );
  };

  return (
    <header className="glass-panel border-b border-slate-800/80 px-6 py-3 sticky top-0 z-40 bg-navy-950/90">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* System Title & Automotive Branding */}
        <div className="flex items-center gap-3.5">
          <div className="p-2.5 rounded-lg bg-cyan-950/60 border border-cyan-800/80 text-cyan-400">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-base font-extrabold tracking-tight text-white uppercase font-sans">
                AIoT Driver Monitoring System
              </h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 text-cyan-400 border border-slate-700">
                TELEMATICS CONSOLE
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium">
              Multi-Sensor Fusion: MobileNetV2 + 10s Rolling PERCLOS + MPU-6050 Inertial IMU
            </p>
          </div>
        </div>

        {/* Global Hardware & Telematics Pills */}
        <div className="flex flex-wrap items-center gap-2">
          {getStateBadge()}

          {/* Vision Pipeline Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs font-mono">
            <span className={`w-1.5 h-1.5 rounded-full ${backendConnected ? 'bg-emerald-400' : 'bg-rose-500'}`} />
            <span className="text-slate-400">TELEMETRY:</span>
            <span className="text-white font-semibold">{backendConnected ? 'ONLINE' : 'LINK LOST'}</span>
          </div>

          {/* CNN Architecture Tag */}
          <button
            onClick={onOpenModelModal}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 hover:bg-slate-850 border border-slate-800 text-xs font-mono text-slate-300 transition-colors"
          >
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">CNN:</span>
            <span className={telemetry?.model_status === 'LOADED' ? 'text-emerald-400 font-bold' : 'text-amber-400'}>
              {telemetry?.model_status === 'LOADED' ? 'MOBILENET-V2' : 'INIT'}
            </span>
          </button>

          {/* Eye State Model Tag */}
          <button
            onClick={onOpenModelModal}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 hover:bg-slate-850 border border-slate-800 text-xs font-mono text-slate-300 transition-colors"
          >
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">EYE CLF:</span>
            <span className={telemetry?.eye_model_status === 'LOADED' ? 'text-emerald-400 font-bold' : 'text-amber-400'}>
              {telemetry?.eye_model_status === 'LOADED' ? 'CONVNET' : 'NOT READY'}
            </span>
          </button>

          {/* ESP32 IMU Hardware Channel */}
          <button
            onClick={onOpenHardwareModal}
            className="px-2.5 py-1 rounded-md bg-slate-900 hover:bg-slate-850 border border-slate-800 text-xs font-mono flex items-center gap-1.5 transition-colors"
          >
            <Radio className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">ESP32:</span>
            <span className={
              telemetry?.sensor?.is_connected
                ? (telemetry?.sensor?.chip_verified ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold')
                : (telemetry?.sensor_mode === 'SIMULATED' ? 'text-cyan-400 font-bold' : 'text-slate-500 font-semibold')
            }>
              {telemetry?.sensor?.is_connected
                ? `${telemetry?.sensor?.port || 'COM9'} (${telemetry?.sensor?.chip_verified ? '0x68' : 'UNVERIFIED'})`
                : (telemetry?.sensor_mode === 'SIMULATED' ? 'SIMULATOR' : `${telemetry?.sensor?.port || 'COM9'} OFFLINE`)}
            </span>
          </button>

          {/* Clock */}
          <div className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300 font-semibold">
            {timeStr}
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {/* Audio Alarm Toggle */}
          <button
            onClick={onToggleAudio}
            title={audioEnabled ? 'Mute In-Cabin Alarm' : 'Unmute In-Cabin Alarm'}
            className={`p-2 rounded-lg border transition-all ${
              audioEnabled
                ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-400 hover:bg-cyan-500/20'
                : 'bg-slate-900 border-slate-800 text-slate-500 hover:bg-slate-850'
            }`}
          >
            {audioEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>

          {/* Technical Modals */}
          <button
            onClick={onOpenDatasetModal}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-200 transition-colors"
          >
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            <span>Dataset Audit (DDD)</span>
          </button>

          <button
            onClick={onOpenHardwareModal}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-200 transition-colors"
          >
            <Wrench className="w-3.5 h-3.5 text-amber-400" />
            <span>Hardware BOM</span>
          </button>

          {/* Start/Stop Camera Monitoring */}
          <button
            onClick={onToggleMonitoring}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-mono font-bold tracking-wider uppercase shadow-lg transition-all ${
              isRunning
                ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-950/40'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-950/40'
            }`}
          >
            {isRunning ? (
              <>
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>Halt Monitoring</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Engage System</span>
              </>
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
