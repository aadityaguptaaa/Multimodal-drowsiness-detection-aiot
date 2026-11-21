import React, { useState } from 'react';
import { Compass, AlertTriangle, Radio, Activity, CheckCircle2, Cpu, RefreshCw, XCircle } from 'lucide-react';
import { TelemetryData, api } from '../services/api';

interface MPU6050PanelProps {
  telemetry: TelemetryData | null;
}

export const MPU6050Panel: React.FC<MPU6050PanelProps> = ({ telemetry }) => {
  const [reconnecting, setReconnecting] = useState(false);
  const [activeTab, setActiveTab] = useState<'DERIVED' | 'RAW'>('DERIVED');

  const sensor = telemetry?.sensor;
  const isConnected = sensor?.is_connected || false;
  const isValid = sensor?.is_valid || false;
  const isVerified = sensor?.chip_verified || false;
  const chipId = sensor?.chip_id || 'UNKNOWN';
  const port = sensor?.port || 'COM9';
  const dataAge = sensor?.data_age_sec ?? 0.0;
  const sensorMode = telemetry?.sensor_mode || 'REAL';

  const pitch = sensor?.pitch ?? 0.0;
  const roll = sensor?.roll ?? 0.0;
  const yaw = sensor?.yaw ?? 0.0;

  const ax = sensor?.ax ?? 0.0;
  const ay = sensor?.ay ?? 0.0;
  const az = sensor?.az ?? 0.0;

  const gx = sensor?.gx ?? 0.0;
  const gy = sensor?.gy ?? 0.0;
  const gz = sensor?.gz ?? 0.0;

  // Posture Stability Index (100% when upright, decreases with extreme nod or tilt)
  const deviation = Math.sqrt(pitch * pitch + roll * roll);
  const stabilityIndex = isValid ? Math.max(0, Math.min(100, Math.round(100 - deviation * 1.5))) : 0;

  const handleReconnect = async () => {
    setReconnecting(true);
    try {
      await api.configureSensorPort(port, 115200);
    } catch (e) {
      console.error(e);
    } finally {
      setTimeout(() => setReconnecting(false), 1200);
    }
  };

  const handleToggleMode = async () => {
    const nextMode = sensorMode === 'REAL' ? 'SIMULATED' : 'REAL';
    try {
      await api.setSensorMode(nextMode);
    } catch (e) {
      console.error(e);
    }
  };

  const handleTestEpisode = async (type: string) => {
    try {
      if (sensorMode !== 'SIMULATED') {
        await api.setSensorMode('SIMULATED');
      }
      if (type === 'NOD') {
        await api.simulateSensorState('HEAD_NOD');
        setTimeout(() => api.simulateSensorState('NORMAL'), 3500);
      } else if (type === 'TILT') {
        await api.simulateSensorState('HEAD_TILT');
        setTimeout(() => api.simulateSensorState('NORMAL'), 3500);
      } else {
        await api.simulateSensorState('NORMAL');
      }
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="glass-panel rounded-xl p-5 border border-slate-800 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
        <div className="flex items-center gap-2">
          <Compass className="w-5 h-5 text-cyan-400" />
          <div>
            <h2 className="text-sm font-bold text-white tracking-wide uppercase font-sans">
              ESP32 IMU Telemetry
            </h2>
          </div>
        </div>

        {/* Real Hardware Link Badge */}
        <div className="flex items-center gap-1.5">
          <span className={`px-2 py-0.5 rounded text-[10px] font-mono flex items-center gap-1 border ${
            isConnected && isVerified
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              : isConnected
              ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
              : sensorMode === 'SIMULATED'
              ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30'
              : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
          }`}>
            <Radio className="w-3 h-3" />
            {isConnected ? `${port}: ${isVerified ? 'MPU-6050 VERIFIED' : 'UNVERIFIED'}` : (sensorMode === 'SIMULATED' ? 'SIMULATOR TEST FIXTURE' : `${port}: DISCONNECTED`)}
          </span>
        </div>
      </div>

      {/* Artificial Horizon / Posture Attitude Visualizer */}
      <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 mb-3 relative overflow-hidden">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-2 font-mono">
          <span>Head Attitude Horizon</span>
          {isValid ? (
            <span className={stabilityIndex > 75 ? 'text-emerald-400' : 'text-amber-400'}>
              Stability: {stabilityIndex}%
            </span>
          ) : (
            <span className="text-rose-400 font-semibold">FEED OFFLINE</span>
          )}
        </div>

        {/* Graphical Horizon Indicator */}
        <div className="h-14 w-full bg-slate-900/90 rounded-lg border border-slate-800 relative flex items-center justify-center overflow-hidden">
          {/* Pitch Lines */}
          <div className="absolute inset-0 flex flex-col justify-between py-1 opacity-20 pointer-events-none">
            <div className="border-b border-cyan-400 w-1/3 mx-auto"></div>
            <div className="border-b border-cyan-400 w-1/2 mx-auto"></div>
            <div className="border-b border-cyan-400 w-1/3 mx-auto"></div>
          </div>

          {/* Dynamic Attitude Bar reflecting Pitch & Roll */}
          {isValid ? (
            <div
              className="w-28 h-1 rounded transition-transform duration-200 shadow-md"
              style={{
                backgroundColor: pitch <= -15 ? '#f43f5e' : (Math.abs(roll) >= 25 ? '#f59e0b' : '#06b6d4'),
                transform: `rotate(${roll}deg) translateY(${Math.max(-20, Math.min(20, -pitch * 0.8))}px)`,
                boxShadow: pitch <= -15 ? '0 0 12px rgba(244,63,94,0.8)' : '0 0 10px rgba(6,182,212,0.6)'
              }}
            />
          ) : (
            <div className="text-[11px] font-mono text-slate-500">
              No Sensor Data ({port} Offline)
            </div>
          )}

          {/* Center Crosshair */}
          <div className="w-2 h-2 rounded-full border border-white/60 pointer-events-none absolute"></div>
        </div>

        <div className="flex justify-between items-center text-[10px] font-mono text-slate-500 mt-1.5">
          <span>Roll Range: ±45°</span>
          <span>Freshness: {isValid ? `${dataAge.toFixed(2)}s` : '—'}</span>
          <span>Pitch Range: -40° to +20°</span>
        </div>
      </div>

      {/* Tabs: Derived Head Angles vs Raw Accelerometer & Gyroscope */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex rounded-lg bg-slate-900 p-0.5 border border-slate-800 text-[11px] font-mono">
          <button
            onClick={() => setActiveTab('DERIVED')}
            className={`px-3 py-1 rounded-md transition-colors ${
              activeTab === 'DERIVED' ? 'bg-cyan-950 text-cyan-300 font-bold border border-cyan-800/80' : 'text-slate-400 hover:text-white'
            }`}
          >
            Derived Angles
          </button>
          <button
            onClick={() => setActiveTab('RAW')}
            className={`px-3 py-1 rounded-md transition-colors ${
              activeTab === 'RAW' ? 'bg-cyan-950 text-cyan-300 font-bold border border-cyan-800/80' : 'text-slate-400 hover:text-white'
            }`}
          >
            Raw 6-DOF (ax/gx)
          </button>
        </div>

        <button
          onClick={handleToggleMode}
          title="Toggle between Real ESP32 Serial Port and Controlled Test Simulator"
          className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 hover:bg-slate-850 border border-slate-800 text-slate-300 transition-colors"
        >
          Mode: <span className={sensorMode === 'REAL' ? 'text-emerald-400 font-bold' : 'text-cyan-400 font-bold'}>{sensorMode}</span>
        </button>
      </div>

      {/* Telemetry Readouts */}
      {activeTab === 'DERIVED' ? (
        <div className="grid grid-cols-3 gap-2 mb-2.5">
          {/* Pitch */}
          <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-center">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">
              Pitch (Nod)
            </span>
            <div className={`text-xl font-bold font-mono my-0.5 ${pitch <= -15 ? 'text-rose-400 glow-rose' : 'text-white'}`}>
              {isValid ? `${pitch.toFixed(1)}°` : '—'}
            </div>
            <span className="text-[9px] font-mono text-slate-500 block">
              {isValid ? (pitch <= -15 ? 'NOD DETECTED' : 'UPRIGHT') : 'NO DATA'}
            </span>
          </div>

          {/* Roll */}
          <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-center">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">
              Roll (Slump)
            </span>
            <div className={`text-xl font-bold font-mono my-0.5 ${Math.abs(roll) >= 25 ? 'text-amber-400' : 'text-white'}`}>
              {isValid ? `${roll.toFixed(1)}°` : '—'}
            </div>
            <span className="text-[9px] font-mono text-slate-500 block">
              {isValid ? (Math.abs(roll) >= 25 ? 'LATERAL SLUMP' : 'BALANCED') : 'NO DATA'}
            </span>
          </div>

          {/* Yaw */}
          <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-center">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">
              Yaw (Azimuth)
            </span>
            <div className="text-xl font-bold font-mono my-0.5 text-white">
              {isValid ? `${yaw.toFixed(1)}°` : '—'}
            </div>
            <span className="text-[9px] font-mono text-slate-500 block">
              {isValid ? 'HEADING' : 'NO DATA'}
            </span>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2 mb-2.5">
          {/* Accelerometer Raw Axes */}
          <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
            <span className="text-[10px] font-mono text-slate-400 block uppercase font-bold">
              Accelerometer (m/s²)
            </span>
            <div className="font-mono text-xs text-slate-200 flex justify-between">
              <span className="text-slate-400">ax:</span> <span>{isValid ? ax.toFixed(2) : '—'}</span>
            </div>
            <div className="font-mono text-xs text-slate-200 flex justify-between">
              <span className="text-slate-400">ay:</span> <span>{isValid ? ay.toFixed(2) : '—'}</span>
            </div>
            <div className="font-mono text-xs text-slate-200 flex justify-between">
              <span className="text-slate-400">az:</span> <span>{isValid ? az.toFixed(2) : '—'}</span>
            </div>
          </div>

          {/* Gyroscope Raw Axes */}
          <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
            <span className="text-[10px] font-mono text-slate-400 block uppercase font-bold">
              Gyroscope (°/s)
            </span>
            <div className="font-mono text-xs text-slate-200 flex justify-between">
              <span className="text-slate-400">gx:</span> <span>{isValid ? gx.toFixed(1) : '—'}</span>
            </div>
            <div className="font-mono text-xs text-slate-200 flex justify-between">
              <span className="text-slate-400">gy:</span> <span>{isValid ? gy.toFixed(1) : '—'}</span>
            </div>
            <div className="font-mono text-xs text-slate-200 flex justify-between">
              <span className="text-slate-400">gz:</span> <span>{isValid ? gz.toFixed(1) : '—'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Driver Posture Diagnostic Status */}
      <div className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-xs mb-2.5 flex items-center justify-between">
        <div className="flex items-center gap-2 truncate">
          {isValid ? (
            pitch <= -15 ? (
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 animate-pulse" />
            ) : (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            )
          ) : (
            <XCircle className="w-4 h-4 text-rose-500 shrink-0" />
          )}
          <span className="text-slate-300 font-medium text-[11px] truncate">
            {sensor?.state_description || 'Sensor Offline'}
          </span>
        </div>
        <span className="font-mono text-[10px] text-slate-400 shrink-0 ml-2">
          {sensor?.chip_id || 'ID: UNKNOWN'}
        </span>
      </div>

      {/* Hardware Telemetry Diagnostics & Calibration */}
      <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between gap-2">
        <button
          onClick={handleReconnect}
          disabled={reconnecting}
          className="flex-1 py-1.5 px-3 rounded-lg bg-slate-900 hover:bg-slate-850 border border-slate-800 text-[11px] font-mono text-slate-300 transition-colors flex items-center justify-center gap-1.5"
        >
          <RefreshCw className={`w-3 h-3 text-cyan-400 ${reconnecting ? 'animate-spin' : ''}`} />
          <span>{reconnecting ? 'Scanning COM9...' : 'Scan / Reconnect ESP32'}</span>
        </button>

        <button
          onClick={() => handleTestEpisode('NOD')}
          title="Verify Head-Nod trigger using controlled test stimulus"
          className="py-1.5 px-3 rounded-lg bg-slate-900 hover:bg-slate-850 border border-slate-800 text-[11px] font-mono text-slate-400 hover:text-cyan-300 transition-colors"
        >
          Simulate Nod Test
        </button>
      </div>

      {/* Scientific disclaimer */}
      <div className="mt-2 text-[9px] font-mono text-slate-500 leading-tight">
        * Physical motion telemetry provides secondary head-posture corroboration; accelerometer/gyro readings alone do not establish driver drowsiness.
      </div>
    </div>
  );
};
