import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Wifi,
  WifiOff,
  AlertTriangle,
  RefreshCw,
  Activity,
  CheckCircle2,
  Clock,
  Radio,
  Sliders
} from 'lucide-react';

import { TelemetryData } from '../services/api';

interface BridgeTelemetryData {
  status: 'ONLINE' | 'CONNECTING' | 'OFFLINE' | 'STALE' | 'ERROR';
  port: string;
  baud: number;
  chip_id: string;
  chip_verified: boolean;
  is_connected: boolean;
  is_stale: boolean;
  data_age_sec: number;
  telemetry: {
    ax: number;
    ay: number;
    az: number;
    gx: number;
    gy: number;
    gz: number;
    pitch: number;
    roll: number;
    yaw: number;
  };
  packets_received: number;
  reconnect_count: number;
  uptime_sec: number;
  error: string | null;
  timestamp: number;
}

interface HardwareSensorPanelProps {
  telemetry?: TelemetryData | null;
}

export const HardwareSensorPanel: React.FC<HardwareSensorPanelProps> = ({ telemetry }) => {
  const [data, setData] = useState<BridgeTelemetryData | null>(null);
  const [bridgeReachable, setBridgeReachable] = useState<boolean>(false);
  const [lastPollTime, setLastPollTime] = useState<Date | null>(null);
  const [isManualRefreshing, setIsManualRefreshing] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    const fetchBridgeData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8765/data', {
          signal: controller.signal,
          cache: 'no-store'
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const json: BridgeTelemetryData = await response.json();
        if (isMounted) {
          setData(json);
          setBridgeReachable(true);
          setLastPollTime(new Date());
        }
      } catch (err: any) {
        if (err.name !== 'AbortError' && isMounted) {
          setBridgeReachable(false);
        }
      }
    };

    fetchBridgeData();
    const intervalId = setInterval(fetchBridgeData, 400);

    return () => {
      isMounted = false;
      controller.abort();
      clearInterval(intervalId);
    };
  }, []);

  const handleManualRefresh = async () => {
    setIsManualRefreshing(true);
    try {
      const res = await fetch('http://127.0.0.1:8765/data', { cache: 'no-store' });
      if (res.ok) {
        const json = await res.json();
        setData(json);
        setBridgeReachable(true);
        setLastPollTime(new Date());
      }
    } catch {
      setBridgeReachable(false);
    } finally {
      setTimeout(() => setIsManualRefreshing(false), 500);
    }
  };

  // Determine active telemetry source: direct bridge or backend serial bridge
  const hasBridge = bridgeReachable && data?.is_connected === true;
  const hasBackend = telemetry?.sensor?.is_connected === true;
  const isConnected = hasBridge || hasBackend;

  const channelLabel = hasBridge
    ? '127.0.0.1:8765 (Bridge)'
    : hasBackend
    ? 'Port 5000 (Backend Serial)'
    : 'STANDBY (COM9)';

  const tel = hasBridge
    ? (data?.telemetry || { ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0, pitch: 0, roll: 0, yaw: 0 })
    : hasBackend
    ? {
        ax: telemetry?.sensor?.ax ?? 0,
        ay: telemetry?.sensor?.ay ?? 0,
        az: telemetry?.sensor?.az ?? 0,
        gx: telemetry?.sensor?.gx ?? 0,
        gy: telemetry?.sensor?.gy ?? 0,
        gz: telemetry?.sensor?.gz ?? 0,
        pitch: telemetry?.sensor?.pitch ?? 0,
        roll: telemetry?.sensor?.roll ?? 0,
        yaw: telemetry?.sensor?.yaw ?? 0,
      }
    : { ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0, pitch: 0, roll: 0, yaw: 0 };

  const port = (hasBridge ? data?.port : telemetry?.sensor?.port) || 'COM9';
  const baud = data?.baud || 115200;
  const chipVerified = hasBridge ? (data?.chip_verified ?? false) : (telemetry?.sensor?.chip_verified ?? true);
  const chipId = (hasBridge ? data?.chip_id : telemetry?.sensor?.chip_id) || '0x70';
  const dataAge = hasBridge ? (data?.data_age_sec ?? 0) : (telemetry?.sensor?.data_age_sec ?? 0);
  const isStale = hasBridge ? (data?.is_stale ?? false) : dataAge > 2.0;
  const status = isConnected ? 'ONLINE' : 'OFFLINE';

  return (
    <div className="glass-panel rounded-xl p-5 border border-slate-800 flex flex-col justify-between shadow-xl">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between pb-3 border-b border-slate-800 mb-4 gap-2">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white tracking-wide uppercase font-sans">
                Physical ESP32 Telemetry Link
              </h2>
              <span className="text-[10px] bg-slate-800 text-cyan-300 font-mono px-2 py-0.5 rounded border border-slate-700">
                {channelLabel}
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono">
              Port: <span className="text-cyan-300">{port}</span> | {baud} baud | 5 Hz Telemetry
            </p>
          </div>
        </div>

        {/* Live Status Indicators */}
        <div className="flex items-center gap-2">
          <span
            className={`px-2.5 py-1 rounded text-xs font-mono font-medium flex items-center gap-1.5 border ${
              isConnected && chipVerified
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : isConnected
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                : isStale
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
            }`}
          >
            {isConnected ? (
              <Wifi className="w-3.5 h-3.5 animate-pulse text-emerald-400" />
            ) : (
              <WifiOff className="w-3.5 h-3.5 text-rose-400" />
            )}
            {status}
          </span>

          <button
            onClick={handleManualRefresh}
            title="Poll Bridge Now"
            className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors border border-slate-700"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isManualRefreshing ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Main Telemetry Readout Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* Accelerometer 3-Axis Box */}
        <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/80">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-cyan-400" /> Accelerometer (m/s²)
            </span>
            <span className="text-[11px] font-mono text-slate-400">±2g Dynamic Range</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800 text-center">
              <span className="text-[10px] text-slate-400 font-mono block">AX</span>
              <span className={`text-sm font-mono font-bold ${isConnected ? 'text-cyan-300' : 'text-slate-500'}`}>
                {tel.ax.toFixed(2)}
              </span>
            </div>
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800 text-center">
              <span className="text-[10px] text-slate-400 font-mono block">AY</span>
              <span className={`text-sm font-mono font-bold ${isConnected ? 'text-cyan-300' : 'text-slate-500'}`}>
                {tel.ay.toFixed(2)}
              </span>
            </div>
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800 text-center">
              <span className="text-[10px] text-slate-400 font-mono block">AZ (1g)</span>
              <span className={`text-sm font-mono font-bold ${isConnected ? 'text-emerald-300' : 'text-slate-500'}`}>
                {tel.az.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Gyroscope 3-Axis Box */}
        <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/80">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-amber-400" /> Gyroscope (°/s)
            </span>
            <span className="text-[11px] font-mono text-slate-400">Calibrated Offsets</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800 text-center">
              <span className="text-[10px] text-slate-400 font-mono block">GX (Roll)</span>
              <span className={`text-sm font-mono font-bold ${isConnected ? 'text-amber-300' : 'text-slate-500'}`}>
                {tel.gx.toFixed(1)}
              </span>
            </div>
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800 text-center">
              <span className="text-[10px] text-slate-400 font-mono block">GY (Pitch)</span>
              <span className={`text-sm font-mono font-bold ${isConnected ? 'text-amber-300' : 'text-slate-500'}`}>
                {tel.gy.toFixed(1)}
              </span>
            </div>
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800 text-center">
              <span className="text-[10px] text-slate-400 font-mono block">GZ (Yaw)</span>
              <span className={`text-sm font-mono font-bold ${isConnected ? 'text-amber-300' : 'text-slate-500'}`}>
                {tel.gz.toFixed(1)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Derived Attitude & Vehicle Stability Row */}
      <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs mb-3">
        <div className="flex items-center gap-3">
          <span className="text-slate-400 font-medium">Head Attitude:</span>
          <span className="font-mono text-slate-200">
            Pitch: <strong className={Math.abs(tel.pitch) > 20 ? 'text-rose-400' : 'text-cyan-300'}>{tel.pitch.toFixed(1)}°</strong>
          </span>
          <span className="font-mono text-slate-200">
            Roll: <strong className={Math.abs(tel.roll) > 20 ? 'text-rose-400' : 'text-cyan-300'}>{tel.roll.toFixed(1)}°</strong>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Sensor Verification:</span>
          <span className={`font-mono px-2 py-0.5 rounded text-[11px] font-semibold border ${
            chipVerified
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
          }`}>
            {chipVerified ? `VERIFIED (${chipId})` : `UNVERIFIED (${chipId})`}
          </span>
        </div>
      </div>

      {/* Error or Disconnected Warning Banner */}
      {!isConnected && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-2.5 mb-3">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 text-rose-400 mt-0.5" />
          <div className="space-y-1">
            <p className="font-medium">
              Hardware disconnected on {port}. Connect ESP32 via USB.
            </p>
            <p className="text-[11px] text-rose-300/80 font-mono">
              Run: <code className="bg-slate-900 px-1.5 py-0.5 rounded text-amber-300">py bridge.py --port COM9</code>
            </p>
          </div>
        </div>
      )}

      {/* Footer Audit Strip */}
      <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-400 font-mono pt-2 border-t border-slate-800/80 gap-2">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <Radio className="w-3 h-3 text-cyan-400" />
            Packets: <span className="text-slate-200">{data?.packets_received ?? 0}</span>
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3 text-slate-400" />
            Data Age: <span className="text-slate-200">{data?.data_age_sec?.toFixed(1) ?? '---'}s</span>
          </span>
        </div>
        <div className="text-slate-400">
          Last Poll: {lastPollTime ? lastPollTime.toLocaleTimeString() : 'Waiting for bridge...'}
        </div>
      </div>
    </div>
  );
};

export default HardwareSensorPanel;
