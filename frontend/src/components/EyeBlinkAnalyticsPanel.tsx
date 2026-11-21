import React, { useState, useEffect } from 'react';
import { Eye, Activity, TrendingUp, BarChart3, Clock } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
import { TelemetryData } from '../services/api';

interface EyeBlinkAnalyticsPanelProps {
  telemetry: TelemetryData | null;
}

interface ChartSample {
  time: string;
  ear: number;
  perclos: number;
}

export const EyeBlinkAnalyticsPanel: React.FC<EyeBlinkAnalyticsPanelProps> = ({ telemetry }) => {
  const [chartData, setChartData] = useState<ChartSample[]>([]);

  const ear = telemetry?.ear;
  const perclos = telemetry?.perclos;
  const blinks = telemetry?.blinks;

  useEffect(() => {
    if (telemetry && telemetry.status === 'RUNNING') {
      const nowStr = new Date().toLocaleTimeString().split(' ')[0];
      setChartData((prev) => {
        const next = [
          ...prev,
          {
            time: nowStr,
            ear: ear?.valid ? Number((ear.avg_ear * 100).toFixed(1)) : 0,
            perclos: perclos?.valid_observation ? Number(perclos.perclos_percent.toFixed(1)) : 0,
          },
        ];
        return next.slice(-30);
      });
    }
  }, [telemetry?.timestamp]);

  const perclosPercent = perclos?.perclos_percent || 0;
  const coveragePercent = Math.round((perclos?.observation_coverage || 0) * 100);

  const getPerclosColor = (pct: number) => {
    if (pct >= 25) return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
    if (pct >= 15) return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
    return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
  };

  return (
    <div className="glass-panel rounded-xl p-5 border border-slate-800 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
        <div className="flex items-center gap-2">
          <Eye className="w-5 h-5 text-emerald-400" />
          <h2 className="text-sm font-bold text-white tracking-wide uppercase font-sans">
            Eye Aperture & PERCLOS Dynamics
          </h2>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
          NHTSA / ISO 15007 Metric
        </span>
      </div>

      {/* Grid of 4 Key Analytics Cards */}
      <div className="grid grid-cols-2 gap-3 mb-3.5">
        {/* Card 1: Eye Aspect Ratio (EAR) */}
        <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-mono text-[11px]">Aperture (EAR)</span>
            <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
              ear?.is_closed ? 'bg-rose-950 text-rose-400 border border-rose-800' : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
            }`}>
              {ear?.is_closed ? 'CLOSED' : 'OPEN'}
            </span>
          </div>
          <div className="text-2xl font-bold font-mono text-white glow-cyan my-0.5">
            {ear?.valid ? ear.avg_ear.toFixed(3) : '0.000'}
          </div>
          <div className="text-[10px] text-slate-500 font-mono">
            Baseline: 0.220 | L: {ear?.left_ear || 0} R: {ear?.right_ear || 0}
          </div>
        </div>

        {/* Card 2: Rolling PERCLOS */}
        <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-mono text-[11px]">Rolling PERCLOS (10s)</span>
            <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${getPerclosColor(perclosPercent)}`}>
              {perclosPercent >= 25 ? 'CRITICAL' : (perclosPercent >= 15 ? 'ELEVATED' : 'NOMINAL')}
            </span>
          </div>
          <div className="text-2xl font-bold font-mono text-white my-0.5">
            {perclos?.valid_observation ? `${perclosPercent}%` : '—'}
          </div>
          <div className="text-[10px] text-slate-500 font-mono flex justify-between">
            <span>Coverage: {coveragePercent}%</span>
            <span>Closed: {perclos?.closed_duration_sec ?? 0}s / {perclos?.window_duration_sec ?? 10}s</span>
          </div>
          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
            Sustained: <span className={(perclos?.sustained_closure_sec ?? 0) >= 1.5 ? 'text-rose-400 font-bold' : 'text-slate-300'}>
              {perclos?.sustained_closure_sec?.toFixed(1) ?? '0.0'}s
            </span> (Threshold: {perclos?.sustained_closure_threshold_sec ?? 1.5}s)
          </div>
        </div>

        {/* Card 3: Blink Frequency (BPM) */}
        <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-mono text-[11px]">Blink Frequency</span>
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white my-0.5">
            {blinks?.blinks_per_minute || 0} <span className="text-xs text-slate-500 font-normal">BPM</span>
          </div>
          <div className="text-[10px] text-slate-500 font-mono">
            Blink Count: {blinks?.total_blinks || 0} | Avg Duration: {blinks?.average_duration_ms || 0} ms
          </div>
        </div>

        {/* Card 4: Blink Trend Indicator */}
        <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-mono text-[11px]">Eyelid Dynamics Trend</span>
            <TrendingUp className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div className="text-xs font-semibold font-mono text-cyan-300 truncate my-1">
            {blinks?.blink_trend || 'CALIBRATED NORMAL'}
          </div>
          <div className="text-[10px] text-slate-500 font-mono">
            Fatigue Watchdog: Active (Window: 60s)
          </div>
        </div>
      </div>

      {/* Industrial Recharts Area Stream */}
      <div className="pt-2 border-t border-slate-800/80">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5 font-mono">
          <span className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-300">
            <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
            Temporal Eye Dynamics Waveform (Last 30 Observations)
          </span>
          <div className="flex items-center gap-3 text-[10px]">
            <span className="flex items-center gap-1 text-cyan-400">
              <span className="w-2 h-0.5 bg-cyan-400"></span> EAR (Normalized)
            </span>
            <span className="flex items-center gap-1 text-rose-400">
              <span className="w-2 h-0.5 bg-rose-400"></span> PERCLOS Closure %
            </span>
          </div>
        </div>

        <div className="h-28 w-full bg-slate-950/80 rounded-xl p-1.5 border border-slate-800">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorEar" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorPerclos" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="time" hide />
              <YAxis hide domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0b1120',
                  borderColor: '#1e293b',
                  fontSize: '11px',
                  borderRadius: '6px',
                  fontFamily: 'monospace'
                }}
              />
              <Area
                type="monotone"
                dataKey="ear"
                name="EAR x100"
                stroke="#06b6d4"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorEar)"
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="perclos"
                name="PERCLOS %"
                stroke="#f43f5e"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorPerclos)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
