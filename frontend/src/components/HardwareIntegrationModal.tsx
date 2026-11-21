import React, { useState, useEffect } from 'react';
import { X, Wrench, Cpu, Eye, Radio, Volume2, DollarSign, ShieldCheck, RefreshCw, CheckCircle2, AlertTriangle, Terminal } from 'lucide-react';
import { api } from '../services/api';

interface HardwareIntegrationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HardwareIntegrationModal: React.FC<HardwareIntegrationModalProps> = ({ isOpen, onClose }) => {
  const [ports, setPorts] = useState<any[]>([]);
  const [selectedPort, setSelectedPort] = useState<string>('COM9');
  const [loadingPorts, setLoadingPorts] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<string>('');

  useEffect(() => {
    if (isOpen) {
      loadPorts();
    }
  }, [isOpen]);

  const loadPorts = async () => {
    setLoadingPorts(true);
    try {
      const res = await api.getAvailablePorts();
      setPorts(res.ports || []);
      if (res.default_port) setSelectedPort(res.default_port);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingPorts(false);
    }
  };

  const handleApplyPort = async () => {
    try {
      const res = await api.configureSensorPort(selectedPort, 115200);
      setStatusMsg(`Configured ${res.configured_port} @ ${res.configured_baud} baud. Connected: ${res.connected}`);
      setTimeout(() => setStatusMsg(''), 4000);
    } catch (e) {
      setStatusMsg('Failed to configure serial port.');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
      <div className="glass-panel w-full max-w-3xl rounded-2xl border border-slate-700 bg-navy-950 p-6 shadow-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400">
              <Wrench className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white font-sans">
                ESP32 Hardware Integration & Telemetry Console
              </h2>
              <p className="text-xs text-slate-400">
                Physical Sensor Wiring, Serial COM9 Bridge, and Hardware-in-the-Loop Architecture
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="mt-5 space-y-4 text-xs">
          {/* Active USB Serial Port Selector & Diagnostics */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-white font-mono flex items-center gap-2 text-xs">
                <Terminal className="w-4 h-4 text-cyan-400" />
                Physical USB Serial Bridge Configuration (Windows COM9 / 115200 Baud)
              </span>
              <button
                onClick={loadPorts}
                disabled={loadingPorts}
                className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono flex items-center gap-1.5"
              >
                <RefreshCw className={`w-3 h-3 text-cyan-400 ${loadingPorts ? 'animate-spin' : ''}`} />
                <span>Scan Host Ports</span>
              </button>
            </div>

            <div className="flex items-center gap-2">
              <div className="flex-1">
                <label className="text-[10px] font-mono text-slate-400 block mb-1">
                  Active Port (Default: COM9, 115200 Baud, 8-N-1):
                </label>
                <select
                  value={selectedPort}
                  onChange={(e) => setSelectedPort(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-white"
                >
                  <option value="COM9">COM9 (ESP32 Default Windows Port)</option>
                  {ports.map((p, i) => (
                    <option key={i} value={p.port}>
                      {p.port} - {p.description}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleApplyPort}
                className="mt-4 px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-mono font-bold text-xs"
              >
                Connect Port
              </button>
            </div>

            {statusMsg && (
              <div className="p-2 rounded bg-cyan-950/60 border border-cyan-800 text-cyan-300 font-mono text-[11px]">
                {statusMsg}
              </div>
            )}
          </div>

          {/* Physical Wiring & Pinout Guide */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <h3 className="font-bold text-white font-mono text-xs flex items-center gap-2">
              <Radio className="w-4 h-4 text-emerald-400" />
              Verified ESP32 to MPU-6050 Wiring Pinout
            </h3>

            <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
              <div className="p-2.5 rounded bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold block">Power Connections</span>
                <div className="text-slate-300">ESP32 3V3 ───► MPU-6050 VCC (3.3V)</div>
                <div className="text-slate-300">ESP32 GND ───► MPU-6050 GND</div>
                <div className="text-slate-300">ESP32 GND ───► MPU-6050 AD0 (Sets 0x68)</div>
              </div>

              <div className="p-2.5 rounded bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-emerald-400 font-bold block">I2C Data Bus (Fast 400kHz)</span>
                <div className="text-slate-300">ESP32 GPIO 21 ───► MPU-6050 SDA</div>
                <div className="text-slate-300">ESP32 GPIO 22 ───► MPU-6050 SCL</div>
                <div className="text-slate-400 text-[10px]">WHO_AM_I Register: 0x75 verifies chip ID 0x68</div>
              </div>
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed pt-1">
              Firmware sketch location: <code className="text-cyan-300 font-mono">firmware/esp32_imu_firmware/esp32_imu_firmware.ino</code>. Flashed via Arduino IDE with ESP32 board support. Emits validated JSON telemetry lines over USB serial at 115200 baud.
            </p>
          </div>

          {/* Component Roadmap Grid */}
          <div className="grid grid-cols-2 gap-3">
            {/* Component 1: Optical Camera */}
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
              <div className="flex items-center gap-2 text-cyan-400 font-bold">
                <Eye className="w-4 h-4" />
                <span className="font-sans">Optical Driver Vision (Primary)</span>
              </div>
              <p className="text-slate-400 text-[11px]">
                Webcam or 850nm USB Infrared Camera. Tracks 478 MediaPipe facial landmarks, extracts eye regions, and computes Rolling PERCLOS over continuous timestamps.
              </p>
              <div className="text-[10px] font-mono text-slate-500">Pipeline: 64x64 Eye Crop CNN & EAR Geometry</div>
            </div>

            {/* Component 2: MPU6050 IMU */}
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
              <div className="flex items-center gap-2 text-emerald-400 font-bold">
                <Radio className="w-4 h-4" />
                <span className="font-sans">MPU-6050 IMU (Secondary Motion)</span>
              </div>
              <p className="text-slate-400 text-[11px]">
                Raw 3-axis accelerometer (ax, ay, az) and 3-axis gyroscope (gx, gy, gz). Detects forward nodding (pitch &le; -15&deg;) and lateral slumping (roll &ge; 25&deg;).
              </p>
              <div className="text-[10px] font-mono text-slate-500">Bridge: pyserial background listener with stale timeout</div>
            </div>

            {/* Component 3: Embedded Compute Board */}
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
              <div className="flex items-center gap-2 text-purple-400 font-bold">
                <Cpu className="w-4 h-4" />
                <span className="font-sans">Edge Telematics Gateway</span>
              </div>
              <p className="text-slate-400 text-[11px]">
                ESP32 dual-core co-processor handles hardware sensor sampling at 50Hz, transmitting serialized telemetry frames to host PC or Raspberry Pi 4 edge node.
              </p>
              <div className="text-[10px] font-mono text-slate-500">Runtime: FreeRTOS / Arduino C++</div>
            </div>

            {/* Component 4: Piezo Buzzer & Power */}
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
              <div className="flex items-center gap-2 text-rose-400 font-bold">
                <Volume2 className="w-4 h-4" />
                <span className="font-sans">Acoustic Transducer & Alarm</span>
              </div>
              <p className="text-slate-400 text-[11px]">
                Active piezo buzzer with cooldown protection (3.5s cooldown guard) preventing alarm fatigue. Dual tone: 1200Hz fatigue warning, 2200Hz critical intervention.
              </p>
              <div className="text-[10px] font-mono text-slate-500">Actuation: Cooldown-gated trigger</div>
            </div>
          </div>

          {/* Target Student Budget Breakdown (₹3,000–₹4,000) */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2.5">
            <h3 className="font-bold text-slate-200 uppercase tracking-wider text-[11px] flex items-center gap-1.5 font-mono">
              <DollarSign className="w-4 h-4 text-emerald-400" />
              Target Hardware Bill of Materials (₹3,000–₹4,000 Student Budget)
            </h3>

            <div className="space-y-1.5 font-mono text-[11px]">
              <div className="flex justify-between py-1 border-b border-slate-800/80">
                <span className="text-slate-400">1. ESP32 NodeMCU Development Board (CP2102 / CH340)</span>
                <span className="text-white font-bold">₹350 – ₹450</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/80">
                <span className="text-slate-400">2. MPU-6050 6-Axis Accelerometer & Gyroscope Module</span>
                <span className="text-white font-bold">₹150 – ₹200</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/80">
                <span className="text-slate-400">3. USB Active Infrared Night-Vision Camera with IR LEDs</span>
                <span className="text-white font-bold">₹1,200 – ₹1,500</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/80">
                <span className="text-slate-400">4. Acoustic Buzzer, LM2596 Step-Down Buck Converter, Enclosure</span>
                <span className="text-white font-bold">₹350 – ₹500</span>
              </div>
              <div className="flex justify-between pt-1.5 font-bold text-xs text-cyan-400">
                <span>Total Target Bill of Materials (BOM)</span>
                <span>₹2,050 – ₹2,650 (Well within ₹3,000–₹4,000 budget)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold"
          >
            Close Hardware Console
          </button>
        </div>
      </div>
    </div>
  );
};
