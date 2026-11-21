import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LiveCameraPanel } from './components/LiveCameraPanel';
import { EyeBlinkAnalyticsPanel } from './components/EyeBlinkAnalyticsPanel';
import { SensorFusionPanel } from './components/SensorFusionPanel';
import { MPU6050Panel } from './components/MPU6050Panel';
import { EventLogPanel } from './components/EventLogPanel';
import { AIModelInfoModal } from './components/AIModelInfoModal';
import { DatasetReportModal } from './components/DatasetReportModal';
import { HardwareIntegrationModal } from './components/HardwareIntegrationModal';
import { HardwareSensorPanel } from './components/HardwareSensorPanel';
import { api, TelemetryData, EventItem } from './services/api';
import { socketService } from './services/socket';
import { webAudioService } from './services/sound';

export const App: React.FC = () => {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const [audioEnabled, setAudioEnabled] = useState<boolean>(true);

  // Modals state
  const [showModelModal, setShowModelModal] = useState<boolean>(false);
  const [showDatasetModal, setShowDatasetModal] = useState<boolean>(false);
  const [showHardwareModal, setShowHardwareModal] = useState<boolean>(false);

  // Initial fetch and health check
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const health = await api.getHealth();
        setBackendConnected(health.status === 'HEALTHY');
        const initialTelemetry = await api.getTelemetry();
        setTelemetry(initialTelemetry);
        setAudioEnabled(initialTelemetry.alert?.enabled ?? true);
        const history = await api.getHistory();
        setEvents(history.events || []);
      } catch (err) {
        setBackendConnected(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 4000);

    // Setup Socket.IO for real-time live streaming (~6 Hz)
    socketService.connect(
      (data: TelemetryData) => {
        setTelemetry(data);
        setBackendConnected(true);
        if (data.fusion?.overall_state === 'CRITICAL' || data.fusion?.overall_state === 'WARNING') {
          webAudioService.playAlert(data.fusion.overall_state);
        }
      },
      (newEvent: EventItem) => {
        setEvents((prev) => [newEvent, ...prev.slice(0, 99)]);
        if (newEvent.level === 'CRITICAL' || newEvent.level === 'WARNING') {
          webAudioService.playAlert(newEvent.level);
        }
      },
      (connected: boolean) => {
        setBackendConnected(connected);
      }
    );

    // Sync physical ESP32 bridge (127.0.0.1:8765) directly into telemetry state for all top panels
    const bridgeSyncInterval = setInterval(async () => {
      try {
        const res = await fetch('http://127.0.0.1:8765/data', { cache: 'no-store' });
        if (res.ok) {
          const bdata = await res.json();
          if (bdata && bdata.is_connected) {
            setTelemetry((prev) => {
              const baseSensor = prev?.sensor || {
                pitch: 0,
                roll: 0,
                yaw: 0,
                is_valid: true,
                is_connected: true,
                mode: 'REAL',
                state_description: 'PHYSICAL_ESP32_LINK_ACTIVE'
              };

              const updatedSensor = {
                ...baseSensor,
                pitch: bdata.telemetry?.pitch ?? 0,
                roll: bdata.telemetry?.roll ?? 0,
                yaw: bdata.telemetry?.yaw ?? 0,
                ax: bdata.telemetry?.ax ?? 0,
                ay: bdata.telemetry?.ay ?? 0,
                az: bdata.telemetry?.az ?? 0,
                gx: bdata.telemetry?.gx ?? 0,
                gy: bdata.telemetry?.gy ?? 0,
                gz: bdata.telemetry?.gz ?? 0,
                is_connected: true,
                is_valid: !bdata.is_stale,
                chip_verified: bdata.chip_verified ?? true,
                chip_id: bdata.chip_id ?? '0x68',
                port: bdata.port ?? 'COM9',
                data_age_sec: bdata.data_age_sec ?? 0,
                mode: 'REAL',
                state_description: 'PHYSICAL_ESP32_LINK_ACTIVE'
              };

              if (!prev) return null;

              return {
                ...prev,
                sensor: updatedSensor,
                fusion: prev.fusion && prev.fusion.readiness ? {
                  ...prev.fusion,
                  readiness: {
                    ...prev.fusion.readiness,
                    esp32_connected: true,
                    motion_sensor_operational: bdata.chip_verified ?? true
                  }
                } : prev.fusion
              };
            });
          }
        }
      } catch {
        // Standalone bridge not active, rely on Flask socket
      }
    }, 400);

    return () => {
      clearInterval(interval);
      clearInterval(bridgeSyncInterval);
      socketService.disconnect();
    };
  }, []);

  const handleToggleMonitoring = async () => {
    try {
      if (telemetry?.status === 'RUNNING') {
        await api.stopMonitoring();
      } else {
        await api.startMonitoring();
      }
      const updated = await api.getTelemetry();
      setTelemetry(updated);
    } catch (e) {
      console.error(e);
    }
  };

  const handleToggleAudio = async () => {
    const nextState = !audioEnabled;
    setAudioEnabled(nextState);
    try {
      await api.toggleAlert(nextState);
    } catch (e) {
      console.error(e);
    }
  };

  const handleClearHistory = async () => {
    try {
      await api.clearHistory();
      setEvents([]);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Top Navigation & System Overview */}
      <Navbar
        telemetry={telemetry}
        backendConnected={backendConnected}
        onToggleMonitoring={handleToggleMonitoring}
        onOpenModelModal={() => setShowModelModal(true)}
        onOpenDatasetModal={() => setShowDatasetModal(true)}
        onOpenHardwareModal={() => setShowHardwareModal(true)}
        onToggleAudio={handleToggleAudio}
        audioEnabled={audioEnabled}
      />

      {/* Main Dashboard Grid */}
      <main className="flex-1 p-5 max-w-7xl mx-auto w-full space-y-5">
        {/* Top Row: Visual Pipeline (Live Camera + Eye/Blink Analytics) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <LiveCameraPanel
            telemetry={telemetry}
            onToggleMonitoring={handleToggleMonitoring}
          />
          <EyeBlinkAnalyticsPanel telemetry={telemetry} />
        </div>

        {/* Bottom Row: Sensor Fusion + MPU6050 + Alert Event Audit Trail */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          <SensorFusionPanel telemetry={telemetry} />
          <MPU6050Panel telemetry={telemetry} />
          <EventLogPanel
            events={events}
            audioEnabled={audioEnabled}
            onToggleAudio={handleToggleAudio}
            onClearHistory={handleClearHistory}
            cooldownRemaining={telemetry?.alert?.cooldown_remaining_sec || 0}
          />
        </div>

        {/* Physical Hardware Bridge Direct Telemetry Channel */}
        <HardwareSensorPanel telemetry={telemetry} />
      </main>

      {/* Footer */}
      <footer className="glass-panel border-t border-slate-900 px-6 py-3 text-center text-xs text-slate-500 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
          <span className="font-medium text-slate-400">Automotive Driver Monitoring System (DMS) | AIoT Edge Telematics Platform</span>
        </div>
        <div className="font-mono text-[11px] text-slate-400">
          MobileNetV2 Transfer Learning | PERCLOS ML Model | MPU-6050 6-DOF IMU
        </div>
      </footer>

      {/* Info Modals */}
      <AIModelInfoModal
        isOpen={showModelModal}
        onClose={() => setShowModelModal(false)}
      />
      <DatasetReportModal
        isOpen={showDatasetModal}
        onClose={() => setShowDatasetModal(false)}
      />
      <HardwareIntegrationModal
        isOpen={showHardwareModal}
        onClose={() => setShowHardwareModal(false)}
      />
    </div>
  );
};

export default App;
