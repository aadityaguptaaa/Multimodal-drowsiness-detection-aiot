/**
 * REST API Client for AIoT Driver Drowsiness Detection System
 */

const BASE_URL = 'http://localhost:5000';

export interface HealthResponse {
  status: string;
  service: string;
  python_version: string;
  monitoring_running: boolean;
  model_status: string;
  timestamp: number;
}

export interface ModelStatusResponse {
  model_status: string;
  model_file_exists: boolean;
  model_path: string;
  model_size_mb: number;
  metadata: {
    model_architecture?: string;
    total_dataset_images?: number;
    train_samples?: number;
    val_samples?: number;
    test_samples?: number;
    training_mode?: string;
    created_at?: string;
    epochs_trained?: number;
  };
  load_error?: string;
}

export interface EvaluationReportResponse {
  status: string;
  accuracy: number;
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
  model_size_mb: number;
  test_samples_count?: number;
  per_class_metrics: {
    [key: string]: {
      precision: number;
      recall: number;
      f1_score: number;
      test_count: number;
    };
  };
  confusion_matrix: {
    true_negative_alert_correct: number;
    false_positive_alert_wrong: number;
    false_negative_drowsy_missed: number;
    true_positive_drowsy_caught: number;
  };
  latency_benchmark: {
    average_latency_ms: number;
    approximate_fps: number;
  };
  safety_failure_analysis: {
    false_negative_safety_impact: string;
    false_positive_convenience_impact: string;
  };
}

export interface TelemetryData {
  status: string;
  camera_available: boolean;
  model_status: string;
  eye_model_status?: string;
  fps: number;
  total_frame_latency_ms: number;
  face_detected: boolean;
  eyes_detected?: boolean;
  eye_state?: {
    state: string;
    confidence: number;
    closed_probability?: number;
    is_closed: boolean;
    is_valid: boolean;
    model_ready: boolean;
    left_eye?: any;
    right_eye?: any;
  };
  ear: {
    avg_ear: number;
    left_ear?: number;
    right_ear?: number;
    is_closed: boolean;
    valid: boolean;
    threshold_used?: number;
  };
  perclos: {
    perclos_ratio: number;
    perclos_percent: number;
    observation_coverage: number;
    coverage_percent?: number;
    is_drowsy: boolean;
    is_warning: boolean;
    valid_observation: boolean;
    is_sustained_closure?: boolean;
    sustained_closure_sec?: number;
    sustained_closure_threshold_sec?: number;
    window_duration_sec: number;
    valid_duration_sec?: number;
    closed_duration_sec?: number;
    open_duration_sec?: number;
    status_text: string;
  };
  blinks: {
    total_blinks: number;
    blinks_per_minute: number;
    average_duration_ms: number;
    blink_trend: string;
  };
  sensor: {
    pitch: number;
    roll: number;
    yaw: number;
    ax?: number;
    ay?: number;
    az?: number;
    gx?: number;
    gy?: number;
    gz?: number;
    is_valid: boolean;
    is_connected?: boolean;
    mode: string;
    port?: string;
    chip_verified?: boolean;
    chip_id?: string;
    state_description: string;
    error_message?: string;
    data_age_sec?: number;
  };
  sensor_mode: string;
  cnn: {
    model_loaded: boolean;
    probability: number;
    smoothed_probability: number;
    predicted_class: string;
    is_valid: boolean;
    latency_ms: number;
  };
  fusion: {
    overall_state: 'NORMAL' | 'WARNING' | 'CRITICAL';
    composite_score: number;
    contributing_factors: string[];
    confidence_level: string;
    readiness?: {
      camera_operational: boolean;
      face_detected: boolean;
      eyes_detected: boolean;
      eye_model_loaded: boolean;
      perclos_valid: boolean;
      esp32_connected: boolean;
      motion_sensor_operational: boolean;
      overall_detection_readiness: string;
      fusion_methodology?: string;
    };
  };
  alert: {
    enabled: boolean;
    in_cooldown: boolean;
    cooldown_remaining_sec: number;
    last_alert_level: string;
  };
  timestamp: number;
}

export interface EventItem {
  id: number;
  timestamp: string;
  epoch: number;
  level: 'INFO' | 'WARNING' | 'CRITICAL';
  message: string;
  score: number;
}

export const api = {
  getHealth: async (): Promise<HealthResponse> => {
    const res = await fetch(`${BASE_URL}/api/health`);
    return res.json();
  },

  getDatasetReport: async () => {
    const res = await fetch(`${BASE_URL}/api/dataset/inspect`);
    if (!res.ok) throw new Error('Dataset not inspected');
    return res.json();
  },

  getModelStatus: async (): Promise<ModelStatusResponse> => {
    const res = await fetch(`${BASE_URL}/api/model/status`);
    return res.json();
  },

  getEvaluationReport: async (): Promise<EvaluationReportResponse> => {
    const res = await fetch(`${BASE_URL}/api/model/evaluate`);
    if (!res.ok) throw new Error('Model not evaluated yet');
    return res.json();
  },

  startMonitoring: async () => {
    const res = await fetch(`${BASE_URL}/api/monitoring/start`, { method: 'POST' });
    return res.json();
  },

  stopMonitoring: async () => {
    const res = await fetch(`${BASE_URL}/api/monitoring/stop`, { method: 'POST' });
    return res.json();
  },

  getTelemetry: async (): Promise<TelemetryData> => {
    const res = await fetch(`${BASE_URL}/api/monitoring/telemetry`);
    return res.json();
  },

  setSensorMode: async (mode: 'SIMULATED' | 'REAL') => {
    const res = await fetch(`${BASE_URL}/api/sensors/mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    return res.json();
  },

  simulateSensorState: async (state: string, pitch?: number, roll?: number) => {
    const res = await fetch(`${BASE_URL}/api/sensors/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state, pitch, roll })
    });
    return res.json();
  },

  toggleAlert: async (enabled: boolean) => {
    const res = await fetch(`${BASE_URL}/api/alert/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled })
    });
    return res.json();
  },

  testAlert: async () => {
    const res = await fetch(`${BASE_URL}/api/alert/test`, { method: 'POST' });
    return res.json();
  },

  getHistory: async (): Promise<{ events: EventItem[] }> => {
    const res = await fetch(`${BASE_URL}/api/history`);
    return res.json();
  },

  clearHistory: async () => {
    const res = await fetch(`${BASE_URL}/api/history/clear`, { method: 'POST' });
    return res.json();
  },

  getEyeModelStatus: async () => {
    const res = await fetch(`${BASE_URL}/api/model/eye/status`);
    return res.json();
  },

  getEyeEvaluationReport: async () => {
    const res = await fetch(`${BASE_URL}/api/model/eye/evaluate`);
    if (!res.ok) throw new Error('Eye model evaluation report not available');
    return res.json();
  },

  getSensorStatus: async () => {
    const res = await fetch(`${BASE_URL}/api/sensors/status`);
    return res.json();
  },

  getAvailablePorts: async () => {
    const res = await fetch(`${BASE_URL}/api/sensors/ports`);
    return res.json();
  },

  configureSensorPort: async (port: string, baudrate: number = 115200) => {
    const res = await fetch(`${BASE_URL}/api/sensors/configure`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ port, baudrate })
    });
    return res.json();
  },

  getStreamUrl: () => `${BASE_URL}/api/camera/stream`
};
