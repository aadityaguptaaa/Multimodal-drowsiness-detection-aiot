import { io, Socket } from 'socket.io-client';
import { TelemetryData, EventItem } from './api';

const SOCKET_URL = 'http://localhost:5000';

class SocketService {
  private socket: Socket | null = null;

  connect(
    onTelemetry: (data: TelemetryData) => void,
    onNewEvent: (event: EventItem) => void,
    onConnectChange: (connected: boolean) => void
  ) {
    if (this.socket) {
      this.socket.disconnect();
    }

    this.socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
    });

    this.socket.on('connect', () => {
      onConnectChange(true);
    });

    this.socket.on('disconnect', () => {
      onConnectChange(false);
    });

    this.socket.on('telemetry_update', (data: TelemetryData) => {
      onTelemetry(data);
    });

    this.socket.on('new_event', (event: EventItem) => {
      onNewEvent(event);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const socketService = new SocketService();
