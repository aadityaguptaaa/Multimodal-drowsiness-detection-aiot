/**
 * Web Audio API Alert Synthesizer for Driver Monitoring System
 * Generates automotive warning and critical danger buzzer sounds directly in the browser.
 */

class WebAudioAlertService {
  private audioCtx: AudioContext | null = null;
  private lastAlertTime = 0;
  private cooldownMs = 2500;

  private getAudioContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    if (!this.audioCtx) {
      const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtxClass) {
        this.audioCtx = new AudioCtxClass();
      }
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
    return this.audioCtx;
  }

  public playAlert(level: 'WARNING' | 'CRITICAL', force = false) {
    const now = Date.now();
    if (!force && now - this.lastAlertTime < this.cooldownMs) {
      return;
    }
    this.lastAlertTime = now;

    const ctx = this.getAudioContext();
    if (!ctx) return;

    if (level === 'CRITICAL') {
      this.playCriticalPulsingBeep(ctx);
    } else {
      this.playWarningChime(ctx);
    }
  }

  private playCriticalPulsingBeep(ctx: AudioContext) {
    // 3 rapid emergency pulses: 1200 Hz
    const pulseCount = 3;
    const pulseDuration = 0.12;
    const gap = 0.08;

    for (let i = 0; i < pulseCount; i++) {
      const startTime = ctx.currentTime + i * (pulseDuration + gap);
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(1200, startTime);
      osc.frequency.exponentialRampToValueAtTime(800, startTime + pulseDuration);

      gain.gain.setValueAtTime(0.3, startTime);
      gain.gain.exponentialRampToValueAtTime(0.01, startTime + pulseDuration);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(startTime);
      osc.stop(startTime + pulseDuration);
    }
  }

  private playWarningChime(ctx: AudioContext) {
    // 2-tone melodic warning chime: 587 Hz (D5) -> 880 Hz (A5)
    const startTime = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(587.33, startTime);
    osc.frequency.setValueAtTime(880.0, startTime + 0.15);

    gain.gain.setValueAtTime(0.25, startTime);
    gain.gain.exponentialRampToValueAtTime(0.01, startTime + 0.35);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(startTime);
    osc.stop(startTime + 0.35);
  }
}

export const webAudioService = new WebAudioAlertService();
