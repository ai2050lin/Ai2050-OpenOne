import { API_BASE } from '../config.js';
export { API_BASE as API_BASE };

export const toFiniteNumber = (value, fallback = 0) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

export const parsePercentToRatio = (raw) => {
  if (raw == null) return 0;
  if (typeof raw === 'number') return raw > 1 ? raw / 100 : raw;
  const text = String(raw).trim();
  const normalized = text.endsWith('%') ? text.slice(0, -1) : text;
  const numeric = Number(normalized);
  if (!Number.isFinite(numeric)) return 0;
  return numeric / 100;
};

export const mapLegacyConsciousField = (payload) => {
  const spectrum = payload?.unified_spectrum || payload || {};
  const emotion = spectrum?.emotion || {};
  const stability = toFiniteNumber(emotion?.stability ?? spectrum?.stability, 0);
  const signalNorm = toFiniteNumber(spectrum?.signal_norm ?? spectrum?.gws_intensity, 0);
  const memorySlots = toFiniteNumber(spectrum?.memory_slots ?? spectrum?.memory_load, 0);
  const energySaving = parsePercentToRatio(spectrum?.energy_saving);
  const resonance = toFiniteNumber(spectrum?.resonance ?? emotion?.energy, 0);
  const resonanceRate = resonance > 1 ? resonance / 100 : resonance;

  return {
    ...spectrum,
    stability,
    gws_intensity: signalNorm,
    memory_load: memorySlots,
    energy_saving: energySaving,
    resonance: resonanceRate,
    glow_color: spectrum?.glow_color || (stability > 0.6 ? 'amber' : 'indigo'),
  };
};

export const mapRuntimeConsciousField = (events) => {
  const activation = events.find((e) => e?.event_type === 'ActivationSnapshot');
  const alignment = events.find((e) => e?.event_type === 'AlignmentSignal');
  if (!activation || !alignment) return null;

  const emotion = alignment?.payload?.emotion || {};
  const stability = toFiniteNumber(emotion?.stability, 0);
  const signalNorm = toFiniteNumber(activation?.payload?.signal_norm, 0);
  const memorySlots = toFiniteNumber(activation?.payload?.memory_slots, 0);
  const energySavingPct = toFiniteNumber(alignment?.payload?.energy_saving_pct, 0);
  const resonance = toFiniteNumber(emotion?.energy, 0);

  return {
    stability,
    gws_intensity: signalNorm,
    memory_load: memorySlots,
    energy_saving: energySavingPct / 100,
    resonance: resonance > 1 ? resonance / 100 : resonance,
    glow_color: stability > 0.6 ? 'amber' : 'indigo',
    winner_module: alignment?.payload?.winner_module || null,
    source: 'runtime-v1',
  };
};
