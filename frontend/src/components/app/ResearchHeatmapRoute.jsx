import { Text } from '@react-three/drei';
import { BarChart3, Info } from 'lucide-react';

import {
  C101_ACTIVATION_COORDINATE_HEATMAP_ROUTE,
  C102_COORDINATE_BARCODE_HEATMAP_ROUTE,
  C104_UPSTREAM_ROLE_BARCODE_HEATMAP_ROUTE,
  C109_ROLE_STATE_FIELD_ATLAS_ROUTE,
  C157_C166_LOCAL_FIELD_HEATMAP_ROUTE,
  C167_C168_RELATION_RESIDUAL_HEATMAP_ROUTE,
  C170_ROLE_CHECKPOINT_HEATMAP_ROUTE,
  C183_NATURAL_RESPONSE_ECOLOGY_HEATMAP_ROUTE,
  C189_NEW_MATERIAL_RESPONSE_SCAFFOLD_HEATMAP_ROUTE,
  C191_RESPONSE_EQUIVALENCE_ATLAS_ROUTE,
  C193_PROGRAM_CENTERED_RESIDUAL_HEATMAP_ROUTE,
  C202_SIGNED_OPERATOR_CAMPAIGN_HEATMAP_ROUTE,
  C215_RESPONSE_INTERVAL_COMPOSITION_ATLAS_ROUTE,
  C220_RESPONSE_STATE_MINIMALITY_ATLAS_ROUTE,
  C222_SURFACE_CONDITIONED_RESPONSE_ATLAS_ROUTE,
  C233_SURFACE_TRANSPORT_COMPOSITION_ATLAS_ROUTE,
  C243_CONDITIONAL_EVENT_ATLAS_ROUTE,
  C244_INDEPENDENT_EVENT_REPLICATION_ROUTE,
  C245_CONFIRMED_EVENT_CORE_ROUTE,
  C254_TRI_MATERIAL_EVENT_ATLAS_ROUTE,
  C260_OUTPUT_PATH_CAUSAL_ATLAS_ROUTE,
  C262_GENERATION_SPECIFICITY_ATLAS_ROUTE,
  C272_STATE_CONDITIONED_OPERATOR_ATLAS_ROUTE,
  C273_RESPONSE_ECOLOGY_ATLAS_ROUTE,
  C275_CROSS_ROLE_REUSE_ATLAS_ROUTE,
  C289_JOINT_RESPONSE_CAMPAIGN_ATLAS_ROUTE,
  C308_CONDITIONAL_HYPERGRAPH_CAMPAIGN_ATLAS_ROUTE,
  C335_DUAL_AXIS_RESPONSE_ATLAS_ROUTE,
  C360_SINGLE_SAMPLE_OPERATOR_FIELD_ROUTE,
  C390_LANGUAGE_OPERATION_FIELD_ROUTE,
  C398_INDEPENDENT_CONSTRUCTION_LOCKBOX_ROUTE,
  C414_OUTPUT_SENSITIVE_LANGUAGE_FIELD_ROUTE,
  C433_AXIS_LOCKBOX_FIELD_ROUTE,
  C26801_RESIDUAL_STATE_OPERATOR_FIELD_ROUTE,
  C32561_LANGUAGE_ENCODING_FIELD_ROUTE,
  C42641_OUTPUT_CONDITIONED_CROSSMODEL_FIELD_ROUTE,
  GRAPH_WALSH_HEATMAP_ROUTE,
  RELATION_CONTRAST_HEATMAP_ROUTE,
  STATE_HEATMAP_ROUTE,
} from '../../researchKernel/heatmapResearchRoute';

import './ResearchHeatmapRoute.css';

function unitId(unit) {
  return Number(unit?.flat_index ?? unit?.unit_index);
}

function signedColor(value, observed = true) {
  if (!observed) return '#172033';
  const magnitude = Math.min(1, Math.abs(Number(value) || 0));
  if (value < 0) {
    return `rgb(${Math.round(35 + magnitude * 55)},${Math.round(75 + magnitude * 80)},${Math.round(145 + magnitude * 105)})`;
  }
  return `rgb(${Math.round(125 + magnitude * 125)},${Math.round(65 + magnitude * 115)},${Math.round(35 + magnitude * 45)})`;
}

function selectDimensions(events, limit) {
  const scores = new Map();
  events.forEach((event) => {
    (event?.top_units || []).forEach((unit) => {
      const id = unitId(unit);
      if (!Number.isFinite(id)) return;
      const score = scores.get(id) || { id, count: 0, magnitude: 0 };
      score.count += 1;
      score.magnitude = Math.max(score.magnitude, Number(unit.magnitude ?? Math.abs(unit.value)) || 0);
      scores.set(id, score);
    });
  });
  return [...scores.values()]
    .sort((left, right) => right.count - left.count || right.magnitude - left.magnitude || left.id - right.id)
    .slice(0, limit)
    .map((item) => item.id);
}

function rowFor(event, dimensions, scale) {
  const observed = new Map((event?.top_units || []).map((unit) => [unitId(unit), Number(unit.value) || 0]));
  return dimensions.map((dimension) => ({
    dimension,
    observed: observed.has(dimension),
    raw: observed.has(dimension) ? observed.get(dimension) : null,
    value: observed.has(dimension) ? observed.get(dimension) / scale : 0,
  }));
}

const LAYER_HEATMAP_LAYOUT = {
  layerAnchorX: 20,
  llmModelAnchorX: 0,
  layerScale: 1.5,
  layerBaseWidth: 3.8,
  layerBaseHeight: 2.8,
  sideGap: 0.55,
  fullStateScale: 1.05,
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function fitHeatmapGrid({
  dimensionsLength,
  targetWidth = null,
  targetHeight = null,
  requestedColumns = null,
  rowCount = 1,
}) {
  const dimCount = Math.max(1, Number.isFinite(dimensionsLength) ? Math.floor(dimensionsLength) : 1);
  const safeRows = Math.max(1, Number.isFinite(rowCount) ? Math.floor(rowCount) : 1);
  const widthTarget = Number.isFinite(targetWidth) ? targetWidth : null;
  const heightTarget = Number.isFinite(targetHeight) ? targetHeight : null;
  const requested = Number.isFinite(requestedColumns) ? Math.max(1, Math.floor(requestedColumns)) : null;
  const ratio = (widthTarget && heightTarget && heightTarget > 0) ? (widthTarget / heightTarget) : 1;
  const estimate = Math.max(1, Math.floor(Math.sqrt(dimCount * ratio)));
  const columns = Math.max(1, Math.min(requested ?? (dimCount <= 4 ? dimCount : estimate), dimCount));
  const rows = Math.max(1, Math.ceil(dimCount / columns));
  if (widthTarget && heightTarget && heightTarget > 0 && widthTarget > 0) {
    const size = clamp(Math.min(widthTarget / columns, heightTarget / (rows * safeRows)), 0.06, 3.0);
    return {
      columnCount: columns,
      size,
      width: widthTarget,
      height: rows * safeRows * size,
    };
  }
  return {
    columnCount: columns,
    size: 0.42,
    width: columns * 0.42,
    height: rows * safeRows * 0.42,
  };
}

function resolveDimensionLimit(displayConfig) {
  if (displayConfig?.mode === 'all') return Number.POSITIVE_INFINITY;
  const requested = Number(displayConfig?.topK ?? STATE_HEATMAP_ROUTE.maxDimensions);
  return Number.isFinite(requested) ? Math.max(1, Math.floor(requested)) : STATE_HEATMAP_ROUTE.maxDimensions;
}

function safeSliceDimensions(sourceDimensions, requested) {
  const dimensions = Array.isArray(sourceDimensions) ? sourceDimensions : [];
  const limit = Number.isFinite(requested) ? requested : dimensions.length;
  return dimensions.slice(0, limit);
}

function displayModeLabel(displayConfig) {
  return displayConfig?.mode === 'all'
    ? '全部参数'
    : `Top-${resolveDimensionLimit(displayConfig)}`;
}

function dimensionsForVector(values, displayConfig) {
  if (!Array.isArray(values)) return [];
  if (displayConfig?.mode === 'all') return values.map((_, index) => index);
  const limit = Math.min(values.length, resolveDimensionLimit(displayConfig));
  return values
    .map((value, dimension) => ({ dimension, magnitude: Math.abs(Number(value) || 0) }))
    .sort((left, right) => right.magnitude - left.magnitude)
    .slice(0, limit)
    .map((item) => item.dimension)
    .sort((left, right) => left - right);
}

function fullVectorCells(values, dimensions = values.map((_, index) => index)) {
  return dimensions.map((dimension) => ({
    dimension,
    observed: true,
    raw: Number(values[dimension]) || 0,
    value: Number(values[dimension]) || 0,
  }));
}

function buildStateHeatmapData(trace, displayConfig, fullStateVectors = null) {
  const events = Array.isArray(trace?.events) ? trace.events : [];
  const embeddingEvent = events.find((event) => event.event_type === STATE_HEATMAP_ROUTE.embeddingEvent) || null;
  const hasMatchingRunVectors = fullStateVectors?.run_id === trace?.run_id;
  const useFullVectors = Boolean(hasMatchingRunVectors);
  const fullEmbedding = useFullVectors && Array.isArray(fullStateVectors?.embedding)
    ? fullStateVectors.embedding
    : null;

  const priority = new Map(STATE_HEATMAP_ROUTE.hiddenStateEvents.map((name, index) => [name, index]));
  const hiddenByLayer = new Map();
  events.forEach((event) => {
    const layer = Number(event.layer);
    if (layer < 0 || !priority.has(event.event_type)) return;
    const previous = hiddenByLayer.get(layer);
    if (!previous || priority.get(event.event_type) < priority.get(previous.event_type)) {
      hiddenByLayer.set(layer, event);
    }
  });
  const hiddenEvents = [...hiddenByLayer.values()].sort((left, right) => left.layer - right.layer);

  const dimensionLimit = resolveDimensionLimit(displayConfig);
  const embeddingDimensions = useFullVectors && Array.isArray(fullEmbedding)
    ? dimensionsForVector(fullEmbedding, displayConfig)
    : selectDimensions(embeddingEvent ? [embeddingEvent] : [], dimensionLimit);
  const hiddenDimensions = useFullVectors
    ? []
    : selectDimensions(hiddenEvents, dimensionLimit);
  const fullHiddenValues = useFullVectors && fullStateVectors?.hidden_state
    ? fullStateVectors.hidden_state
    : {};
  const allUnits = [embeddingEvent, ...hiddenEvents].flatMap((event) => event?.top_units || []);
  const fullScale = [fullEmbedding, ...Object.values(fullHiddenValues)].reduce(
    (maximum, values) => Array.isArray(values)
      ? values.reduce((innerMaximum, value) => Math.max(innerMaximum, Math.abs(Number(value) || 0)), maximum)
      : maximum,
    1e-9,
  );
  const scale = allUnits.reduce(
    (maximum, unit) => Math.max(maximum, Math.abs(Number(unit?.value) || 0)),
    fullScale,
  );
  const hasEmbedding = Boolean(embeddingEvent || fullEmbedding);
  const fallbackHidden = useFullVectors && fullHiddenValues && typeof fullHiddenValues === 'object'
    ? Object.entries(fullHiddenValues)
      .filter((entry) => Array.isArray(entry[1]))
      .map(([layer]) => ({ layer: Number(layer), top_units: [], source: 'full_vectors' }))
    : [];
  const hiddenSourceEvents = hiddenEvents.length ? hiddenEvents : fallbackHidden;
  const hiddenRows = hiddenSourceEvents
    .sort((left, right) => Number(left.layer) - Number(right.layer))
    .map((event) => {
      const values = useFullVectors
        ? fullHiddenValues?.[String(event.layer)]
        : null;
      const dimensions = Array.isArray(values) ? dimensionsForVector(values, displayConfig) : selectDimensions([event], dimensionLimit);
      const cells = Array.isArray(values)
        ? fullVectorCells(values, dimensions).map((cell) => ({ ...cell, value: cell.value / scale }))
        : rowFor(event, dimensions, scale);
      return {
        layer: event.layer,
        cells,
        dimensions,
      };
    });

  return {
    available: Boolean(hasEmbedding),
    runId: trace?.run_id || '',
    model: trace?.model || '',
    token: trace?.tokens?.[trace?.token_position] || trace?.target_label || '',
    embedding: fullEmbedding
      ? fullVectorCells(fullEmbedding, embeddingDimensions).map((cell) => ({ ...cell, value: cell.value / scale }))
      : embeddingEvent ? rowFor(embeddingEvent, embeddingDimensions, scale) : [],
    hidden: hiddenRows,
    embeddingDimensions,
    hiddenDimensions: useFullVectors ? [] : hiddenDimensions,
    usingFullVectors: useFullVectors,
    displayLabel: displayModeLabel(displayConfig),
    fullVectorPending: displayConfig?.mode === 'all' && !useFullVectors,
    boundary: STATE_HEATMAP_ROUTE.boundary,
  };
}

function buildRelationContrastHeatmapData(payload, displayConfig) {
  if (payload?.schema !== RELATION_CONTRAST_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const dimensions = safeSliceDimensions(payload?.dimensions, requested);
  const rows = payload.common_rows
    .filter((row) => [31, 32].includes(Number(row.state)))
    .map((row) => ({
      ...row,
      label: `${row.partition.replace('response_', '')} / ${row.surface} / S${row.state}`,
      cells: dimensions.map((dimension, index) => ({
        dimension,
        observed: true,
        raw: row.values[index],
        value: row.normalized[index],
      })),
    }));
  return {
    available: Boolean(dimensions.length && rows.length),
    dimensions,
    rows,
    title: payload.title,
    phase: payload.phase,
    model: payload.model,
    evidence: payload.evidence,
  };
}

function buildGraphWalshHeatmapData(payload, displayConfig) {
  if (payload?.schema !== GRAPH_WALSH_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const dimensions = safeSliceDimensions(payload?.dimensions, requested);
  const scale = Math.max(1e-9, Number(payload?.scale?.symmetric_abs_q99) || 1);
  const rows = payload.rows
    .filter((row) => [24, 31, 32, 35].includes(Number(row.state)) && row.role === 'boundary')
    .map((row) => ({
      ...row,
      label: `${row.partition.replace('response_', '')} / ${row.world} / ${row.family} / S${row.state}`,
      cells: dimensions.map((dimension, index) => ({
        dimension,
        observed: true,
        raw: row.values[index],
        value: Number(row.values[index]) / scale,
      })),
    }));
  return {
    available: Boolean(dimensions.length && rows.length),
    dimensions,
    rows,
    phase: payload.phase,
    model: payload.model,
  };
}

function c101Dimensions(payload, displayConfig) {
  const requested = resolveDimensionLimit(displayConfig);
  const dimensions = displayConfig?.mode === 'all'
    ? payload?.dimensions
    : payload?.default_coordinates;
  return safeSliceDimensions(dimensions, requested);
}

function cellsForCoordinateRow(row, dimensions, scale) {
  return dimensions.map((dimension) => ({
    dimension,
    observed: true,
    raw: Number(row.values?.[dimension]) || 0,
    value: (Number(row.values?.[dimension]) || 0) / scale,
  }));
}

function robustCoordinateScale(rows, dimensions) {
  const values = rows.flatMap((row) => dimensions.map((dimension) => Math.abs(Number(row.values?.[dimension]) || 0)));
  if (!values.length) return 1;
  values.sort((left, right) => left - right);
  return Math.max(1e-9, values[Math.min(values.length - 1, Math.floor(values.length * 0.99))]);
}

function buildC101ActivationHeatmapData(payload, displayConfig) {
  if (payload?.schema !== C101_ACTIVATION_COORDINATE_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], walshRows: [], rawRows: [] };
  }
  const dimensions = c101Dimensions(payload, displayConfig);
  const walshSource = payload.walsh_rows.filter((row) => (
    row.partition === 'response_confirmation'
    && Number(row.state) === 24
    && row.role === 'boundary'
    && ((row.arm === 'confirmation' && row.effect === 'xy') || (row.arm === 'breadth' && row.effect === 'truth'))
  ));
  const firstCaseByArm = new Map();
  payload.raw_rows.forEach((row) => {
    if (!firstCaseByArm.has(row.arm)) firstCaseByArm.set(row.arm, row.case_id);
  });
  const rawSource = payload.raw_rows.filter((row) => {
    if (row.case_id !== firstCaseByArm.get(row.arm)) return false;
    if (Number(row.state) === 0) return ['target_pre', 'focus_pre'].includes(row.role);
    return [16, 24, 31, 32, 36].includes(Number(row.state))
      && ['target_pre', 'focus_pre', 'boundary'].includes(row.role);
  });
  const walshScale = Math.max(1e-9, Number(payload?.scale?.symmetric_abs_q99) || 1);
  const rawScale = robustCoordinateScale(rawSource, dimensions);
  const walshRows = walshSource.map((row) => ({
    ...row,
    label: row.arm === 'confirmation'
      ? `${row.arm} / ${row.world} / ${row.family} / ${row.effect} / S${row.state}`
      : `${row.arm} / ${row.family} / ${row.effect} / S${row.state}`,
    cells: cellsForCoordinateRow(row, dimensions, walshScale),
  }));
  const rawRows = rawSource.map((row) => ({
    ...row,
    label: `${row.arm} / ${row.role} / ${row.token_text || row.token_id} / ${row.state_kind === 'embedding' ? 'embedding' : `S${row.state}`}`,
    cells: cellsForCoordinateRow(row, dimensions, rawScale),
  }));
  return {
    available: Boolean(dimensions.length && walshRows.length && rawRows.length),
    dimensions,
    walshRows,
    rawRows,
    phase: payload.phase,
    model: payload.model,
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC102CoordinateBarcodeHeatmapData(payload, displayConfig) {
  if (payload?.schema !== C102_COORDINATE_BARCODE_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], effectRows: [], rawRows: [] };
  }
  const dimensions = c101Dimensions(payload, displayConfig);
  const effectSource = payload.effect_rows.filter((row) => (
    (row.dataset === 'c102_fresh' && row.partition === 'lockbox')
    || (row.dataset === 'c101_discovery' && row.effect === 'primary')
  ));
  const selectedStateByFamily = new Map(
    payload.effect_rows
      .filter((row) => row.dataset === 'c102_fresh' && row.partition === 'lockbox' && row.effect === 'primary')
      .map((row) => [row.family, Number(row.state)]),
  );
  const rawSource = displayConfig?.mode === 'all'
    ? payload.raw_rows.filter((row) => row.scope === 'all_states_boundary' && [0, 16, 24, 31, 35, 36].includes(Number(row.state)))
    : payload.raw_rows.filter((row) => (
      (row.scope === 'all_tokens_representative' && Number(row.state) === selectedStateByFamily.get(row.family))
      || (row.scope === 'all_states_boundary' && [0, selectedStateByFamily.get(row.family), 35, 36].includes(Number(row.state)))
    ));
  const effectScale = Math.max(1e-9, Number(payload?.scale?.effect_symmetric_abs_q99) || 1);
  const rawScale = robustCoordinateScale(rawSource, dimensions);
  const effectRows = effectSource.map((row) => ({
    ...row,
    label: `${row.dataset} / ${row.partition} / ${row.family} / ${row.effect} / S${row.state}`,
    cells: cellsForCoordinateRow(row, dimensions, effectScale),
  }));
  const rawRows = rawSource.map((row) => ({
    ...row,
    label: `${row.family} / p${row.token_position} ${row.token_text || row.token_id} / ${row.role} / ${row.state_kind === 'embedding' ? 'embedding' : `S${row.state}`}`,
    cells: cellsForCoordinateRow(row, dimensions, rawScale),
  }));
  return {
    available: Boolean(dimensions.length && effectRows.length && rawRows.length),
    dimensions,
    effectRows,
    rawRows,
    phase: payload.phase,
    model: payload.model,
    headline: payload.headline,
    coordinateSemantics: payload.coordinate_semantics,
  };
}

function buildC104UpstreamRoleBarcodeHeatmapData(payload, displayConfig) {
  if (payload?.schema !== C104_UPSTREAM_ROLE_BARCODE_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], effectRows: [], rawRows: [] };
  }
  const dimensions = c101Dimensions(payload, displayConfig);
  const effectSource = payload.effect_rows.filter((row) => (
    row.dataset === 'c103_frozen_source'
    || (row.dataset === 'c104_fresh' && row.partition === 'lockbox')
  ));
  const selectedStateByFamily = new Map(
    payload.effect_rows
      .filter((row) => row.dataset === 'c104_fresh' && row.partition === 'lockbox' && row.effect === 'truth')
      .map((row) => [row.family, Number(row.state)]),
  );
  const rawSource = displayConfig?.mode === 'all'
    ? payload.raw_rows.filter((row) => row.scope === 'frozen_role_all_states' && [0, 3, 19, 23, 35, 36].includes(Number(row.state)))
    : payload.raw_rows.filter((row) => (
      (row.scope === 'all_tokens_representative' && [0, selectedStateByFamily.get(row.family)].includes(Number(row.state)))
      || (row.scope === 'frozen_role_all_states' && [0, selectedStateByFamily.get(row.family), 35, 36].includes(Number(row.state)))
    ));
  const effectScale = Math.max(1e-9, Number(payload?.scale?.effect_symmetric_abs_q99) || 1);
  const rawScale = robustCoordinateScale(rawSource, dimensions);
  const effectRows = effectSource.map((row) => ({
    ...row,
    label: `${row.dataset} / ${row.partition} / ${row.family} / ${row.effect} / ${row.role} S${row.state}`,
    cells: cellsForCoordinateRow(row, dimensions, effectScale),
  }));
  const rawRows = rawSource.map((row) => ({
    ...row,
    label: `${row.family} / p${row.token_position} ${row.token_text || row.token_id} / ${row.role} / ${row.state_kind === 'embedding' ? 'embedding' : `S${row.state}`}`,
    cells: cellsForCoordinateRow(row, dimensions, rawScale),
  }));
  const supportRows = (payload.support_rows || []).map((row) => ({
    ...row,
    label: `${row.family} / raw-response discovery K=${row.k}`,
    cells: cellsForCoordinateRow(row, dimensions, 1),
  }));
  return {
    available: Boolean(dimensions.length && effectRows.length && rawRows.length),
    dimensions,
    effectRows,
    rawRows,
    supportRows,
    phase: payload.phase,
    model: payload.model,
    headline: payload.headline,
    codeAware: payload.code_aware_adjudication,
    freshC108: payload.fresh_c108,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC109RoleStateFieldAtlasData(payload, displayConfig) {
  if (payload?.schema !== C109_ROLE_STATE_FIELD_ATLAS_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], effectRows: [], rawRows: [], supportRows: [] };
  }
  const dimensions = c101Dimensions(payload, displayConfig);
  const c116Nomination = payload.c116_batch?.nomination || null;
  const c117Nomination = payload.c117_batch?.nomination || null;
  const effectSource = payload.effect_rows.filter((row) => (
    (row.role === 'query_anchor' && Number(row.state) === 19)
    || (row.role === 'boundary' && [24, 32].includes(Number(row.state)))
    || (row.role === 'focus_record' && [16, 19].includes(Number(row.state)))
    || (row.dataset === 'C116' && row.role === c116Nomination?.role && Number(row.state) === Number(c116Nomination?.state))
    || (row.dataset === 'C117' && row.role === c117Nomination?.role && Number(row.state) === Number(c117Nomination?.state))
  ));
  const rawSource = payload.raw_rows.filter((row) => (
    [0, 8, 16, 19, 24, 32, 36].includes(Number(row.state))
    || (row.dataset === 'C116' && row.role === c116Nomination?.role && Number(row.state) === Number(c116Nomination?.state))
    || (row.dataset === 'C117' && row.role === c117Nomination?.role && Number(row.state) === Number(c117Nomination?.state))
  ));
  const effectScale = Math.max(1e-9, Number(payload?.scale?.effect_symmetric_abs_q99) || 1);
  const rawScale = Math.max(1e-9, Number(payload?.scale?.raw_symmetric_abs_q99) || 1);
  const effectRows = effectSource.map((row) => ({
    ...row,
    label: `${row.dataset || 'C109'} / ${row.family} / ${row.partition.replace('prospective_', '').replace('independent_', '')} / ${row.role} / ${row.state_kind === 'embedding' ? 'embedding' : `S${row.state}`}`,
    cells: cellsForCoordinateRow(row, dimensions, effectScale),
  }));
  const rawRows = rawSource.map((row) => ({
    ...row,
    label: `${row.dataset || 'C109'} / ${row.family} / ${row.partition.replace('prospective_', '').replace('independent_', '')} / ${row.token_text} / ${row.state_kind === 'embedding' ? 'embedding' : `S${row.state}`}`,
    cells: cellsForCoordinateRow(row, dimensions, rawScale),
  }));
  const supportRows = payload.support_rows.map((row) => ({
    ...row,
    label: `${row.name} / K=${row.k}`,
    cells: cellsForCoordinateRow(row, dimensions, 1),
  }));
  const candidates = new Map(payload.candidate_query_anchor_state19.map((row) => [row.family, row]));
  const boundaryLocator = (family) => {
    const rows = payload.trajectory_rows.filter((row) => (
      row.family === family
      && row.role === 'boundary'
      && Number(row.state) > 0
      && Number(row.cross_partition_cosine) >= 0.9
      && Math.min(Number(row.prospective_norm), Number(row.lockbox_norm)) >= 1
    ));
    return rows.length ? Math.min(...rows.map((row) => Number(row.state))) : null;
  };
  const leverageRollup = (family) => (payload.leverage_summary || [])
    .filter((row) => row.family === family)
    .reduce((sum, row) => sum + Number(row.target_efficiency_exceeds_wrong_pairs || 0), 0);
  const c111TrajectoryRows = [];
  for (const family of ['attribute_binding', 'agent_patient']) {
    for (const role of ['focus_pre', 'focus_record', 'focus_post', 'query_focus', 'query_anchor', 'code_instruction', 'boundary']) {
      const states = (payload.c111_observation?.trajectory_rows || [])
        .filter((row) => row.family === family && row.role === role)
        .sort((left, right) => Number(left.state) - Number(right.state));
      if (states.length) {
        c111TrajectoryRows.push({
          family,
          role,
          label: `${family} / ${role}`,
          cells: states.map((row) => ({
            state: Number(row.state),
            value: Number(row.c109_c110_mean_cosine),
            crossPartition: Number(row.c110_cross_partition_cosine),
            oldNorm: Number(row.c109_mean_norm),
            newNorm: Number(row.c110_mean_norm),
          })),
        });
      }
    }
  }
  const c112Summaries = payload.c112_batch?.summaries || [];
  const c112ModeDefinitions = [
    { key: 'frozen_support', label: 'frozen support', value: (row) => row.frozen_support_median_gain },
    ...Array.from({ length: 8 }, (_, index) => ({ key: `perm_${index}`, label: `movement permutation ${index}`, value: (row) => row.movement_permutation_median_gains[index] })),
    ...['focus_pre', 'focus_record', 'focus_post', 'query_focus', 'query_anchor', 'code_instruction', 'boundary'].map((role) => ({ key: `single_${role}`, label: `single ${role}`, value: (row) => row.single_role_median_gains[role] })),
    ...['query_plus_record', 'query_plus_query_focus', 'record_to_query_path', 'all_registered_roles'].map((name) => ({ key: `coalition_${name}`, label: `coalition ${name}`, value: (row) => row.coalition_median_gains[name] })),
  ];
  const c112ModeRows = c112ModeDefinitions.map((mode) => ({
    key: mode.key,
    label: mode.label,
    cells: c112Summaries.map((row) => ({
      key: `${row.family}-${row.partition}-${row.code}`,
      family: row.family,
      partition: row.partition,
      code: Number(row.code),
      value: Number(mode.value(row) || 0),
    })),
  }));
  const c112Scale = Math.max(1e-9, ...c112ModeRows.flatMap((row) => row.cells.map((cell) => Math.abs(cell.value))));
  const c113Summaries = payload.c113_batch?.summaries || [];
  const c113ModeDefinitions = [
    { key: 'frozen_support', label: 'C113 frozen support', value: (row) => row.frozen_support_median_gain },
    ...Array.from({ length: 8 }, (_, index) => ({ key: `perm_${index}`, label: `C113 movement permutation ${index}`, value: (row) => row.movement_permutation_median_gains[index] })),
    ...['focus_pre', 'focus_record', 'focus_post', 'query_focus', 'query_anchor', 'code_instruction', 'boundary'].map((role) => ({ key: `single_${role}`, label: `C113 single ${role}`, value: (row) => row.single_role_median_gains[role] })),
    ...['record_to_query_path', 'path_plus_code', 'path_plus_code_boundary', 'all_registered_roles', 'path_without_focus_record', 'path_without_focus_post', 'path_without_query_focus', 'path_without_query_anchor'].map((name) => ({ key: `coalition_${name}`, label: `C113 coalition ${name}`, value: (row) => row.coalition_median_gains[name] })),
  ];
  const c113ModeRows = c113ModeDefinitions.map((mode) => ({
    key: mode.key,
    label: mode.label,
    cells: c113Summaries.map((row) => ({
      key: `${row.family}-${row.partition}-${row.code}`,
      family: row.family,
      partition: row.partition,
      code: Number(row.code),
      value: Number(mode.value(row) || 0),
    })),
  }));
  const c113Scale = Math.max(1e-9, ...c113ModeRows.flatMap((row) => row.cells.map((cell) => Math.abs(cell.value))));
  const c114Cells = payload.c114_structural_atlas?.cells || [];
  const c114Definitions = [
    { key: 'assignment_margin', label: 'C114 correct - permutation median', value: (row) => row.correct_minus_permutation_median },
    { key: 'path_increment', label: 'C114 path - query', value: (row) => row.path_minus_query },
    { key: 'all_increment', label: 'C114 all - path', value: (row) => row.all_minus_path },
  ];
  const c114Rows = c114Definitions.map((metric) => ({
    key: metric.key,
    label: metric.label,
    cells: c114Cells.map((row) => ({
      key: `${row.dataset}-${row.family}-${row.partition}-${row.code}`,
      dataset: row.dataset,
      family: row.family,
      partition: row.partition,
      code: Number(row.code),
      value: Number(metric.value(row) || 0),
    })),
  }));
  const c114Scale = Math.max(1e-9, ...c114Rows.flatMap((row) => row.cells.map((cell) => Math.abs(cell.value))));
  const c115Summaries = payload.c115_batch?.summaries || [];
  const c115Definitions = [
    { key: 'frozen_support', label: 'C115 frozen support', value: (row) => row.frozen_support_median_gain },
    ...Array.from({ length: 8 }, (_, index) => ({ key: `perm_${index}`, label: `C115 movement permutation ${index}`, value: (row) => row.movement_permutation_median_gains[index] })),
    ...['focus_pre', 'focus_record', 'focus_post', 'query_focus', 'query_anchor', 'code_instruction', 'boundary'].map((role) => ({ key: `single_${role}`, label: `C115 single ${role}`, value: (row) => row.single_role_median_gains[role] })),
    ...['record_to_query_path', 'path_plus_code', 'path_plus_code_boundary', 'all_registered_roles', 'path_without_focus_record', 'path_without_focus_post', 'path_without_query_focus', 'path_without_query_anchor'].map((name) => ({ key: `coalition_${name}`, label: `C115 coalition ${name}`, value: (row) => row.coalition_median_gains[name] })),
  ];
  const c115Rows = c115Definitions.map((metric) => ({
    key: metric.key,
    label: metric.label,
    cells: c115Summaries.map((row) => ({
      key: `${row.family}-${row.partition}-${row.code}`,
      family: row.family,
      partition: row.partition,
      code: Number(row.code),
      value: Number(metric.value(row) || 0),
    })),
  }));
  const c115Scale = Math.max(1e-9, ...c115Rows.flatMap((row) => row.cells.map((cell) => Math.abs(cell.value))));
  const c116Summaries = payload.c116_batch?.summaries || [];
  const c116Definitions = [
    { key: 'frozen_support', label: 'C116 frozen discovery support', value: (row) => row.frozen_support_median_gain },
    ...Array.from({ length: 8 }, (_, index) => ({ key: `perm_${index}`, label: `C116 movement permutation ${index}`, value: (row) => row.permutation_median_gains[index] })),
    ...['selected_role', 'query_anchor', 'record_to_query_path', 'all_registered_roles'].map((mode) => ({ key: mode, label: `C116 ${mode}`, value: (row) => row.mode_median_gains[mode] })),
  ];
  const c116Rows = c116Definitions.map((metric) => ({
    key: metric.key,
    label: metric.label,
    cells: c116Summaries.map((row) => ({
      key: `${row.partition}-${row.code}`,
      family: 'negation_scope',
      partition: row.partition,
      code: Number(row.code),
      value: Number(metric.value(row) || 0),
    })),
  }));
  const c116Scale = Math.max(1e-9, ...c116Rows.flatMap((row) => row.cells.map((cell) => Math.abs(cell.value))));
  const c117Summaries = payload.c117_batch?.summaries || [];
  const c117Definitions = [
    { key: 'frozen_support', label: 'C117 frozen discovery support', value: (row) => row.frozen_support_median_gain },
    ...Array.from({ length: 8 }, (_, index) => ({ key: `perm_${index}`, label: `C117 movement permutation ${index}`, value: (row) => row.permutation_median_gains[index] })),
    ...['selected_role', 'query_anchor', 'record_to_query_path', 'all_registered_roles'].map((mode) => ({ key: mode, label: `C117 ${mode}`, value: (row) => row.mode_median_gains[mode] })),
  ];
  const c117Rows = c117Definitions.map((metric) => ({
    key: metric.key,
    label: metric.label,
    cells: c117Summaries.map((row) => ({
      key: `${row.partition}-${row.code}`,
      family: 'whole_part_exception',
      partition: row.partition,
      code: Number(row.code),
      value: Number(metric.value(row) || 0),
    })),
  }));
  const c117Scale = Math.max(1e-9, ...c117Rows.flatMap((row) => row.cells.map((cell) => Math.abs(cell.value))));
  const transitionSource = (payload.transition_rows || []).filter((row) => (
    (row.state_kind !== 'layer_increment' && [0, 16, 24, 32, 35, 36].includes(Number(row.state)))
    || (row.state_kind === 'layer_increment' && [16, 24, 32, 35, 36].includes(Number(row.to_state)))
  ));
  const transitionScale = robustCoordinateScale(transitionSource, dimensions);
  const transitionCoordinateRows = transitionSource.map((row) => ({
    ...row,
    label: row.state_kind === 'layer_increment'
      ? `C123-C124 / ${row.family} / ${row.partition} / ${row.role} / S${row.from_state}->S${row.to_state}`
      : `C123-C124 / ${row.family} / ${row.partition} / ${row.role} / ${row.state_kind === 'embedding' ? 'embedding' : `S${row.state}`}`,
    cells: cellsForCoordinateRow(row, dimensions, transitionScale),
  }));
  const transitionProfiles = (payload.c123_c124_transition_batch?.profiles || []).map((row) => ({
    ...row,
    label: `${row.family} / ${row.partition} / ${row.role}`,
    cells: row.values.map((value, index) => ({ state: index + 1, value: Number(value) })),
  }));
  const transitionProfileScale = Math.max(1e-9, ...transitionProfiles.flatMap((row) => row.cells.map((cell) => Math.abs(cell.value))));
  const c125Source = payload.c125_final_transition_batch?.effect_rows || [];
  const c125Scale = robustCoordinateScale(c125Source, dimensions);
  const c125Rows = c125Source.map((row) => ({
    ...row,
    label: `C125 / ${row.family} / ${row.role} / ${row.kind} / ${row.checkpoint}`,
    cells: cellsForCoordinateRow(row, dimensions, c125Scale),
  }));
  const c126Source = (payload.c126_factor_response_batch?.effect_rows || []).filter((row) => ['truth', 'truth_x_code'].includes(row.effect));
  const c126Scale = robustCoordinateScale(c126Source, dimensions);
  const c126Rows = c126Source.map((row) => ({
    ...row,
    label: `C126 / ${row.family} / ${row.role} / ${row.effect} / ${row.kind}`,
    cells: cellsForCoordinateRow(row, dimensions, c126Scale),
  }));
  const c129Batch = payload.c129_direct_precedence_typed_transition_batch || null;
  const c129Source = (c129Batch?.effect_rows || []).filter((row) => (
    (row.kind === 'truth_response' && [0, 8, 16, 24, 32, 35, 36, 37].includes(Number(row.checkpoint_index)))
    || (row.kind === 'truth_response_increment' && [16, 24, 32, 35, 36, 37].includes(Number(row.transition_index) + 1))
  ));
  const c129Scale = robustCoordinateScale(c129Source, dimensions);
  const c129Rows = c129Source.map((row) => ({
    ...row,
    label: row.kind === 'truth_response'
      ? `C129 / ${row.partition} / ${row.role} / ${row.checkpoint}`
      : `C129 / ${row.partition} / ${row.role} / ${row.from_checkpoint}->${row.to_checkpoint}`,
    cells: cellsForCoordinateRow(row, dimensions, c129Scale),
  }));
  const c129Profiles = (c129Batch?.profiles || []).map((row) => ({
    ...row,
    label: `C129 / ${row.partition} / ${row.role}`,
    cells: row.values.map((value, index) => ({ transition: index, value: Number(value) })),
  }));
  const c129ProfileScale = Math.max(1e-9, ...c129Profiles.flatMap((row) => row.cells.map((cell) => Math.abs(cell.value))));
  const c129RawSource = c129Batch?.representative_raw_rows || [];
  const c129RawScale = robustCoordinateScale(c129RawSource, dimensions);
  const c129RawRows = c129RawSource.map((row) => ({
    ...row,
    label: `C129 raw / ${row.role} / ${row.checkpoint}`,
    cells: cellsForCoordinateRow(row, dimensions, c129RawScale),
  }));
  const c139Batch = payload.c133_c139_observation_batch || null;
  const c139Source = [
    ...(c139Batch?.c135?.gain_rows || []),
    ...(c139Batch?.c135?.representative_raw_rows || []),
    ...(c139Batch?.c136?.response_rows || []),
    ...(c139Batch?.c136?.representative_raw_rows || []),
    ...(c139Batch?.c138?.qwen_response_rows || []),
  ];
  const c139Scale = robustCoordinateScale(c139Source, dimensions);
  const c139Rows = c139Source.map((row) => ({
    ...row,
    label: [
      row.dataset,
      row.kind,
      row.task || row.depth || row.length_stratum || row.case_id,
      row.partition,
      row.role,
      row.checkpoint,
    ].filter((value) => value !== undefined && value !== null && value !== '').join(' / '),
    cells: cellsForCoordinateRow(row, dimensions, c139Scale),
  }));
  const c148Batch = payload.c140_c148_observation_batch || null;
  const c148Source = [
    ...(c148Batch?.c141?.representative_raw_rows || []),
    ...(c148Batch?.c142?.response_rows || []),
    ...(c148Batch?.c145?.error_rows || []),
    ...(payload.c149_c150_transition_window?.coordinate_rows || []),
    ...(payload.c151_fresh_transition_window?.coordinate_rows || []),
    ...(payload.c153_type_graph_confirmation?.coordinate_rows || []),
    ...(payload.c154_type_graph_causal?.coordinate_rows || []),
  ];
  const c148Scale = robustCoordinateScale(c148Source, dimensions);
  const c148Rows = c148Source.map((row) => ({
    ...row,
    label: [row.dataset, row.kind, row.arm, row.effect, row.stratum, row.partition, row.role, row.checkpoint]
      .filter((value) => value !== undefined && value !== null && value !== '').join(' / '),
    cells: cellsForCoordinateRow(row, dimensions, c148Scale),
  }));
  return {
    available: Boolean(dimensions.length && effectRows.length && rawRows.length),
    dimensions,
    effectRows,
    rawRows,
    supportRows,
    phase: payload.phase,
    model: payload.model,
    candidates,
    boundaryLocators: {
      attribute_binding: boundaryLocator('attribute_binding'),
      agent_patient: boundaryLocator('agent_patient'),
    },
    leverageRollup: {
      attribute_binding: leverageRollup('attribute_binding'),
      agent_patient: leverageRollup('agent_patient'),
    },
    freshC110: payload.fresh_c110 || null,
    c111Observation: payload.c111_observation || null,
    c111TrajectoryRows,
    c112Batch: payload.c112_batch || null,
    c112ModeRows,
    c112Scale,
    c113Batch: payload.c113_batch || null,
    c113ModeRows,
    c113Scale,
    c114Atlas: payload.c114_structural_atlas || null,
    c114Rows,
    c114Scale,
    c115Batch: payload.c115_batch || null,
    c115Rows,
    c115Scale,
    c116Batch: payload.c116_batch || null,
    c116Rows,
    c116Scale,
    c117Batch: payload.c117_batch || null,
    c117Rows,
    c117Scale,
    transitionBatch: payload.c123_c124_transition_batch || null,
    transitionCoordinateRows,
    transitionProfiles,
    transitionProfileScale,
    c125Batch: payload.c125_final_transition_batch || null,
    c125Rows,
    c125Scale,
    c126Batch: payload.c126_factor_response_batch || null,
    c126Rows,
    c126Scale,
    c129Batch,
    c129Rows,
    c129Scale,
    c129Profiles,
    c129ProfileScale,
    c129RawRows,
    c129RawScale,
    c139Batch,
    c139Rows,
    c139Scale,
    c148Batch,
    c148Rows,
    c148Scale,
    c153Confirmation: payload.c153_type_graph_confirmation?.confirmation || null,
    c154Causal: payload.c154_type_graph_causal?.causal || null,
    c155Transfer: payload.c155_checkpoint_transfer || null,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC157C166LocalFieldData(payload, displayConfig) {
  if (payload?.schema !== C157_C166_LOCAL_FIELD_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const sourceDimensions = displayConfig?.mode === 'all'
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const topSources = [...new Set(
    payload.rows
      .filter((row) => row.dataset === 'C161' && row.kind === 'source_target_response')
      .map((row) => row.source_coordinate),
  )].slice(0, 8);
  const sourceRows = payload.rows.filter((row) => (
    (row.dataset === 'C159' && row.kind === 'paired_response' && Number(row.checkpoint) === 32)
    || row.dataset === 'C160'
    || (row.dataset === 'C161' && (
      row.kind === 'outgoing_rms'
      || (topSources.includes(row.source_coordinate) && ['relation', 'boundary'].includes(row.target_role))
    ))
    || row.dataset === 'C162'
    || (row.dataset === 'C165' && Number(row.case_index) === 0)
  ));
  const scale = robustCoordinateScale(sourceRows, dimensions);
  const rows = sourceRows.map((row) => ({
    ...row,
    cells: cellsForCoordinateRow(row, dimensions, scale),
  }));
  return {
    available: Boolean(dimensions.length && rows.length),
    dimensions,
    rows,
    phase: payload.phase,
    model: payload.model,
    summaries: payload.summaries || {},
    c161: payload.c161 || null,
    c164: payload.c164 || null,
    c165: payload.c165 || null,
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC167C168RelationResidualData(payload, displayConfig) {
  if (payload?.schema !== C167_C168_RELATION_RESIDUAL_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const sourceDimensions = displayConfig?.mode === 'all'
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const sourceRows = payload.rows.filter((row) => (
    row.kind === 'relation_component'
    && (displayConfig?.mode !== 'all' || (
      row.source_coordinate === payload.source_coordinates?.[0]
      && row.target_role === 'relation'
    ))
  ));
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    c168: payload.c168 || null,
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC170RoleCheckpointData(payload, displayConfig) {
  if (payload?.schema !== C170_ROLE_CHECKPOINT_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const sourceDimensions = displayConfig?.mode === 'all'
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const sourceRows = payload.rows.filter((row) => (
    displayConfig?.mode !== 'all'
    || (
      row.source_coordinate === payload.source_coordinates?.[0]
      && row.target_role === 'relation'
      && row.relation === 'is_a'
    )
  ));
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    c170: payload.c170 || null,
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC183NaturalResponseEcologyData(payload, displayConfig) {
  if (payload?.schema !== C183_NATURAL_RESPONSE_ECOLOGY_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const allMode = displayConfig?.mode === 'all';
  const sourceDimensions = allMode
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const sourceRows = payload.rows.filter((row) => {
    if (allMode) {
      return (
        row.family === 'is_a'
        && (
          (row.kind === 'anchor_state' && row.role === 'relation' && [0, 24, 25, 37].includes(row.checkpoint))
          || (row.kind === 'local_response' && row.partition === 'fresh' && row.source_coordinate === payload.source_coordinates?.[0] && ['query', 'relation'].includes(row.target_role))
        )
      );
    }
    return (
      (row.kind === 'anchor_state' && row.family === 'is_a' && [0, 16, 24, 25, 37].includes(row.checkpoint))
      || (row.kind === 'local_response' && row.partition === 'fresh' && row.source_coordinate === payload.source_coordinates?.[0] && ['query', 'relation'].includes(row.target_role))
    );
  });
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    synthesis: payload.synthesis || null,
    totalRows: payload.total_rows ?? payload.rows.length,
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC189NewMaterialResponseScaffoldData(payload, displayConfig) {
  if (payload?.schema !== C189_NEW_MATERIAL_RESPONSE_SCAFFOLD_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const allMode = displayConfig?.mode === 'all';
  const sourceDimensions = allMode
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const sourceRows = payload.rows.filter((row) => {
    if (row.kind === 'aggregate_target_energy_profile') return true;
    if (allMode) return row.family === 'is_a';
    return (
      (row.kind === 'target_energy_profile' && row.unit === 0)
      || (row.kind === 'signed_mean_response' && row.family === 'is_a' && row.unit === 0)
    );
  });
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    synthesis: payload.synthesis || null,
    totalRows: payload.rows.length,
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC191ResponseEquivalenceData(payload, displayConfig) {
  if (payload?.schema !== C191_RESPONSE_EQUIVALENCE_ATLAS_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const allMode = displayConfig?.mode === 'all';
  const sourceDimensions = allMode
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const sourceRows = allMode
    ? payload.rows.filter((row) => row.unit === 0 && row.phrase_variant === 0 && row.wrapper_variant === 0)
    : payload.rows;
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: payload.rows.length,
    summary: payload.nearest_neighbor_summary || {},
    dominant: payload.dominant_registered_label,
    missing: payload.registered_missing?.length || 0,
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC193ProgramCenteredResidualData(payload, displayConfig) {
  if (payload?.schema !== C193_PROGRAM_CENTERED_RESIDUAL_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const allMode = displayConfig?.mode === 'all';
  const sourceDimensions = allMode
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const sourceRows = allMode
    ? payload.rows.filter((row) => row.program === 'direct_target' && row.unit === 1 && row.phrase_variant === 0)
    : payload.rows;
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: payload.rows.length,
    rawSummary: payload.raw_summary || {},
    residualResult: payload.residual_result || {},
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC202SignedOperatorCampaignData(payload, displayConfig) {
  if (payload?.schema !== C202_SIGNED_OPERATOR_CAMPAIGN_HEATMAP_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const allMode = displayConfig?.mode === 'all';
  const sourceDimensions = allMode
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const sourceRows = allMode
    ? payload.rows.filter((row) => row.kind === 'C198_natural_signed_response' && row.role === 'boundary')
    : payload.rows;
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: payload.rows.length,
    summary: payload.summary || {},
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC215ResponseIntervalCompositionData(payload, displayConfig) {
  if (payload?.schema !== C215_RESPONSE_INTERVAL_COMPOSITION_ATLAS_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const allMode = displayConfig?.mode === 'all';
  const sourceDimensions = allMode
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const sourceRows = allMode
    ? payload.rows.filter((row) => row.program === 'type_chain' || (row.program === 'path_factorial' && row.role === 'boundary'))
    : payload.rows.filter((row) => (
      (row.program === 'type_chain' && row.role === 'relation')
      || (row.program === 'path_factorial' && ['relation', 'boundary'].includes(row.role))
    ));
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: payload.rows.length,
    summary: payload.summary || {},
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildSignedCoordinateAtlasData(payload, route, displayConfig, compactSources) {
  if (payload?.schema !== route.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const allMode = displayConfig?.mode === 'all';
  const sourceDimensions = allMode
    ? payload?.dimensions
    : payload?.default_coordinates;
  const dimensions = safeSliceDimensions(sourceDimensions, requested);
  const sourceRows = allMode
    ? payload.rows
    : payload.rows.filter((row) => compactSources.includes(row.source));
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: payload.total_rows ?? payload.rows.length,
    summary: payload.summary || {},
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC390LanguageOperationData(payload, displayConfig) {
  if (payload?.schema !== C390_LANGUAGE_OPERATION_FIELD_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const dimensions = safeSliceDimensions(payload.dimensions, requested);
  const familyRows = payload.family_operation_rows.filter((row) => (
    row.operation === 'I' && [0, 24].includes(Number(row.checkpoint))
  ));
  const tokenRows = payload.all_token_rows.filter((row) => (
    Number(row.checkpoint) === 24 && Number(row.token) < 8
  ));
  const sourceRows = [...familyRows, ...tokenRows].map((row) => ({
    ...row,
    label: row.family
      ? `${row.family} / ${row.operation} / q${row.checkpoint} / ${row.role}`
      : `token ${row.token} / q${row.checkpoint}`,
  }));
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: payload.family_operation_rows.length + payload.all_token_rows.length,
    coordinateSemantics: 'embedding / HiddenState signed response at physical activation-coordinate resolution',
    claimBoundary: payload.claim_boundary,
  };
}

function buildC398IndependentConstructionData(payload, displayConfig) {
  if (payload?.schema !== C398_INDEPENDENT_CONSTRUCTION_LOCKBOX_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const dimensions = safeSliceDimensions(payload.dimensions, requested);
  const sourceRows = payload.rows
    .filter((row) => [0, 24].includes(Number(row.checkpoint)))
    .map((row) => ({
      ...row,
      label: `${row.family} / ${row.operation} / q${row.checkpoint} / ${row.role}`,
    }));
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: payload.rows.length,
    summary: payload.summary || {},
    coordinateSemantics: 'fresh-construction embedding / HiddenState interaction centroid at physical activation-coordinate resolution',
    claimBoundary: payload.claim_boundary,
  };
}

function buildC414OutputSensitiveLanguageData(payload, displayConfig) {
  if (payload?.schema !== C414_OUTPUT_SENSITIVE_LANGUAGE_FIELD_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const dimensions = safeSliceDimensions(payload.dimensions, requested);
  const sourceRows = payload.rows
    .filter((row) => (
      row.source === 'family_interaction'
      && [0, 24].includes(Number(row.checkpoint))
      && ['primary', 'boundary'].includes(row.role)
    ))
    .map((row) => ({
      ...row,
      label: `${row.family} / q${row.checkpoint} / ${row.role}`,
    }));
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: payload.rows.length,
    coordinateSemantics: 'output-sensitive family interaction at embedding/HiddenState physical activation-coordinate resolution',
    claimBoundary: payload.claim_boundary,
  };
}

function buildC433AxisLockboxData(payload, displayConfig) {
  if (payload?.schema !== C433_AXIS_LOCKBOX_FIELD_ROUTE.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const dimensions = safeSliceDimensions(payload.dimensions, requested);
  const sourceRows = payload.rows
    .filter((row) => ['primary', 'boundary'].includes(row.role))
    .map((row) => ({
      ...row,
      label: `${row.family} / ${row.query_axis} / mask ${row.mask} / ${row.role}`,
    }));
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: payload.rows.length,
    coordinateSemantics: 'frozen-axis Mobius interaction at q24 HiddenState physical activation-coordinate resolution',
    claimBoundary: payload.claim_boundary,
  };
}

function buildC26801ResidualStateOperatorData(payload, displayConfig, route = C26801_RESIDUAL_STATE_OPERATOR_FIELD_ROUTE) {
  if (payload?.schema !== route.sourceSchema) {
    return { available: false, dimensions: [], rows: [] };
  }
  const sourceRows = Array.isArray(payload.rows)
    ? payload.rows.map((row) => ({
      ...row,
      label: row.label || (row.source === 'sample_state'
        ? `${row.coordinate_kind} / q${row.layer} / ${row.event}`
        : row.source === 'family_passport'
          ? `${row.family} / ${row.component} / q${row.layer} / ${row.event}`
          : `${row.source} / ${row.component} / q${row.layer} / ${row.event}`),
    }))
    : [];
  const requested = resolveDimensionLimit(displayConfig);
  let dimensions = Array.isArray(payload.dimensions) ? payload.dimensions : [];
  if (displayConfig?.mode !== 'all') {
    dimensions = dimensions
      .map((dimension) => ({
        dimension,
        magnitude: sourceRows.reduce(
          (maximum, row) => Math.max(maximum, Math.abs(Number(row.values?.[dimension]) || 0)),
          0,
        ),
      }))
      .sort((left, right) => right.magnitude - left.magnitude || left.dimension - right.dimension)
      .slice(0, requested)
      .map((item) => item.dimension)
      .sort((left, right) => left - right);
  }
  const scale = robustCoordinateScale(sourceRows, dimensions);
  return {
    available: Boolean(dimensions.length && sourceRows.length),
    dimensions,
    rows: sourceRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    phase: payload.phase,
    model: payload.model,
    totalRows: sourceRows.length,
    coordinateSemantics: payload.coordinate_semantics,
    claimBoundary: payload.claim_boundary,
  };
}

function buildC42641CrossmodelFieldData(payload, displayConfig) {
  if (payload?.schema !== C42641_OUTPUT_CONDITIONED_CROSSMODEL_FIELD_ROUTE.sourceSchema) {
    return { available: false, panels: [] };
  }
  const requested = resolveDimensionLimit(displayConfig);
  const panels = (payload.models || []).map((section) => {
    const sourceRows = (section.rows || []).map((row) => ({ ...row, label: row.label || row.source }));
    const physicalDimensions = Array.from({ length: Number(section.coordinate_count) || 0 }, (_, index) => index);
    const requestedOrderKey = ['fingerprint', 'chain_fingerprint', 'orthogonal_signed', 'nonce_family', 'semantic_walsh', 'relation_graph', 'event_path'].includes(displayConfig?.coordinateOrder)
      ? displayConfig.coordinateOrder
      : 'physical';
    const fingerprintDimensions = Array.isArray(section.coordinate_orders?.[requestedOrderKey])
      && section.coordinate_orders[requestedOrderKey].length === physicalDimensions.length
      ? section.coordinate_orders[requestedOrderKey].map(Number)
      : null;
    const useFingerprintOrder = requestedOrderKey !== 'physical' && fingerprintDimensions;
    const coordinateOrder = useFingerprintOrder ? fingerprintDimensions : physicalDimensions;
    let dimensions = [...coordinateOrder];
    if (displayConfig?.mode !== 'all') {
      const selected = new Set(physicalDimensions
        .map((dimension) => ({
          dimension,
          magnitude: sourceRows.reduce(
            (maximum, row) => Math.max(maximum, Math.abs(Number(row.values?.[dimension]) || 0)),
            0,
          ),
        }))
        .sort((left, right) => right.magnitude - left.magnitude || left.dimension - right.dimension)
        .slice(0, requested)
        .map((item) => item.dimension)
      );
      dimensions = coordinateOrder.filter((dimension) => selected.has(dimension));
    }
    const previewRows = sourceRows.filter((row) => row.preview !== false);
    const scale = robustCoordinateScale(previewRows, dimensions);
    return {
      key: section.key,
      model: section.model,
      precision: section.precision,
      coordinateCount: section.coordinate_count,
      coordinateOrderLabel: useFingerprintOrder
        ? (requestedOrderKey === 'chain_fingerprint'
          ? 'frozen knowledge-chain response-fingerprint order'
          : requestedOrderKey === 'orthogonal_signed'
            ? 'frozen orthogonal-family signed-response order'
            : requestedOrderKey === 'nonce_family'
              ? 'frozen nonce-marker family-response order'
              : requestedOrderKey === 'semantic_walsh'
                ? 'frozen behavior-necessary semantic-selection Walsh order'
              : requestedOrderKey === 'relation_graph'
                 ? 'frozen partner-recombined relation-graph order'
                : requestedOrderKey === 'event_path'
                  ? 'frozen natural-event residual-path order'
             : 'frozen discovery-set response-fingerprint order')
        : 'model-local physical coordinate order',
      dimensions,
      rows: previewRows.map((row) => ({ ...row, cells: cellsForCoordinateRow(row, dimensions, scale) })),
    };
  }).filter((panel) => panel.dimensions.length && panel.rows.length);
  return { available: panels.length > 0, panels, phase: payload.phase, claimBoundary: payload.claim_boundary };
}

export function ResearchHeatmapRouteCard({ trace, displayConfig, fullStateVectors, relationContrastHeatmap, graphWalshHeatmap, c101ActivationHeatmap, c102CoordinateBarcodeHeatmap, c104UpstreamRoleBarcodeHeatmap, c109RoleStateFieldAtlas, c157C166LocalFieldHeatmap, c167C168RelationResidualHeatmap, c170RoleCheckpointHeatmap, c183NaturalResponseEcologyHeatmap, c189NewMaterialResponseScaffoldHeatmap, c191ResponseEquivalenceAtlas, c193ProgramCenteredResidualHeatmap, c202SignedOperatorCampaignHeatmap, c215ResponseIntervalCompositionAtlas, c220ResponseStateMinimalityAtlas, c222SurfaceConditionedResponseAtlas, c233SurfaceTransportCompositionAtlas, c243ConditionalEventAtlas, c244IndependentEventReplication, c245ConfirmedEventCore, c254TriMaterialEventAtlas, c260OutputPathCausalAtlas, c262GenerationSpecificityAtlas, c272StateConditionedOperatorAtlas, c273ResponseEcologyAtlas, c275CrossRoleReuseAtlas, c289JointResponseCampaignAtlas, c308ConditionalHypergraphCampaignAtlas, c335DualAxisResponseAtlas, c360SingleSampleOperatorField, c390LanguageOperationField, c398IndependentConstructionLockbox, c414OutputSensitiveLanguageField, c433AxisLockboxField, onDisplayConfigChange }) {
  const data = buildStateHeatmapData(trace, displayConfig, fullStateVectors);
  const relationData = buildRelationContrastHeatmapData(relationContrastHeatmap, displayConfig);
  const graphData = buildGraphWalshHeatmapData(graphWalshHeatmap, displayConfig);
  const c101Data = buildC101ActivationHeatmapData(c101ActivationHeatmap, displayConfig);
  const c102Data = buildC102CoordinateBarcodeHeatmapData(c102CoordinateBarcodeHeatmap, displayConfig);
  const c104Data = buildC104UpstreamRoleBarcodeHeatmapData(c104UpstreamRoleBarcodeHeatmap, displayConfig);
  const c109Data = buildC109RoleStateFieldAtlasData(c109RoleStateFieldAtlas, displayConfig);
  const c157C166Data = buildC157C166LocalFieldData(c157C166LocalFieldHeatmap, displayConfig);
  const c167C168Data = buildC167C168RelationResidualData(c167C168RelationResidualHeatmap, displayConfig);
  const c170Data = buildC170RoleCheckpointData(c170RoleCheckpointHeatmap, displayConfig);
  const c183Data = buildC183NaturalResponseEcologyData(c183NaturalResponseEcologyHeatmap, displayConfig);
  const c189Data = buildC189NewMaterialResponseScaffoldData(c189NewMaterialResponseScaffoldHeatmap, displayConfig);
  const c191Data = buildC191ResponseEquivalenceData(c191ResponseEquivalenceAtlas, displayConfig);
  const c193Data = buildC193ProgramCenteredResidualData(c193ProgramCenteredResidualHeatmap, displayConfig);
  const c202Data = buildC202SignedOperatorCampaignData(c202SignedOperatorCampaignHeatmap, displayConfig);
  const c215Data = buildC215ResponseIntervalCompositionData(c215ResponseIntervalCompositionAtlas, displayConfig);
  const c220Data = buildSignedCoordinateAtlasData(
    c220ResponseStateMinimalityAtlas,
    C220_RESPONSE_STATE_MINIMALITY_ATLAS_ROUTE,
    displayConfig,
    ['C219_shared_interface_fresh', 'fresh_minus_template'],
  );
  const c222Data = buildSignedCoordinateAtlasData(
    c222SurfaceConditionedResponseAtlas,
    C222_SURFACE_CONDITIONED_RESPONSE_ATLAS_ROUTE,
    displayConfig,
    ['C221_fresh_mean', 'C221_fresh_minus_C216'],
  );
  const c233Data = buildSignedCoordinateAtlasData(
    c233SurfaceTransportCompositionAtlas,
    C233_SURFACE_TRANSPORT_COMPOSITION_ATLAS_ROUTE,
    displayConfig,
    ['C225_passport', 'C227_transport_lockbox', 'C229_composition_lockbox'],
  );
  const c243Data = buildSignedCoordinateAtlasData(
    c243ConditionalEventAtlas,
    C243_CONDITIONAL_EVENT_ATLAS_ROUTE,
    displayConfig,
    ['C243_core'],
  );
  const c244Data = buildSignedCoordinateAtlasData(
    c244IndependentEventReplication,
    C244_INDEPENDENT_EVENT_REPLICATION_ROUTE,
    displayConfig,
    ['C244_independent'],
  );
  const c245Data = buildSignedCoordinateAtlasData(
    c245ConfirmedEventCore,
    C245_CONFIRMED_EVENT_CORE_ROUTE,
    displayConfig,
    ['C245_confirmed_core'],
  );
  const c254Data = buildSignedCoordinateAtlasData(
    c254TriMaterialEventAtlas,
    C254_TRI_MATERIAL_EVENT_ATLAS_ROUTE,
    displayConfig,
    ['tri_material_role_core', 'full_token_signed_balance'],
  );
  const c260Data = buildSignedCoordinateAtlasData(
    c260OutputPathCausalAtlas,
    C260_OUTPUT_PATH_CAUSAL_ATLAS_ROUTE,
    displayConfig,
    ['c260_early_output_path'],
  );
  const c262Data = buildSignedCoordinateAtlasData(
    c262GenerationSpecificityAtlas,
    C262_GENERATION_SPECIFICITY_ATLAS_ROUTE,
    displayConfig,
    ['c262_generation_specificity'],
  );
  const c272Data = buildSignedCoordinateAtlasData(
    c272StateConditionedOperatorAtlas,
    C272_STATE_CONDITIONED_OPERATOR_ATLAS_ROUTE,
    displayConfig,
    ['fourth_edit_response', 'state_conditioned_passport_sign', 'nested_interaction', 'typed_interaction'],
  );
  const c273Data = buildSignedCoordinateAtlasData(
    c273ResponseEcologyAtlas,
    C273_RESPONSE_ECOLOGY_ATLAS_ROUTE,
    displayConfig,
    ['c273_full_coordinate_failure_ecology'],
  );
  const c275Data = buildSignedCoordinateAtlasData(
    c275CrossRoleReuseAtlas,
    C275_CROSS_ROLE_REUSE_ATLAS_ROUTE,
    displayConfig,
    ['c275_cross_role_same_sign_reuse'],
  );
  const c289Data = buildSignedCoordinateAtlasData(
    c289JointResponseCampaignAtlas,
    C289_JOINT_RESPONSE_CAMPAIGN_ATLAS_ROUTE,
    displayConfig,
    ['c280_joint_word_prediction'],
  );
  const c308Data = buildSignedCoordinateAtlasData(
    c308ConditionalHypergraphCampaignAtlas,
    C308_CONDITIONAL_HYPERGRAPH_CAMPAIGN_ATLAS_ROUTE,
    displayConfig,
    ['c295_sixth_embedding_response', 'c295_sixth_hidden_response', 'c296_complete_transition', 'c297_amplitude_regime', 'c300_sixth_lockbox_tournament', 'c302_composition_forecast', 'c305_causal_qualification'],
  );
  const c335Data = buildSignedCoordinateAtlasData(
    c335DualAxisResponseAtlas,
    C335_DUAL_AXIS_RESPONSE_ATLAS_ROUTE,
    displayConfig,
    ['c314_operator_passport', 'c333_graph_depth_operator'],
  );
  const c360Data = buildSignedCoordinateAtlasData(
    c360SingleSampleOperatorField,
    C360_SINGLE_SAMPLE_OPERATOR_FIELD_ROUTE,
    displayConfig,
    ['c360_embedding', 'c360_hidden_state'],
  );
  const c390Data = buildC390LanguageOperationData(c390LanguageOperationField, displayConfig);
  const c398Data = buildC398IndependentConstructionData(c398IndependentConstructionLockbox, displayConfig);
  const c414Data = buildC414OutputSensitiveLanguageData(c414OutputSensitiveLanguageField, displayConfig);
  const c433Data = buildC433AxisLockboxData(c433AxisLockboxField, displayConfig);
  return (
    <section className="research-heatmap-card" aria-label="Embedding + HiddenState Heatmap Card">
      <header>
        <div>
          <BarChart3 size={13} />
          状态热力图
        </div>
        <span>{data.available ? `Run ${data.runId || 'N/A'}` : '等待真实 trace'}</span>
      </header>
      <label style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 7, alignItems: 'center', marginTop: 9, color: '#8aa0b7', fontSize: 9 }}>
        显示范围
        <select
          value={displayConfig?.mode === 'all' ? 'all' : String(displayConfig?.topK ?? STATE_HEATMAP_ROUTE.maxDimensions)}
          onChange={(event) => {
            const value = event.target.value;
            onDisplayConfigChange?.(value === 'all'
              ? { ...displayConfig, mode: 'all' }
              : { ...displayConfig, mode: 'top_k', topK: Number(value) });
          }}
          style={{ minWidth: 0, padding: '5px 6px', border: '1px solid rgba(125,211,252,0.22)', borderRadius: 6, background: 'rgba(2,6,23,0.72)', color: '#dbeafe', fontSize: 9 }}
        >
          {[4, 8, 12, 16].map((count) => <option key={count} value={count}>{count} 个坐标</option>)}
          <option value="all">全部参数</option>
        </select>
      </label>
      <label style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 7, alignItems: 'center', marginTop: 7, color: '#8aa0b7', fontSize: 9 }}>
        坐标顺序
        <select
          value={['fingerprint', 'chain_fingerprint', 'orthogonal_signed', 'nonce_family', 'semantic_walsh', 'relation_graph', 'event_path'].includes(displayConfig?.coordinateOrder) ? displayConfig.coordinateOrder : 'physical'}
          onChange={(event) => onDisplayConfigChange?.({ ...displayConfig, coordinateOrder: event.target.value })}
          style={{ minWidth: 0, padding: '5px 6px', border: '1px solid rgba(125,211,252,0.22)', borderRadius: 6, background: 'rgba(2,6,23,0.72)', color: '#dbeafe', fontSize: 9 }}
        >
          <option value="physical">物理坐标顺序</option>
          <option value="fingerprint">冻结响应指纹顺序</option>
          <option value="chain_fingerprint">知识链冻结指纹顺序</option>
          <option value="orthogonal_signed">正交语言族有符号顺序</option>
          <option value="nonce_family">无意义标记语言族顺序</option>
          <option value="semantic_walsh">语义选择四格交互顺序</option>
          <option value="relation_graph">关系伙伴闭环顺序</option>
          <option value="event_path">自然事件残差路径顺序</option>
        </select>
      </label>
      {data.available ? (
        <div className="research-heatmap-card__summary">
          <strong>词嵌入</strong><span>{data.embedding.length} 个维度（{data.displayLabel}）</span>
          <strong>HiddenState</strong><span>{data.hidden.length} 层，每层 {data.hidden[0]?.cells.length || 0} 个维度</span>
          <strong>模型 / Token</strong><span>{data.model || 'unknown'} / {data.token || 'N/A'}</span>
        </div>
      ) : (
        <p>
          当前运行尚未同时提供词嵌入与 HiddenState 的 top-k 原始记录；界面保持可用，等待下一次 Run 写入。
        </p>
      )}
      {data.fullVectorPending && <p><Info size={11} /> 正在加载本次 Run 的完整参数向量；加载完成后将替换当前稀疏显示。</p>}
      <p><Info size={11} /> {data.boundary}</p>
      {relationData.available && (
        <section className="research-heatmap-card__relation" aria-label="Relation Contrast Heatmap">
          <header>
            <strong>{RELATION_CONTRAST_HEATMAP_ROUTE.title}</strong>
            <span>Phase {relationData.phase} / {relationData.model}</span>
          </header>
          <div className="research-heatmap-card__relation-scroll">
            {relationData.rows.map((row) => (
              <div className="research-heatmap-card__relation-row" key={row.label}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': relationData.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`coordinate ${cell.dimension}: ${Number(cell.raw).toFixed(5)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {RELATION_CONTRAST_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {graphData.available && (
        <section className="research-heatmap-card__relation" aria-label="Directed Graph Walsh Heatmap">
          <header>
            <strong>{GRAPH_WALSH_HEATMAP_ROUTE.title}</strong>
            <span>Phase {graphData.phase} / {graphData.model}</span>
          </header>
          <div className="research-heatmap-card__relation-scroll">
            {graphData.rows.map((row) => (
              <div className="research-heatmap-card__relation-row" key={row.label}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': graphData.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`coordinate ${cell.dimension}: ${Number(cell.raw).toFixed(5)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {GRAPH_WALSH_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c101Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C101 Activation Coordinate Heatmap">
          <header>
            <strong>{C101_ACTIVATION_COORDINATE_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c101Data.phase} / {c101Data.model} / {c101Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__coordinate-axis">
            <span>Walsh response field</span>
            <b>{c101Data.dimensions[0]} ... {c101Data.dimensions[c101Data.dimensions.length - 1]}</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c101Data.walshRows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c101Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>Raw embedding + Hidden State</span>
            <b>token, subtoken, role preserved</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c101Data.rawRows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.case_id}-${row.role}-${row.subtoken}-${row.state}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c101Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`${row.case_id} / ${row.role} / token ${row.token_id} ${row.token_text} / subtoken ${row.subtoken} / ${row.state_kind} ${row.state} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {C101_ACTIVATION_COORDINATE_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c102Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C102 Coordinate Barcode Heatmap">
          <header>
            <strong>{C102_COORDINATE_BARCODE_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c102Data.phase} / {c102Data.model} / {c102Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>前瞻条码</strong><span>{c102Data.headline?.barcode_three_stage_passed || 0}/{c102Data.headline?.barcode_total || 0}</span>
            <strong>受控干预</strong><span>{c102Data.headline?.controlled_intervention_passed || 0}/{c102Data.headline?.controlled_intervention_total || 0}</span>
            <strong>行为准确率</strong><span>{Number(c102Data.headline?.behavior_accuracy || 0).toFixed(3)}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>C101 source vs C102 lockbox: primary, code, interaction</span>
            <b>{c102Data.dimensions[0]} ... {c102Data.dimensions[c102Data.dimensions.length - 1]}</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c102Data.effectRows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c102Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>Token embedding + Hidden State coordinates</span>
            <b>physical token, semantic role and state preserved</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c102Data.rawRows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.case_id}-${row.token_position}-${row.state}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c102Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`${row.case_id} / token position ${row.token_position} / token ${row.token_id} ${row.token_text} / ${row.role} / ${row.state_kind} ${row.state} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {C102_COORDINATE_BARCODE_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c104Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C104 Upstream Role-State Heatmap">
          <header>
            <strong>{C104_UPSTREAM_ROLE_BARCODE_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c104Data.phase} / {c104Data.model} / {c104Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>新材料条码</strong><span>{c104Data.headline?.fresh_barcode_passed || 0}/{c104Data.headline?.fresh_barcode_total || 0}</span>
            <strong>Raw truth 四格</strong><span>属性 4/4 / 施事 4/4</span>
            <strong>Task-aligned 四格</strong><span>属性 2/4 / 施事 2/4</span>
            <strong>行为 标准/反转</strong><span>{Number(c104Data.headline?.standard_accuracy || 0).toFixed(3)} / {Number(c104Data.headline?.reversed_accuracy || 0).toFixed(3)}</span>
            <strong>首次 raw 响应 K</strong><span>属性 {c104Data.headline?.raw_first_tested_all_four_k?.attribute_binding ?? 'N/A'} / 施事 {c104Data.headline?.raw_first_tested_all_four_k?.agent_patient ?? 'N/A'}</span>
            <strong>Task-aligned K</strong><span>属性 {c104Data.headline?.task_aligned_all_four_k?.attribute_binding ?? '无'} / 施事 {c104Data.headline?.task_aligned_all_four_k?.agent_patient ?? '无'}</span>
            <strong>C108 新材料 写/删</strong><span>属性 4/4 · 4/4 / 施事 0/4 · 3/4</span>
            <strong>C108 平均真值翻转</strong><span>属性 {Number(c104Data.freshC108?.family_rollup?.find((row) => row.family === 'attribute_binding')?.mean_truth_target_flip_rate || 0).toFixed(3)} / 施事 {Number(c104Data.freshC108?.family_rollup?.find((row) => row.family === 'agent_patient')?.mean_truth_target_flip_rate || 0).toFixed(3)}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>C103 frozen source vs C104 lockbox: truth, code, interaction</span>
            <b>{c104Data.dimensions[0]} ... {c104Data.dimensions[c104Data.dimensions.length - 1]}</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c104Data.effectRows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c104Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          {c104Data.supportRows.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C106 raw truth-response discovery support</span>
                <b>1 = selected activation coordinate</b>
              </div>
              <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
                {c104Data.supportRows.map((row) => (
                  <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': c104Data.dimensions.length }}>
                      {row.cells.map((cell) => (
                        <i
                          key={cell.dimension}
                          title={`${row.label} / activation coordinate ${cell.dimension}: ${cell.raw ? 'selected' : 'not selected'}`}
                          style={{ background: cell.raw ? '#22c55e' : 'rgba(30,41,59,0.42)' }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          <div className="research-heatmap-card__coordinate-axis">
            <span>Frozen role span: embedding + all Hidden States</span>
            <b>token, subtoken, role and state preserved</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c104Data.rawRows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.case_id}-${row.token_position}-${row.state}-${row.subtoken}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c104Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`${row.case_id} / token ${row.token_id} ${row.token_text} / ${row.role} / ${row.state_kind} ${row.state} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c104Data.claimBoundary || C104_UPSTREAM_ROLE_BARCODE_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c109Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C109-C110 Prospective Role-State Readout Control Atlas">
          <header>
            <strong>{C109_ROLE_STATE_FIELD_ATLAS_ROUTE.title}</strong>
            <span>Phase {c109Data.phase} / {c109Data.model} / {c109Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>属性 query@S19</strong><span>cos {Number(c109Data.candidates.get('attribute_binding')?.cross_partition_cosine || 0).toFixed(4)}</span>
            <strong>施事 query@S19</strong><span>cos {Number(c109Data.candidates.get('agent_patient')?.cross_partition_cosine || 0).toFixed(4)}</span>
            <strong>边界稳定起点</strong><span>属性 S{c109Data.boundaryLocators.attribute_binding ?? 'N/A'} / 施事 S{c109Data.boundaryLocators.agent_patient ?? 'N/A'}</span>
            <strong>单位L2杠杆胜出</strong><span>属性 {c109Data.leverageRollup.attribute_binding}/96 / 施事 {c109Data.leverageRollup.agent_patient}/96</span>
            <strong>C110新词场</strong><span>属性 cos {Number(c109Data.freshC110?.field_prediction?.results?.find((row) => row.family === 'attribute_binding')?.cross_fresh_partition_cosine || 0).toFixed(4)} / 施事 cos {Number(c109Data.freshC110?.field_prediction?.results?.find((row) => row.family === 'agent_patient')?.cross_fresh_partition_cosine || 0).toFixed(4)}</span>
            <strong>C110等能量杠杆</strong><span>属性目标胜 4/4 / 施事目标负 4/4</span>
            <strong>坐标值身份门</strong><span>属性目标中位胜置换 {c109Data.freshC110?.transport_pair_contrasts?.filter((row) => row.family === 'attribute_binding' && row.frozen_support_median_gt_permuted).length || 0}/4</span>
            <strong>双角色附加翻转</strong><span>{c109Data.freshC110?.transport_pair_contrasts?.reduce((sum, row) => sum + Number(row.query_plus_record_additional_truth_flips || 0), 0) || 0}/192</span>
            <strong>C111置换L2/目标</strong><span>属性 {Number(c109Data.c111Observation?.family_rollup?.attribute_binding?.permuted_to_target_l2_ratio_range?.[0] || 0).toFixed(2)}-{Number(c109Data.c111Observation?.family_rollup?.attribute_binding?.permuted_to_target_l2_ratio_range?.[1] || 0).toFixed(2)} / 施事 {Number(c109Data.c111Observation?.family_rollup?.agent_patient?.permuted_to_target_l2_ratio_range?.[0] || 0).toFixed(2)}-{Number(c109Data.c111Observation?.family_rollup?.agent_patient?.permuted_to_target_l2_ratio_range?.[1] || 0).toFixed(2)}</span>
            <strong>记录角色正增益</strong><span>属性 {c109Data.c111Observation?.family_rollup?.attribute_binding?.positive_focus_record_increment_pairs || 0}/96 / 施事 {c109Data.c111Observation?.family_rollup?.agent_patient?.positive_focus_record_increment_pairs || 0}/96</span>
            <strong>C112坐标赋值</strong><span>属性胜全部8置换 {c109Data.c112Batch?.family_rollup?.attribute_binding?.frozen_support_gt_all_permutation_medians_cells || 0}/4</span>
            <strong>C112属性翻转</strong><span>query/path/all {c109Data.c112Batch?.family_rollup?.attribute_binding?.query_truth_flips || 0}/{c109Data.c112Batch?.family_rollup?.attribute_binding?.record_path_truth_flips || 0}/{c109Data.c112Batch?.family_rollup?.attribute_binding?.all_role_truth_flips || 0}</span>
            <strong>C112施事翻转</strong><span>query/path/all {c109Data.c112Batch?.family_rollup?.agent_patient?.query_truth_flips || 0}/{c109Data.c112Batch?.family_rollup?.agent_patient?.record_path_truth_flips || 0}/{c109Data.c112Batch?.family_rollup?.agent_patient?.all_role_truth_flips || 0}</span>
            <strong>置换L2最大误差</strong><span>{Number(c109Data.c112Batch?.max_permutation_l2_relative_error || 0).toExponential(2)}</span>
            <strong>C113第四词汇场</strong><span>属性 cos {Number(c109Data.c113Batch?.field_prediction?.results?.find((row) => row.family === 'attribute_binding')?.cross_partition_cosine || 0).toFixed(4)} / 施事 cos {Number(c109Data.c113Batch?.field_prediction?.results?.find((row) => row.family === 'agent_patient')?.cross_partition_cosine || 0).toFixed(4)}</span>
            <strong>C113属性坐标赋值</strong><span>胜中位 4/4 / 胜全部 {c109Data.c113Batch?.predictions?.attribute_frozen_gt_all_permutation_cells || 0}/4</span>
            <strong>C113施事联合场</strong><span>path&gt;query {c109Data.c113Batch?.predictions?.agent_record_path_gt_query_cells || 0}/4 / all&gt;path {c109Data.c113Batch?.predictions?.agent_all_roles_gt_path_cells || 0}/4</span>
            <strong>C113留一角色</strong><span>移除 query_anchor 降低 {c109Data.c113Batch?.predictions?.agent_leave_query_anchor_lowers_cells || 0}/4 / query_focus {c109Data.c113Batch?.predictions?.agent_leave_query_focus_lowers_cells || 0}/4</span>
            <strong>C113行为 标准/反转</strong><span>{Number(c109Data.c113Batch?.behavior?.by_code?.['1'] || 0).toFixed(4)} / {Number(c109Data.c113Batch?.behavior?.by_code?.['-1'] || 0).toFixed(4)}</span>
            <strong>C114跨词汇坐标规律</strong><span>胜置换中位 属性 {c109Data.c114Atlas?.rollups?.attribute_binding?.beats_permutation_median_cells || 0}/8 / 施事 {c109Data.c114Atlas?.rollups?.agent_patient?.beats_permutation_median_cells || 0}/8</span>
            <strong>C114严格赋值</strong><span>胜全部置换 属性 {c109Data.c114Atlas?.rollups?.attribute_binding?.strictly_beats_all_permutations_cells || 0}/8 / 施事 {c109Data.c114Atlas?.rollups?.agent_patient?.strictly_beats_all_permutations_cells || 0}/8</span>
            <strong>C114施事查询联合</strong><span>path&gt;query {c109Data.c114Atlas?.rollups?.agent_patient?.path_gt_query_cells || 0}/8 / all&gt;path {c109Data.c114Atlas?.rollups?.agent_patient?.all_gt_path_cells || 0}/8</span>
            <strong>C115第五词汇场</strong><span>属性 cos {Number(c109Data.c115Batch?.field_prediction?.results?.find((row) => row.family === 'attribute_binding')?.cross_partition_cosine || 0).toFixed(4)} / 施事 cos {Number(c109Data.c115Batch?.field_prediction?.results?.find((row) => row.family === 'agent_patient')?.cross_partition_cosine || 0).toFixed(4)}</span>
            <strong>C115坐标赋值</strong><span>胜中位 属性 {c109Data.c115Batch?.predictions?.attribute_median_win_cells || 0}/4 / 施事 {c109Data.c115Batch?.predictions?.agent_median_win_cells || 0}/4</span>
            <strong>C115施事角色联合</strong><span>path&gt;query {c109Data.c115Batch?.predictions?.agent_record_path_gt_query_cells || 0}/4 / 留 anchor/focus/post {c109Data.c115Batch?.predictions?.agent_leave_query_anchor_lowers_cells || 0}/{c109Data.c115Batch?.predictions?.agent_leave_query_focus_lowers_cells || 0}/{c109Data.c115Batch?.predictions?.agent_leave_focus_post_lowers_cells || 0}</span>
            <strong>C116发现候选</strong><span>{c109Data.c116Batch?.nomination?.role || 'N/A'}@S{c109Data.c116Batch?.nomination?.state ?? 'N/A'} / K{c109Data.c116Batch?.nomination?.support_k || 0}</span>
            <strong>C116否定作用域场</strong><span>confirm-lockbox cos {Number(c109Data.c116Batch?.validation?.field_metrics?.confirmation_lockbox_cosine || 0).toFixed(4)} / support {Number(c109Data.c116Batch?.validation?.field_metrics?.confirmation_support_overlap || 0).toFixed(3)}-{Number(c109Data.c116Batch?.validation?.field_metrics?.lockbox_support_overlap || 0).toFixed(3)}</span>
            <strong>C116规律与边界</strong><span>坐标胜中位 {c109Data.c116Batch?.validation?.predictions?.correct_movement_gt_permutation_median_cells || 0}/4 / path&gt;query {c109Data.c116Batch?.validation?.predictions?.path_gt_query_cells || 0}/4</span>
            <strong>C117发现候选</strong><span>{c109Data.c117Batch?.nomination?.role || 'N/A'}@S{c109Data.c117Batch?.nomination?.state ?? 'N/A'} / K{c109Data.c117Batch?.nomination?.support_k || 0}</span>
            <strong>C117显式例外日志场</strong><span>confirm-lockbox cos {Number(c109Data.c117Batch?.validation?.field_metrics?.confirmation_lockbox_cosine || 0).toFixed(4)} / support {Number(c109Data.c117Batch?.validation?.field_metrics?.confirmation_support_overlap || 0).toFixed(3)}-{Number(c109Data.c117Batch?.validation?.field_metrics?.lockbox_support_overlap || 0).toFixed(3)}</span>
            <strong>C117赋值与路线</strong><span>坐标胜中位 {c109Data.c117Batch?.validation?.predictions?.correct_movement_gt_permutation_median_cells || 0}/4 / path&gt;query {c109Data.c117Batch?.validation?.predictions?.path_gt_query_cells_descriptive || 0}/4</span>
            <strong>C117公共轴投影/条件残差</strong><span>公共轴 cos {Number(c109Data.c117Batch?.validation?.common_component_residual?.whole_part_to_common?.confirmation || 0).toFixed(4)} / 残差跨分区 {Number(c109Data.c117Batch?.validation?.common_component_residual?.residual_cross_partition_cosine || 0).toFixed(4)}，仅描述性几何</span>
            <strong>C123-C124转移复验</strong><span>全向量/坐标时钟/错状态/错角色 {c109Data.transitionBatch?.validation?.level_counts?.level_1_full_vector || 0}/{c109Data.transitionBatch?.validation?.level_counts?.level_2_coordinate_clock || 0}/{c109Data.transitionBatch?.validation?.level_counts?.level_3_state_specific || 0}/{c109Data.transitionBatch?.validation?.level_counts?.level_3_role_specific || 0} of {c109Data.transitionBatch?.results?.length || 0}</span>
            <strong>C125末端分解门</strong><span>{c109Data.c125Batch?.adjudication?.results?.filter((row) => row.frozen_prediction_passed).length || 0}/{c109Data.c125Batch?.adjudication?.results?.length || 0}；行为 {Number(c109Data.c125Batch?.adjudication?.behavior?.global_accuracy || 0).toFixed(4)}</span>
            <strong>C126真值效应排名</strong><span>{c109Data.c126Batch?.adjudication?.truth_effects?.filter((row) => row.signed_rank === 1).length || 0}/{c109Data.c126Batch?.adjudication?.truth_effects?.length || 0}；析因重构误差 {Number(c109Data.c126Batch?.adjudication?.reconstruction_max_abs || 0).toExponential(2)}</span>
            <strong>C129直接先后关系</strong><span>{c109Data.c129Batch?.confirmation?.nominee?.role || 'N/A'} / block {Number(c109Data.c129Batch?.confirmation?.nominee?.transition_index ?? -1) + 1}；confirm cos {Number(c109Data.c129Batch?.confirmation?.metrics?.target_cosine || 0).toFixed(4)}</span>
            <strong>C129坐标复现</strong><span>Top-256 {Number(c109Data.c129Batch?.confirmation?.metrics?.top256_overlap || 0).toFixed(3)} / 时钟 {Number(c109Data.c129Batch?.confirmation?.metrics?.coordinate_clock_within_one || 0).toFixed(3)}</span>
          </div>
          {c109Data.c112ModeRows.length > 0 && c109Data.c112ModeRows[0].cells.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C112等L2坐标与角色响应格</span>
                <b>属性/施事 × confirmation/lockbox × code</b>
              </div>
              <div className="research-heatmap-card__relation-scroll">
                {c109Data.c112ModeRows.map((row) => (
                  <div className="research-heatmap-card__relation-row" key={`c112-${row.key}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': row.cells.length }}>
                      {row.cells.map((cell) => (
                        <i
                          key={cell.key}
                          title={`${row.label} / ${cell.family} / ${cell.partition} / code ${cell.code} / median raw truth gain ${cell.value.toPrecision(7)}`}
                          style={{ background: signedColor(cell.value / c109Data.c112Scale, true) }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c114Rows.length > 0 && c109Data.c114Rows[0].cells.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C114跨 C112-C113 描述性结构图谱</span>
                <b>2 datasets × 2 families × 2 partitions × 2 codes</b>
              </div>
              <div className="research-heatmap-card__relation-scroll">
                {c109Data.c114Rows.map((row) => (
                  <div className="research-heatmap-card__relation-row" key={`c114-${row.key}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': row.cells.length }}>
                      {row.cells.map((cell) => (
                        <i
                          key={cell.key}
                          title={`${row.label} / ${cell.dataset} / ${cell.family} / ${cell.partition} / code ${cell.code}: ${cell.value.toPrecision(7)}`}
                          style={{ background: signedColor(cell.value / c109Data.c114Scale, true) }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c113ModeRows.length > 0 && c109Data.c113ModeRows[0].cells.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C113第四词汇：等L2坐标、分阶段联盟与留一角色</span>
                <b>属性/施事 × confirmation/lockbox × code</b>
              </div>
              <div className="research-heatmap-card__relation-scroll">
                {c109Data.c113ModeRows.map((row) => (
                  <div className="research-heatmap-card__relation-row" key={`c113-${row.key}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': row.cells.length }}>
                      {row.cells.map((cell) => (
                        <i
                          key={cell.key}
                          title={`${row.label} / ${cell.family} / ${cell.partition} / code ${cell.code} / median raw truth gain ${cell.value.toPrecision(7)}`}
                          style={{ background: signedColor(cell.value / c109Data.c113Scale, true) }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c115Rows.length > 0 && c109Data.c115Rows[0].cells.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C115第五词汇大样本：坐标赋值与角色联合</span>
                <b>48 independent lexical units / 384 intervention pairs</b>
              </div>
              <div className="research-heatmap-card__relation-scroll">
                {c109Data.c115Rows.map((row) => (
                  <div className="research-heatmap-card__relation-row" key={`c115-${row.key}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': row.cells.length }}>
                      {row.cells.map((cell) => (
                        <i key={cell.key} title={`${row.label} / ${cell.family} / ${cell.partition} / code ${cell.code}: ${cell.value.toPrecision(7)}`} style={{ background: signedColor(cell.value / c109Data.c115Scale, true) }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c116Rows.length > 0 && c109Data.c116Rows[0].cells.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C116否定作用域：discovery冻结候选与独立验证</span>
                <b>boundary@S30 / confirmation + lockbox</b>
              </div>
              <div className="research-heatmap-card__relation-scroll">
                {c109Data.c116Rows.map((row) => (
                  <div className="research-heatmap-card__relation-row" key={`c116-${row.key}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': row.cells.length }}>
                      {row.cells.map((cell) => (
                        <i key={cell.key} title={`${row.label} / ${cell.partition} / code ${cell.code}: ${cell.value.toPrecision(7)}`} style={{ background: signedColor(cell.value / c109Data.c116Scale, true) }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c117Rows.length > 0 && c109Data.c117Rows[0].cells.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C117带默认背景的显式例外日志：冻结候选、公共轴投影与条件残差</span>
                <b>boundary@S30 / confirmation + lockbox</b>
              </div>
              <div className="research-heatmap-card__relation-scroll">
                {c109Data.c117Rows.map((row) => (
                  <div className="research-heatmap-card__relation-row" key={`c117-${row.key}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': row.cells.length }}>
                      {row.cells.map((cell) => (
                        <i key={cell.key} title={`${row.label} / ${cell.partition} / code ${cell.code}: ${cell.value.toPrecision(7)}`} style={{ background: signedColor(cell.value / c109Data.c117Scale, true) }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.transitionProfiles.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C123-C124 登记角色响应增量轮廓</span>
                <b>36 recorded transitions; S35-&gt;S36 is instrument-heterogeneous</b>
              </div>
              <div className="research-heatmap-card__relation-scroll">
                {c109Data.transitionProfiles.map((row) => (
                  <div className="research-heatmap-card__relation-row" key={`transition-profile-${row.family}-${row.partition}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': row.cells.length }}>
                      {row.cells.map((cell) => (
                        <i key={cell.state} title={`${row.label} / S${cell.state - 1}->S${cell.state} / L2 ${cell.value.toPrecision(7)}`} style={{ background: signedColor(cell.value / c109Data.transitionProfileScale, true) }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.transitionCoordinateRows.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C123-C124 状态与增量物理激活坐标</span>
                <b>embedding + selected HiddenStates + typed increments</b>
              </div>
              <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
                {c109Data.transitionCoordinateRows.map((row) => (
                  <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                      {row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c125Rows.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C125 最终块与最终归一化响应分解</span>
                <b>full 2560 activation coordinates; 2/3 frozen cells passed</b>
              </div>
              <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
                {c109Data.c125Rows.map((row) => (
                  <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                      {row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c126Rows.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C126 真值与答案码交互响应</span>
                <b>2 of 15 effects shown; all effects remain in the asset</b>
              </div>
              <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
                {c109Data.c126Rows.map((row) => (
                  <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                      {row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c129Profiles.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C129 行为合格直接先后关系：统一类型增量轮廓</span>
                <b>embedding + 36 post-block pre-norm + final norm</b>
              </div>
              <div className="research-heatmap-card__relation-scroll">
                {c109Data.c129Profiles.map((row) => (
                  <div className="research-heatmap-card__relation-row" key={row.label}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': row.cells.length }}>
                      {row.cells.map((cell) => <i key={cell.transition} title={`${row.label} / typed transition ${cell.transition}: ${cell.value.toPrecision(7)}`} style={{ background: signedColor(cell.value / c109Data.c129ProfileScale, true) }} />)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c129Rows.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C129 真值响应状态与增量</span>
                <b>discovery + confirmation / full 2560 coordinates in asset</b>
              </div>
              <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
                {c109Data.c129Rows.map((row) => (
                  <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                      {row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c129RawRows.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C129 代表样本原始词嵌入与 HiddenState</span>
                <b>token role + exact typed checkpoint + activation coordinate</b>
              </div>
              <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
                {c109Data.c129RawRows.map((row) => (
                  <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                      {row.cells.map((cell) => <i key={cell.dimension} title={`${row.case_id} / ${row.role} / ${row.checkpoint} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c139Rows.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C135-C138 参数级观察场</span>
                <b>embedding + typed HiddenState / all 2560 activation coordinates in asset</b>
              </div>
              <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
                {c109Data.c139Rows.map((row, index) => (
                  <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`c139-${row.dataset}-${row.kind}-${index}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                      {row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c148Rows.length > 0 && (
            <>
              <div className="research-heatmap-card__summary">
                <strong>C142复现</strong><span>{c109Data.c148Batch?.c142?.confirmation?.passing_semantic_nominees || 0}/{c109Data.c148Batch?.c142?.confirmation?.total_semantic_nominees || 0}</span>
                <strong>C143轨迹门</strong><span>{c109Data.c148Batch?.c143?.confirmation?.prediction_gate_passed ? 'pass' : 'fail'}</span>
                <strong>C144重建</strong><span>order {c109Data.c148Batch?.c144?.confirmation?.frozen_order || 'N/A'} / {c109Data.c148Batch?.c144?.confirmation?.composition_gate_passed ? 'aggregate pass' : 'fail'}</span>
                <strong>C146共同接口</strong><span>{c109Data.c148Batch?.c146?.confirmation?.winner || 'none'}</span>
                <strong>C153类型图预测</strong><span>{c109Data.c153Confirmation?.confirmation_gate_passed ? 'pass' : 'fail'}</span>
                <strong>C154因果身份</strong><span>{c109Data.c154Causal?.causal_gate_passed ? 'pass' : 'fail'}</span>
                <strong>C155检查点图</strong><span>{c109Data.c155Transfer ? `best q${c109Data.c155Transfer.best_state} / ${c109Data.c155Transfer.broad_checkpoint_count} broad` : 'N/A'}</span>
              </div>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C141-C145 多语言族逐坐标观察</span>
                <b>embedding + HiddenState / all 2560 activation coordinates in asset</b>
              </div>
              <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
                {c109Data.c148Rows.map((row, index) => (
                  <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`c148-${row.dataset}-${row.kind}-${index}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                      {row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / HiddenState activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {c109Data.c111TrajectoryRows.length > 0 && (
            <>
              <div className="research-heatmap-card__coordinate-axis">
                <span>C111跨词表角色形成图</span>
                <b>S0 embedding / S1-S36 Hidden State</b>
              </div>
              <div className="research-heatmap-card__relation-scroll">
                {c109Data.c111TrajectoryRows.map((row) => (
                  <div className="research-heatmap-card__relation-row" key={`c111-${row.family}-${row.role}`}>
                    <span>{row.label}</span>
                    <div style={{ '--relation-columns': 37 }}>
                      {row.cells.map((cell) => (
                        <i
                          key={cell.state}
                          title={`${row.label} / S${cell.state} / C109-C110 cosine ${cell.value.toPrecision(6)} / C110 partition cosine ${cell.crossPartition.toPrecision(6)} / norms ${cell.oldNorm.toPrecision(5)} -> ${cell.newNorm.toPrecision(5)}`}
                          style={{ background: signedColor(cell.value, true) }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          <div className="research-heatmap-card__coordinate-axis">
            <span>Balanced truth field: registered roles across selected states</span>
            <b>{c109Data.dimensions[0]} ... {c109Data.dimensions[c109Data.dimensions.length - 1]}</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c109Data.effectRows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>Frozen support masks</span>
            <b>green = selected coordinate</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c109Data.supportRows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.label}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`${row.label} / activation coordinate ${cell.dimension}: ${cell.raw ? 'selected' : 'not selected'}`}
                      style={{ background: cell.raw ? '#22c55e' : 'rgba(30,41,59,0.42)' }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>Representative raw embedding + all-state samples</span>
            <b>token, position, state and 2560 coordinates preserved</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c109Data.rawRows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.case_id}-${row.state}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c109Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i
                      key={cell.dimension}
                      title={`${row.case_id} / token ${row.token_id} ${row.token_text} / ${row.role} / ${row.state_kind} ${row.state} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`}
                      style={{ background: signedColor(cell.value, true) }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c109Data.claimBoundary || C109_ROLE_STATE_FIELD_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c157C166Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C157-C166 Local Field Coordinate Heatmap">
          <header>
            <strong>{C157_C166_LOCAL_FIELD_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c157C166Data.phase} / {c157C166Data.model} / {c157C166Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>C160新词预测</strong><span>cos {Number(c157C166Data.summaries?.C160?.fresh?.aggregate?.median_cosine || 0).toFixed(4)}</span>
            <strong>C161通用传动</strong><span>{c157C166Data.c161?.first_order_replication_passed ? 'pass' : 'fail'}</span>
            <strong>C161关系特异</strong><span>{c157C166Data.c161?.generic_transport_diagnostic?.relation_specific_transport_supported ? 'pass' : 'candidate only'}</span>
            <strong>C163自然调用</strong><span>{c157C166Data.summaries?.C163?.headline?.gates?.controls ? 'pass' : 'fail'}</span>
            <strong>C164共同接口</strong><span>{c157C166Data.c164?.preferred_common_interface || 'none'}</span>
            <strong>C165跨模型拓扑</strong><span>{c157C166Data.c165?.status?.startsWith('typed_not_tested') ? 'not tested' : (c157C166Data.c165?.topology_gate_passed ? 'pass' : 'fail')}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>Embedding, predicted/actual field, local transmission and linguistic terms</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c157C166Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.dataset}-${row.kind}-${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c157C166Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c157C166Data.coordinateSemantics} {c157C166Data.claimBoundary || C157_C166_LOCAL_FIELD_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c167C168Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C167-C168 Relation Residual Coordinate Heatmap">
          <header>
            <strong>{C167_C168_RELATION_RESIDUAL_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c167C168Data.phase} / {c167C168Data.model} / {c167C168Data.dimensions.length} target coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>fresh匹配余弦</strong><span>{Number(c167C168Data.c168?.headline?.aggregate?.matched_median_cosine || 0).toFixed(4)}</span>
            <strong>错关系裕量</strong><span>{Number(c167C168Data.c168?.headline?.aggregate?.relation_margin || 0).toFixed(4)}</span>
            <strong>去同坐标裕量</strong><span>{Number(c167C168Data.c168?.headline?.aggregate?.identity_removed_relation_margin || 0).toFixed(4)}</span>
            <strong>错源坐标优势</strong><span>{Number(c167C168Data.c168?.headline?.aggregate?.source_permutation_advantage || 0).toFixed(4)}</span>
            <strong>前瞻门</strong><span>{c167C168Data.c168?.headline?.passed ? '5/5 pass' : 'failed'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>old reference / fresh relation component by q24 source coordinate and q25 role</span>
            <b>Qwen3 q25 target activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c167C168Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.relation}-${row.source_coordinate}-${row.target_role}-${row.split}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c167C168Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / q25 activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c167C168Data.coordinateSemantics} {c167C168Data.claimBoundary || C167_C168_RELATION_RESIDUAL_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c170Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C170 Role Checkpoint Coordinate Heatmap">
          <header>
            <strong>{C170_ROLE_CHECKPOINT_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c170Data.phase} / {c170Data.model} / {c170Data.dimensions.length} target coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>stable</strong><span>{c170Data.c170?.headline?.label_counts?.stable || 0}</span>
            <strong>partial</strong><span>{c170Data.c170?.headline?.label_counts?.partial || 0}</span>
            <strong>absent</strong><span>{c170Data.c170?.headline?.label_counts?.absent || 0}</span>
            <strong>角色拓扑</strong><span>relation &gt; query &gt; primary</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>fresh relation component by source checkpoint, source role, source coordinate and target role</span>
            <b>Qwen3 next-checkpoint target activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c170Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.source_checkpoint}-${row.source_role}-${row.relation}-${row.source_coordinate}-${row.target_role}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c170Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / target activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c170Data.coordinateSemantics} {c170Data.claimBoundary || C170_ROLE_CHECKPOINT_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c183Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C183 Natural Relation Response Ecology Heatmap">
          <header>
            <strong>{C183_NATURAL_RESPONSE_ECOLOGY_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c183Data.phase} / {c183Data.model} / {c183Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c183Data.totalRows} rows x 2560</span>
            <strong>query</strong><span>{c183Data.synthesis?.response_summary?.query?.externality_label || 'N/A'}</span>
            <strong>relation</strong><span>{c183Data.synthesis?.response_summary?.relation?.externality_label || 'N/A'}</span>
            <strong>primary</strong><span>{c183Data.synthesis?.response_summary?.primary?.externality_label || 'N/A'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>embedding / HiddenState role states and fresh q24-to-q25 signed local responses</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c183Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.kind}-${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c183Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c183Data.coordinateSemantics} {c183Data.claimBoundary || C183_NATURAL_RESPONSE_ECOLOGY_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c189Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C189 New Material Response Scaffold Heatmap">
          <header>
            <strong>{C189_NEW_MATERIAL_RESPONSE_SCAFFOLD_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c189Data.phase} / {c189Data.model} / {c189Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c189Data.totalRows} rows x 2560</span>
            <strong>稳定层</strong><span>generic relation-source scaffold</span>
            <strong>可变层</strong><span>phrase/context-conditioned field</span>
            <strong>跨模型</strong><span>内部拓扑尚未测试</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>new-vocabulary and paraphrase q24-to-q25 target energy / signed mean response</span>
            <b>Qwen3 q25 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c189Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.kind}-${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c189Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / q25 activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c189Data.coordinateSemantics} {c189Data.claimBoundary || C189_NEW_MATERIAL_RESPONSE_SCAFFOLD_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c191Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C191 Response Equivalence Atlas Heatmap">
          <header>
            <strong>{C191_RESPONSE_EQUIVALENCE_ATLAS_ROUTE.title}</strong>
            <span>Phase {c191Data.phase} / {c191Data.model} / {c191Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c191Data.totalRows} rows x 2560</span>
            <strong>最近邻同关系族</strong><span>{((c191Data.summary?.family?.nearest_match_rate || 0) * 100).toFixed(1)}%</span>
            <strong>同族基线优势</strong><span>{Number(c191Data.summary?.family?.advantage || 0).toFixed(3)}</span>
            <strong>登记缺失</strong><span>{c191Data.missing} / 56</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>missing-aware factorial cells; normalized q24 relation-source to q25 target energy profiles</span>
            <b>Qwen3 q25 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c191Data.rows.map((row) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={row.cell_index}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c191Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / q25 activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c191Data.coordinateSemantics} {c191Data.claimBoundary || C191_RESPONSE_EQUIVALENCE_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c193Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C193 Program-Centered Response Residual Heatmap">
          <header>
            <strong>{C193_PROGRAM_CENTERED_RESIDUAL_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c193Data.phase} / {c193Data.model} / {c193Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c193Data.totalRows} rows x 2560</span>
            <strong>跨关系词同族率</strong><span>{((c193Data.rawSummary?.cross_phrase?.same_family_rate || 0) * 100).toFixed(1)}%</span>
            <strong>严格原谱同族率</strong><span>{((c193Data.rawSummary?.strict_cross_all?.same_family_rate || 0) * 100).toFixed(1)}%</span>
            <strong>程序中心化后</strong><span>{((c193Data.residualResult?.same_family_rate || 0) * 100).toFixed(1)}% / 未通过</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>signed family-excluded program-centered response residuals</span>
            <b>Qwen3 q25 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c193Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.case_id}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c193Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / centered q25 activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c193Data.coordinateSemantics} {c193Data.claimBoundary || C193_PROGRAM_CENTERED_RESIDUAL_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c202Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C202 Signed Operator Campaign Heatmap">
          <header>
            <strong>{C202_SIGNED_OPERATOR_CAMPAIGN_HEATMAP_ROUTE.title}</strong>
            <span>Phase {c202Data.phase} / {c202Data.model} / {c202Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c202Data.totalRows} rows x 2560</span>
            <strong>有符号延续</strong><span>{Number(c202Data.summary?.c195_sign_persistence || 0).toFixed(3)}</span>
            <strong>跨程序预测</strong><span>{c202Data.summary?.c197_primary_gate ? '通过' : '未通过'}</span>
            <strong>自然因果</strong><span>{c202Data.summary?.c200_causal_tested ? '已测试' : '类型化未测试'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>embedding / HiddenState baseline and signed q23 to q24/q25 response</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c202Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c202Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c202Data.coordinateSemantics} {c202Data.claimBoundary || C202_SIGNED_OPERATOR_CAMPAIGN_HEATMAP_ROUTE.boundary}</p>
        </section>
      )}
      {c215Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C215 Response Interval and Composition Atlas Heatmap">
          <header>
            <strong>{C215_RESPONSE_INTERVAL_COMPOSITION_ATLAS_ROUTE.title}</strong>
            <span>Phase {c215Data.phase} / {c215Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c215Data.totalRows} rows x 2560</span>
            <strong>自然q24→q25符号</strong><span>{Number(c215Data.summary?.natural_q24_q25_sign || 0).toFixed(3)}</span>
            <strong>路径组合NRMSE</strong><span>{Number(c215Data.summary?.path_composition_fresh_nrmse || 0).toFixed(3)}</span>
            <strong>理论准备度</strong><span>{c215Data.summary?.theory_readiness_passed || 0}/{c215Data.summary?.theory_readiness_total || 5}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>baseline / signed intervention response / additive prediction / interaction residual</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c215Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c215Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c215Data.coordinateSemantics} {c215Data.claimBoundary || C215_RESPONSE_INTERVAL_COMPOSITION_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c220Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C220 Response-State Minimality Atlas Heatmap">
          <header>
            <strong>{C220_RESPONSE_STATE_MINIMALITY_ATLAS_ROUTE.title}</strong>
            <span>Phase {c220Data.phase} / {c220Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c220Data.totalRows} rows x 2560</span>
            <strong>最小观测</strong><span>{c220Data.summary?.selected_subset || 'N/A'}</span>
            <strong>新鲜五分类</strong><span>{Number(c220Data.summary?.fresh?.accuracy || 0).toFixed(3)}</span>
            <strong>坐标置乱</strong><span>{Number(c220Data.summary?.negative_controls?.query_coordinate_permutation?.accuracy || 0).toFixed(3)}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>factor A / factor B / interaction at q24-q25 relation and boundary roles</span>
            <b>Qwen3 physical HiddenState coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c220Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c220Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / HiddenState coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c220Data.coordinateSemantics} {c220Data.claimBoundary || C220_RESPONSE_STATE_MINIMALITY_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c222Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C222 Surface-Conditioned Signed Response Atlas Heatmap">
          <header>
            <strong>{C222_SURFACE_CONDITIONED_RESPONSE_ATLAS_ROUTE.title}</strong>
            <span>Phase {c222Data.phase} / {c222Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c222Data.totalRows} rows x 2560</span>
            <strong>同表面迁移</strong><span>{Number(c222Data.summary?.within_C221_surface?.accuracy || 0).toFixed(3)}</span>
            <strong>共同偏移</strong><span>{Number(c222Data.summary?.common_surface_offset?.accuracy || 0).toFixed(3)}</span>
            <strong>面板中心化</strong><span>{Number(c222Data.summary?.panel_centered_arm_residual?.accuracy || 0).toFixed(3)}</span>
            <strong>同表面场NRMSE</strong><span>{Number(c222Data.summary?.exact_field?.C221_confirmation_raw_to_fresh?.median_nrmse || 0).toFixed(3)}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>surface template / fresh response / signed cross-surface difference</span>
            <b>Qwen3 physical HiddenState coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c222Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c222Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / HiddenState coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c222Data.coordinateSemantics} {c222Data.claimBoundary || C222_SURFACE_CONDITIONED_RESPONSE_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c233Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C223-C233 Surface Transport and Composition Atlas Heatmap">
          <header>
            <strong>{C233_SURFACE_TRANSPORT_COMPOSITION_ATLAS_ROUTE.title}</strong>
            <span>Phase {c233Data.phase} / {c233Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c233Data.totalRows} rows x 2560</span>
            <strong>运输锁箱</strong><span>{c233Data.summary?.transport_lockbox_passed ? 'PASS' : 'FAIL'}</span>
            <strong>运输NRMSE</strong><span>{Number(c233Data.summary?.transport_selected_nrmse || 0).toFixed(3)}</span>
            <strong>组合通过族</strong><span>{c233Data.summary?.composition_families_passed || 0}/5</span>
            <strong>三模型拓扑</strong><span>{c233Data.summary?.cross_model_passed ? 'PASS' : 'FAIL'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>passport / transport prediction-truth-error / composition prediction-truth-interaction</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c233Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c233Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c233Data.coordinateSemantics} {c233Data.claimBoundary || C233_SURFACE_TRANSPORT_COMPOSITION_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c243Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C234-C243 Conditional Event Atlas Heatmap">
          <header>
            <strong>{C243_CONDITIONAL_EVENT_ATLAS_ROUTE.title}</strong>
            <span>Phase {c243Data.phase} / {c243Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>完整资产</strong><span>{c243Data.totalRows} rows x 2560</span>
            <strong>未见事件族</strong><span>{c243Data.summary?.unseen_event_families_passed || 0}/5</span>
            <strong>组合通过族</strong><span>{c243Data.summary?.composition_families_passed || 0}/5</span>
            <strong>因果状态</strong><span>{c243Data.summary?.causal_status || 'N/A'}</span>
            <strong>数学升级门</strong><span>{c243Data.summary?.mathematical_upgrade_gates_passed || 0}/{c243Data.summary?.mathematical_upgrade_gate_total || 4}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>family / factorial effect / embedding-or-HiddenState checkpoint / semantic role</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c243Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c243Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)} / stable discovery events ${row.stable_event_count}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c243Data.coordinateSemantics} {c243Data.claimBoundary || C243_CONDITIONAL_EVENT_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c244Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C244 Independent Event Replication Heatmap">
          <header>
            <strong>{C244_INDEPENDENT_EVENT_REPLICATION_ROUTE.title}</strong>
            <span>Phase {c244Data.phase} / {c244Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>行为准确率</strong><span>{Number(c244Data.summary?.behavior_accuracy || 0).toFixed(3)}</span>
            <strong>事件通过族</strong><span>{(c244Data.summary?.families_passed || []).join(', ') || 'none'}</span>
            <strong>候选换序一致</strong><span>{Number(c244Data.summary?.candidate_order_signed_agreement_median || 0).toFixed(3)}</span>
            <strong>跨模型诊断</strong><span>{c244Data.summary?.cross_model_diagnostics_passed || 0}/5</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>new surface and vocabulary / family / effect / checkpoint / role</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c244Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c244Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)} / frozen rule events ${row.stable_discovery_event_count}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c244Data.coordinateSemantics} {c244Data.claimBoundary || C244_INDEPENDENT_EVENT_REPLICATION_ROUTE.boundary}</p>
        </section>
      )}
      {c245Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C245 Confirmed Signed Event Core Heatmap">
          <header>
            <strong>{C245_CONFIRMED_EVENT_CORE_ROUTE.title}</strong>
            <span>Phase {c245Data.phase} / {c245Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>确认事件</strong><span>{Number(c245Data.summary?.confirmed_events || 0).toLocaleString()}</span>
            <strong>保留比例</strong><span>{Number(c245Data.summary?.overall_retention || 0).toFixed(3)}</span>
            <strong>HiddenState事件</strong><span>{Number(c245Data.summary?.hidden_events || 0).toLocaleString()}</span>
            <strong>跨族Jaccard</strong><span>{Number(c245Data.summary?.cross_family_signed_jaccard_median || 0).toFixed(3)}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>confirmed signed event (-1 / 0 / +1)</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c245Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c245Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: event ${Number(cell.raw)} / row events ${row.confirmed_event_count}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c245Data.coordinateSemantics} {c245Data.claimBoundary || C245_CONFIRMED_EVENT_CORE_ROUTE.boundary}</p>
        </section>
      )}
      {c254Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C254 Tri-Material Event Atlas Heatmap">
          <header>
            <strong>{C254_TRI_MATERIAL_EVENT_ATLAS_ROUTE.title}</strong>
            <span>Phase {c254Data.phase} / {c254Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>三材料事件</strong><span>{Number(c254Data.summary?.tri_material_events || 0).toLocaleString()}</span>
            <strong>HiddenState事件</strong><span>{Number(c254Data.summary?.hidden_events || 0).toLocaleString()}</span>
            <strong>Token对齐覆盖</strong><span>{Number(c254Data.summary?.token_alignment_coverage || 0).toFixed(3)}</span>
            <strong>因果轨迹门</strong><span>{c254Data.summary?.causal_trajectory_passed ? '通过' : '未通过/未测试'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>tri-material event or full-token signed balance</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c254Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c254Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)} / event observations ${row.event_count}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c254Data.coordinateSemantics} {c254Data.claimBoundary || C254_TRI_MATERIAL_EVENT_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c260Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C260 Output Path Causal Atlas Heatmap">
          <header>
            <strong>{C260_OUTPUT_PATH_CAUSAL_ATLAS_ROUTE.title}</strong>
            <span>Phase {c260Data.phase} / {c260Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>最早通过前缀</strong><span>q1-q{c260Data.summary?.earliest_passing_prefix_end ?? 'N/A'}</span>
            <strong>前缀控制余量</strong><span>{Number(c260Data.summary?.prefix16_vs_best_control_margin || 0).toFixed(3)}</span>
            <strong>直接词控制余量</strong><span>{Number(c260Data.summary?.natural_word_control_margin || 0).toFixed(3)}</span>
            <strong>直接词门</strong><span>{c260Data.summary?.natural_word_gate_passed ? '通过' : '未通过'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>tri-material signed event (-1 / 0 / +1)</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c260Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c260Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: event ${Number(cell.raw)} / row events ${row.event_count}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c260Data.coordinateSemantics} {c260Data.claimBoundary || C260_OUTPUT_PATH_CAUSAL_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c262Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C262 Generation Specificity Atlas Heatmap">
          <header>
            <strong>{C262_GENERATION_SPECIFICITY_ATLAS_ROUTE.title}</strong>
            <span>Phase {c262Data.phase} / {c262Data.dimensions.length} coordinates</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>最早坐标覆盖</strong><span>{Number(c262Data.summary?.earliest_fraction || 0).toLocaleString(undefined, { style: 'percent' })}</span>
            <strong>中点擦除门</strong><span>{c262Data.summary?.midpoint_erasure_passed ? '通过（人工探针）' : '未通过'}</span>
            <strong>完整词控制余量</strong><span>{Number(c262Data.summary?.correct_minus_best_control || 0).toFixed(3)}</span>
            <strong>完整词特异性门</strong><span>{c262Data.summary?.full_word_generation_gate_passed ? '通过' : '失败'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>tri-material signed event (-1 / 0 / +1)</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c262Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c262Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: event ${Number(cell.raw)} / row events ${row.event_count}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c262Data.coordinateSemantics} {c262Data.claimBoundary || C262_GENERATION_SPECIFICITY_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c272Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C272 State Conditioned Operator Atlas Heatmap">
          <header>
            <strong>{C272_STATE_CONDITIONED_OPERATOR_ATLAS_ROUTE.title}</strong>
            <span>Phase {c272Data.phase} / {c272Data.dimensions.length} coordinates / {c272Data.totalRows} rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>行为资格</strong><span>{c272Data.summary?.behavior_eligible ? '通过' : '失败'}</span>
            <strong>逐坐标护照</strong><span>{c272Data.summary?.passport_families_passing ?? 0}/6</span>
            <strong>滚动/局部因果</strong><span>{c272Data.summary?.rolling_gate || c272Data.summary?.local_causal_gate ? '部分通过' : '均失败'}</span>
            <strong>跨模型功能拓扑</strong><span>{c272Data.summary?.cross_model_gate ? '通过' : '失败'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis">
            <span>activation / response / passport / interaction</span>
            <b>Qwen3 physical activation coordinate id</b>
          </div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c272Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c272Data.dimensions.length }}>
                  {row.cells.map((cell) => (
                    <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c272Data.coordinateSemantics} {c272Data.claimBoundary || C272_STATE_CONDITIONED_OPERATOR_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c273Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C273 Response Ecology Atlas Heatmap">
          <header>
            <strong>{C273_RESPONSE_ECOLOGY_ATLAS_ROUTE.title}</strong>
            <span>Phase {c273Data.phase} / {c273Data.dimensions.length} coordinates / {c273Data.totalRows} rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>同坐标持续</strong><span>{Number(c273Data.summary?.stable_fraction || 0).toLocaleString(undefined, { style: 'percent', maximumFractionDigits: 2 })}</span>
            <strong>新生事件</strong><span>{Number(c273Data.summary?.emergence_fraction || 0).toLocaleString(undefined, { style: 'percent', maximumFractionDigits: 2 })}</span>
            <strong>护照漏报</strong><span>{Number(c273Data.summary?.passport_missed_fraction || 0).toLocaleString(undefined, { style: 'percent', maximumFractionDigits: 2 })}</span>
            <strong>护照错报</strong><span>{Number(c273Data.summary?.passport_wrong_fraction || 0).toLocaleString(undefined, { style: 'percent', maximumFractionDigits: 2 })}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>per-coordinate event category fraction</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c273Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c273Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c273Data.coordinateSemantics} {c273Data.claimBoundary || C273_RESPONSE_ECOLOGY_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c275Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C275 Cross Role Reuse Atlas Heatmap">
          <header>
            <strong>{C275_CROSS_ROLE_REUSE_ATLAS_ROUTE.title}</strong>
            <span>Phase {c275Data.phase} / {c275Data.dimensions.length} coordinates / {c275Data.totalRows} rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>重复语言族</strong><span>{c275Data.summary?.families_passing ?? 0}/6</span>
            <strong>描述门</strong><span>{c275Data.summary?.descriptive_gate_passed ? '通过' : '失败'}</span>
            <strong>前瞻预测</strong><span>0/6（C276）</span>
            <strong>证据级别</strong><span>结构性共现</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>same-sign source precedence fraction</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c275Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c275Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c275Data.coordinateSemantics} {c275Data.claimBoundary || C275_CROSS_ROLE_REUSE_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c289Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C289 Joint Response Event Automaton Atlas Heatmap">
          <header>
            <strong>{C289_JOINT_RESPONSE_CAMPAIGN_ATLAS_ROUTE.title}</strong>
            <span>Phase {c289Data.phase} / {c289Data.dimensions.length} coordinates / {c289Data.totalRows} rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>一步预测</strong><span>{c289Data.summary?.one_step_candidates_passing ?? 0}/3</span>
            <strong>自主滚动</strong><span>{c289Data.summary?.rollout_candidates_passing ?? 0}/3</span>
            <strong>跨模型拓扑</strong><span>{c289Data.summary?.cross_model_pairs_passing ?? 0}/3</span>
            <strong>局部因果</strong><span>0/2（C291）</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>event accuracy / signed interaction / edit response</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c289Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c289Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c289Data.coordinateSemantics} {c289Data.claimBoundary || C289_JOINT_RESPONSE_CAMPAIGN_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c308Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C308 Conditional Hypergraph Campaign Atlas Heatmap">
          <header>
            <strong>{C308_CONDITIONAL_HYPERGRAPH_CAMPAIGN_ATLAS_ROUTE.title}</strong>
            <span>Phase {c308Data.phase} / {c308Data.dimensions.length} coordinates / {c308Data.totalRows} rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>幅值预测</strong><span>{c308Data.summary?.amplitude_families_passing ?? 0}/6</span>
            <strong>锁箱锦标赛</strong><span>{c308Data.summary?.lockbox_tournament_families_passing ?? 0}/6</span>
            <strong>场组合预测</strong><span>{c308Data.summary?.composition_families_passing ?? 0}/6</span>
            <strong>因果补丁</strong><span>{c308Data.summary?.causal_branches_passing ?? 0}/1</span>
            <strong>跨模型拓扑</strong><span>{c308Data.summary?.cross_model_pairs_passing ?? 0}/3</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>embedding / HiddenState / transition / amplitude / composition / qualification</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c308Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c308Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c308Data.coordinateSemantics} {c308Data.claimBoundary || C308_CONDITIONAL_HYPERGRAPH_CAMPAIGN_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c335Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C335 Dual-Axis Response Atlas Heatmap">
          <header>
            <strong>{C335_DUAL_AXIS_RESPONSE_ATLAS_ROUTE.title}</strong>
            <span>Phase {c335Data.phase} / {c335Data.dimensions.length} coordinates / {c335Data.totalRows} rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>残差特异性</strong><span>{c335Data.summary?.specificity_families_passing ?? 0}/6</span>
            <strong>原子运输</strong><span>{c335Data.summary?.atomic_transport_families_passing ?? 0}/6</span>
            <strong>分布式因果</strong><span>{c335Data.summary?.distributed_width_families_passing ?? 0}/6</span>
            <strong>自然组合</strong><span>{c335Data.summary?.natural_composition_families_passing ?? 0}/6</span>
            <strong>跨模型门</strong><span>{c335Data.summary?.cross_model_model_gates_passing ?? 0}/3</span>
            <strong>改名图深度</strong><span>{c335Data.summary?.renamed_graph_depth_gate ? '保留' : '未保留'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>embedding / HiddenState interaction / graph-depth increment</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c335Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c335Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c335Data.coordinateSemantics} {c335Data.claimBoundary || C335_DUAL_AXIS_RESPONSE_ATLAS_ROUTE.boundary}</p>
        </section>
      )}
      {c360Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C360 Single-Sample Operator Field Heatmap">
          <header>
            <strong>{C360_SINGLE_SAMPLE_OPERATOR_FIELD_ROUTE.title}</strong>
            <span>Phase {c360Data.phase} / {c360Data.dimensions.length} coordinates / {c360Data.totalRows} rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>单样本主门</strong><span>{c360Data.summary?.single_sample ? '通过' : '未通过'}</span>
            <strong>组合更新</strong><span>{c360Data.summary?.composition ? '保留' : '未保留'}</span>
            <strong>图递归</strong><span>{c360Data.summary?.graph ? '通过' : '未测试/未通过'}</span>
            <strong>因果中介</strong><span>{c360Data.summary?.mediation ? '通过' : '未获资格'}</span>
            <strong>功能双模拟</strong><span>{c360Data.summary?.bisimulation ? '通过' : '未建立'}</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>embedding / HiddenState / final norm × semantic role</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c360Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c360Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c360Data.coordinateSemantics} {c360Data.claimBoundary || C360_SINGLE_SAMPLE_OPERATOR_FIELD_ROUTE.boundary}</p>
        </section>
      )}
      {c390Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C390 Typed Language Operation Field Heatmap">
          <header>
            <strong>{C390_LANGUAGE_OPERATION_FIELD_ROUTE.title}</strong>
            <span>Phase {c390Data.phase} / {c390Data.dimensions.length} coordinates / {c390Data.totalRows} archived rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>Displayed slice</strong><span>16 families x embedding/q24 interaction + 8 q24 tokens</span>
            <strong>Native axis</strong><span>Qwen3-4B / 2560 coordinates</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>embedding / HiddenState interaction and token field</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c390Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c390Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c390Data.coordinateSemantics} {c390Data.claimBoundary || C390_LANGUAGE_OPERATION_FIELD_ROUTE.boundary}</p>
        </section>
      )}
      {c398Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C398 Independent Construction Lockbox Heatmap">
          <header>
            <strong>{C398_INDEPENDENT_CONSTRUCTION_LOCKBOX_ROUTE.title}</strong>
            <span>Phase {c398Data.phase} / {c398Data.dimensions.length} coordinates / {c398Data.totalRows} archived rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>Displayed slice</strong><span>3 families x embedding/q24 x 6 roles</span>
            <strong>Native axis</strong><span>Qwen3-4B / 2560 coordinates</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>fresh-construction interaction centroids</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c398Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c398Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c398Data.coordinateSemantics} {c398Data.claimBoundary || C398_INDEPENDENT_CONSTRUCTION_LOCKBOX_ROUTE.boundary}</p>
        </section>
      )}
      {c414Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C414 Output-Sensitive Language Field Heatmap">
          <header>
            <strong>{C414_OUTPUT_SENSITIVE_LANGUAGE_FIELD_ROUTE.title}</strong>
            <span>Phase {c414Data.phase} / {c414Data.dimensions.length} coordinates / {c414Data.totalRows} archived rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>Displayed slice</strong><span>16 families x q0/q24 x primary/boundary</span>
            <strong>Native axis</strong><span>Qwen3-4B / 2560 coordinates</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>output-sensitive family interactions</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c414Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c414Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c414Data.coordinateSemantics} {c414Data.claimBoundary || C414_OUTPUT_SENSITIVE_LANGUAGE_FIELD_ROUTE.boundary}</p>
        </section>
      )}
      {c433Data.available && (
        <section className="research-heatmap-card__relation" aria-label="C433 Axis Lockbox Interaction Field Heatmap">
          <header>
            <strong>{C433_AXIS_LOCKBOX_FIELD_ROUTE.title}</strong>
            <span>Phase {c433Data.phase} / {c433Data.dimensions.length} coordinates / {c433Data.totalRows} archived rows</span>
          </header>
          <div className="research-heatmap-card__summary">
            <strong>Displayed slice</strong><span>3 families x 2 axes x 6 masks x primary/boundary</span>
            <strong>Native axis</strong><span>Qwen3-4B / 2560 coordinates</span>
          </div>
          <div className="research-heatmap-card__coordinate-axis"><span>q24 Mobius interaction field</span><b>Qwen3 physical activation coordinate id</b></div>
          <div className="research-heatmap-card__relation-scroll research-heatmap-card__relation-scroll--coordinates">
            {c433Data.rows.map((row, index) => (
              <div className="research-heatmap-card__relation-row research-heatmap-card__relation-row--coordinates" key={`${row.label}-${index}`}>
                <span>{row.label}</span>
                <div style={{ '--relation-columns': c433Data.dimensions.length }}>{row.cells.map((cell) => <i key={cell.dimension} title={`${row.label} / activation coordinate ${cell.dimension}: ${Number(cell.raw).toPrecision(7)}`} style={{ background: signedColor(cell.value, true) }} />)}</div>
              </div>
            ))}
          </div>
          <p><Info size={11} /> {c433Data.coordinateSemantics} {c433Data.claimBoundary || C433_AXIS_LOCKBOX_FIELD_ROUTE.boundary}</p>
        </section>
      )}
    </section>
  );
}

function HeatmapPanel({
  title,
  rows,
  dimensions,
  position,
  rowLabel,
  axisLabel,
  size = 0.42,
  titleSize = 0.34,
  gridColumns = null,
  targetWidth = null,
  targetHeight = null,
}) {
  const rowCount = rows?.length || 1;
  const fitted = fitHeatmapGrid({
    dimensionsLength: dimensions?.length || 0,
    requestedColumns: gridColumns,
    rowCount,
    targetWidth,
    targetHeight,
  });
  const finalSize = fitted?.size || size;
  const finalColumnCount = Math.max(1, Math.min(fitted.columnCount || dimensions.length || 1, dimensions.length || 1));
  const cellsPerRow = Math.max(1, Math.ceil((dimensions.length || 1) / finalColumnCount));
  const width = Math.max(1, (targetWidth && targetWidth > 0) ? targetWidth : finalColumnCount * finalSize);
  const height = Math.max(1, (targetHeight && targetHeight > 0) ? fitted.height : rows.length * cellsPerRow * finalSize);
  const textOffset = finalSize * 1.2;
  return (
    <group position={position}>
      <Text position={[0, height / 2 + finalSize + 0.28, 0]} fontSize={titleSize} color="#e0f2fe" anchorX="center">
        {title}
      </Text>
      {rows.flatMap((row, rowIndex) =>
        row.cells.map((cellData, columnIndex) => (
          <mesh
            key={`${rowIndex}-${cellData.dimension}`}
            position={[
              (columnIndex % finalColumnCount - (finalColumnCount - 1) / 2) * finalSize,
              height / 2 - (rowIndex * cellsPerRow + Math.floor(columnIndex / finalColumnCount)) * finalSize,
              0,
            ]}
          >
            <boxGeometry args={[finalSize * 0.88, finalSize * 0.88, 0.1 + Math.abs(cellData.value) * 0.28]} />
            <meshBasicMaterial
              color={signedColor(cellData.value, cellData.observed)}
              transparent
              opacity={cellData.observed ? 0.96 : 0.28}
            />
          </mesh>
        )),
      )}
      {rows.map((row, index) => (
        <Text
          key={rowLabel(row)}
          position={[-width / 2 - textOffset, height / 2 - (index * cellsPerRow + (cellsPerRow - 1) / 2) * finalSize, 0]}
          fontSize={0.17}
          color="#94a3b8"
          anchorX="right"
        >
          {rowLabel(row)}
        </Text>
      ))}
      <Text position={[0, -height / 2 - 0.4, 0]} fontSize={0.16} color="#64748b" anchorX="center">
        {axisLabel}
      </Text>
    </group>
  );
}

export function ResearchHeatmapPreview3D({ trace, currentLayer = null, displayConfig, fullStateVectors, liveState = null, c390LanguageOperationField = null, c398IndependentConstructionLockbox = null, c414OutputSensitiveLanguageField = null, c433AxisLockboxField = null, c26801ResidualStateOperatorField = null, c32561LanguageEncodingField = null, c42641OutputConditionedCrossmodelField = null }) {
  const data = buildStateHeatmapData(trace, displayConfig, fullStateVectors);
  const c390Data = buildC390LanguageOperationData(c390LanguageOperationField, displayConfig);
  const c398Data = buildC398IndependentConstructionData(c398IndependentConstructionLockbox, displayConfig);
  const c414Data = buildC414OutputSensitiveLanguageData(c414OutputSensitiveLanguageField, displayConfig);
  const c433Data = buildC433AxisLockboxData(c433AxisLockboxField, displayConfig);
  const c26801Data = buildC26801ResidualStateOperatorData(c26801ResidualStateOperatorField, displayConfig);
  const c32561Data = buildC26801ResidualStateOperatorData(c32561LanguageEncodingField, displayConfig, C32561_LANGUAGE_ENCODING_FIELD_ROUTE);
  const c42641Data = buildC42641CrossmodelFieldData(c42641OutputConditionedCrossmodelField, displayConfig);
  if (!data.available && liveState) {
    const stageLabels = {
      queued: '等待 GPU',
      loading_model: '正在从本地硬盘加载模型',
      preparing_forward: '模型已加载，正在准备推理',
      forward: '正在捕获 Embedding',
      complete: '本次推理没有捕获到可显示状态',
    };
    return (
      <Text position={[0, 5, 0]} fontSize={0.42} color="#67e8f9" anchorX="center">
        {stageLabels[liveState.stage] || '正在连接本地模型'}
      </Text>
    );
  }
  if (!liveState && currentLayer == null && c42641Data.available) {
    const densePanelLayout = c42641Data.panels.length > 5;
    const positions = densePanelLayout
      ? c42641Data.panels.map((_, index) => [-5.25 + (index % 4) * 3.5, 2.1 - Math.floor(index / 4) * 4.2, 0])
      : [[-3.5, 2.1, 0], [0, 2.1, 0], [3.5, 2.1, 0], [-1.8, -2.1, 0], [1.8, -2.1, 0]];
    return (
      <group>
        {c42641Data.panels.map((panel, index) => (
          <HeatmapPanel
            key={panel.key}
            title={`${panel.model} / ${panel.coordinateCount} model-local coordinates`}
            rows={panel.rows}
            dimensions={panel.dimensions}
            position={positions[index] || [0, 0, 0]}
            rowLabel={(row) => row.label}
            gridColumns={displayConfig?.mode === 'all' ? 64 : null}
            axisLabel={displayConfig?.mode === 'all' ? `all ${panel.coordinateCount} coordinates / ${panel.coordinateOrderLabel}` : `selected coordinates / ${panel.coordinateOrderLabel}`}
            targetWidth={c42641Data.panels.length > 4 ? 3.2 : 4.8}
            targetHeight={3.4}
            titleSize={0.22}
          />
        ))}
      </group>
    );
  }
  if (!liveState && currentLayer == null && c32561Data.available) {
    const previewRows = c32561Data.rows.filter((row) => row.preview !== false);
    return (
      <group>
        <HeatmapPanel
          title="C32561-C39440 embedding, HiddenState, semantic passports, output-conditioned VJP, finite effects, and cross-language evidence"
          rows={previewRows}
          dimensions={c32561Data.dimensions}
          position={[0, 0, 0]}
          rowLabel={(row) => row.label}
          gridColumns={displayConfig?.mode === 'all' ? 64 : null}
          axisLabel={displayConfig?.mode === 'all' ? 'all 2560 Qwen3-4B physical coordinates / parameters' : 'selected physical coordinates / parameters'}
          targetWidth={10}
          targetHeight={7}
        />
      </group>
    );
  }
  if (!liveState && currentLayer == null && c26801Data.available) {
    const previewRows = c26801Data.rows.filter((row) => (
      (row.source === 'sample_state' && (
        row.coordinate_kind === 'embedding'
        || (row.coordinate_kind === 'hidden_state' && Number(row.layer) === 24 && row.event === 'answer_boundary')
      ))
      || (row.source === 'family_passport' && Number(row.layer) === 23 && row.event === 'answer_boundary' && (
        row.component === 'total'
        || (['comparison', 'taxonomy'].includes(row.family) && ['attention', 'mlp'].includes(row.component))
      ))
      || (['state_slope', 'physical_gain'].includes(row.source) && Number(row.layer) === 23 && row.event === 'answer_boundary')
    ));
    return (
      <group>
        <HeatmapPanel
          title="C26801 embedding, HiddenState, component passports, and fixed-coordinate operator"
          rows={previewRows}
          dimensions={c26801Data.dimensions}
          position={[0, 0, 0]}
          rowLabel={(row) => row.label}
          gridColumns={displayConfig?.mode === 'all' ? 64 : null}
          axisLabel={displayConfig?.mode === 'all' ? 'all 2560 Qwen3-4B physical coordinates' : 'selected physical coordinates'}
          targetWidth={10}
          targetHeight={7}
        />
      </group>
    );
  }
  if (!liveState && currentLayer == null && c433Data.available) {
    const previewRows = c433Data.rows.filter((row) => row.query_axis === 'attitude' && row.role === 'boundary');
    return (
      <group>
        <HeatmapPanel
          title="C433 q24 attitude-axis high-order interaction field"
          rows={previewRows}
          dimensions={c433Data.dimensions}
          position={[0, 0, 0]}
          rowLabel={(row) => row.label}
          gridColumns={displayConfig?.mode === 'all' ? 64 : null}
          axisLabel={displayConfig?.mode === 'all' ? 'all 2560 activation coordinates' : 'selected activation coordinates'}
          targetWidth={10}
          targetHeight={7}
        />
      </group>
    );
  }
  if (!liveState && currentLayer == null && c414Data.available) {
    const previewRows = c414Data.rows.filter((row) => (
      ['comparison', 'temporal_order'].includes(row.family)
      && [0, 24].includes(Number(row.checkpoint))
    ));
    return (
      <group>
        <HeatmapPanel
          title="C414 sparse cross-construction candidates: comparison and temporal order"
          rows={previewRows}
          dimensions={c414Data.dimensions}
          position={[0, 0, 0]}
          rowLabel={(row) => row.label}
          gridColumns={displayConfig?.mode === 'all' ? 64 : null}
          axisLabel={displayConfig?.mode === 'all' ? 'all 2560 activation coordinates' : 'selected activation coordinates'}
          targetWidth={10}
          targetHeight={7}
        />
      </group>
    );
  }
  if (!liveState && currentLayer == null && c398Data.available) {
    const previewRows = c398Data.rows.filter((row) => (
      (Number(row.checkpoint) === 0 && row.role === 'primary')
      || (Number(row.checkpoint) === 24 && row.role === 'boundary')
    ));
    return (
      <group>
        <HeatmapPanel
          title="C398 fresh-construction response: q0 primary vs q24 boundary"
          rows={previewRows}
          dimensions={c398Data.dimensions}
          position={[0, 0, 0]}
          rowLabel={(row) => row.label}
          gridColumns={displayConfig?.mode === 'all' ? 64 : null}
          axisLabel={displayConfig?.mode === 'all' ? 'all 2560 activation coordinates' : 'selected activation coordinates'}
          targetWidth={10}
          targetHeight={7}
        />
      </group>
    );
  }
  if (!liveState && currentLayer == null && c390Data.available) {
    const previewRows = c390Data.rows.filter((row) => row.family === 'attitude_event' && Number(row.checkpoint) === 0);
    return (
      <group>
        <HeatmapPanel
          title="C390 attitude-event embedding interaction"
          rows={previewRows}
          dimensions={c390Data.dimensions}
          position={[0, 0, 0]}
          rowLabel={(row) => row.label}
          gridColumns={displayConfig?.mode === 'all' ? 64 : null}
          axisLabel={displayConfig?.mode === 'all' ? 'all 2560 activation coordinates' : 'selected activation coordinates'}
          targetWidth={10}
          targetHeight={7}
        />
      </group>
    );
  }
  if (!data.available) {
    return (
      <Text position={[0, 5, 0]} fontSize={0.42} color="#fbbf24" anchorX="center">
        热力图需要同一次 Run 的词嵌入与 HiddenState top-k 原始记录
      </Text>
    );
  }
  const activeHiddenState = currentLayer == null
    ? null
    : data.hidden.find((row) => Number(row.layer) === Number(currentLayer)) || null;
  const layerAnchorX = LAYER_HEATMAP_LAYOUT.layerAnchorX;
  const llmModelAnchorX = LAYER_HEATMAP_LAYOUT.llmModelAnchorX;
  const modelToLayerDistance = layerAnchorX - llmModelAnchorX;
  const modelToLayerAbsDistance = Math.abs(modelToLayerDistance);
  const modelToLayerDirection = Math.sign(modelToLayerDistance) === 0 ? 1 : Math.sign(modelToLayerDistance);
  const layerScaledWidth = LAYER_HEATMAP_LAYOUT.layerBaseWidth * LAYER_HEATMAP_LAYOUT.layerScale;
  const layerScaledHeight = LAYER_HEATMAP_LAYOUT.layerBaseHeight * LAYER_HEATMAP_LAYOUT.layerScale;
  const heatmapWidth = layerScaledWidth * LAYER_HEATMAP_LAYOUT.fullStateScale;
  const heatmapHeight = layerScaledHeight * LAYER_HEATMAP_LAYOUT.fullStateScale;
  const fullStateColumns = data.usingFullVectors && displayConfig?.mode === 'all' ? 64 : null;
  const sideOffset = layerScaledWidth / 2 + heatmapWidth / 2 + LAYER_HEATMAP_LAYOUT.sideGap;
  const embeddingPositionX = llmModelAnchorX - modelToLayerDirection * modelToLayerAbsDistance;
  const hiddenPositionX = layerAnchorX + sideOffset;

  return (
    <group>
      <HeatmapPanel
        title="词嵌入（Embedding）"
        rows={[{ cells: data.embedding }]}
        dimensions={data.embeddingDimensions}
        position={[embeddingPositionX, 0, 0]}
        rowLabel={() => 'embedding'}
        gridColumns={fullStateColumns}
        axisLabel={data.displayLabel}
        targetWidth={heatmapWidth}
        targetHeight={heatmapHeight}
      />
      {activeHiddenState ? (
        <HeatmapPanel
          title={`L${activeHiddenState.layer} HiddenState`}
          rows={[activeHiddenState]}
          dimensions={activeHiddenState.dimensions}
          position={[hiddenPositionX, 0, 0]}
          rowLabel={() => `L${activeHiddenState.layer}`}
          gridColumns={fullStateColumns}
          axisLabel={`当前层：${data.displayLabel}`}
          titleSize={0.27}
          targetWidth={heatmapWidth}
          targetHeight={heatmapHeight}
        />
      ) : (
        <Text position={[hiddenPositionX, heatmapHeight / 2 + 0.55, 0]} fontSize={0.26} color="#fbbf24" anchorX="center">
          运行到某个 Layer 后显示该层 HiddenState
        </Text>
      )}
    </group>
  );
}
