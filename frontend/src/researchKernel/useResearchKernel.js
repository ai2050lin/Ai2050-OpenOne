import { useEffect, useMemo, useState } from 'react';

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
} from './heatmapResearchRoute';

import { researchAssetUrl } from '../config/researchAssets';

const KERNEL_BASE = researchAssetUrl('research_kernel');
const TRACE_MANIFEST = researchAssetUrl('real_component_trace/manifest.json');

const MODEL_KEY_MAP = {
  'qwen3-4b': 'qwen3',
  'glm4-9b': 'glm4',
  ds7b: 'deepseek7b',
};

const EVENT_BY_SUBPHASE = {
  input: ['residual_input'],
  ln1: ['norm1'],
  qkv: ['q_projection', 'qkv_projection'],
  // Phase287 does not expose pre-softmax scores or attention probabilities.
  // Keep these empty instead of presenting K/V projections as direct evidence.
  attn_score: [],
  softmax: [],
  attn_out: ['attention_output'],
  residual1: ['residual1'],
  ln2: ['norm2'],
  ffn_up: ['mlp_up', 'mlp_gate_up_merged', 'mlp_gate'],
  ffn_act: ['mlp_product'],
  ffn_down: ['mlp_down'],
  residual2: ['residual2'],
};

async function fetchJson(path) {
  const response = await fetch(researchAssetUrl(path), { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.json();
}

async function fetchJsonl(path) {
  const response = await fetch(researchAssetUrl(path), { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  const text = await response.text();
  return text
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function eventFor(events, layer, subphaseId) {
  const accepted = EVENT_BY_SUBPHASE[subphaseId] || [];
  return events.find((event) => Number(event.layer) === Number(layer) && accepted.includes(event.event_type)) || null;
}

function buildForwardData(trace) {
  if (!trace?.events?.length) return null;
  const layerCount = Number(trace.model_snapshot?.num_hidden_layers || 0);
  const byLayer = new Map();
  trace.events.forEach((event) => {
    const layer = Number(event.layer);
    if (layer < 0) return;
    if (!byLayer.has(layer)) byLayer.set(layer, {});
    byLayer.get(layer)[event.event_type] = event;
  });
  return {
    schema_version: trace.schema_version,
    model: trace.model,
    model_info: trace.model_snapshot,
    sentence: trace.prompt,
    tokens: trace.tokens || [],
    evidence_level: 'L2',
    source_run_id: trace.run_id,
    layers: Array.from({ length: layerCount }, (_, layer) => {
      const events = byLayer.get(layer) || {};
      const attention = events.attention_output || {};
      const gate = events.mlp_gate || events.mlp_gate_up_merged || {};
      const product = events.mlp_product || {};
      const residual = events.residual2 || {};
      return {
        layer,
        label: `L${layer} real trace`,
        attention: { norm: attention.norm ?? null },
        ffn: {
          gate_activation: gate.norm ?? null,
          norm: product.norm ?? null,
          top_neurons: product.top_units || [],
        },
        residual_norm: residual.norm ?? null,
        candidate_field: residual.candidate_field || null,
      };
    }),
  };
}

function latestRunForModel(runs, model) {
  const candidates = (runs || []).filter((run) => run?.model === model);
  if (!candidates.length) return null;

  return candidates.reduce((best, current) => {
    const bestPhase = Number(best?.phase ?? 0);
    const currentPhase = Number(current?.phase ?? 0);
    if (Number.isNaN(currentPhase) || Number.isNaN(bestPhase)) {
      return best ?? current;
    }
    return currentPhase >= bestPhase ? current : best;
  });
}

export function useResearchKernel(fpModel, fpCurrentLayer, subphaseId) {
  const model = MODEL_KEY_MAP[fpModel] || 'qwen3';
  const [kernelManifest, setKernelManifest] = useState(null);
  const [traceManifest, setTraceManifest] = useState(null);
  const [traceState, setTraceState] = useState({ model: '', payload: null });
  const [fullStateVectorState, setFullStateVectorState] = useState({ model: '', payload: null });
  const [unitState, setUnitState] = useState({ model: '', rows: [] });
  const [relationContrastHeatmap, setRelationContrastHeatmap] = useState(null);
  const [graphWalshHeatmap, setGraphWalshHeatmap] = useState(null);
  const [c101ActivationHeatmap, setC101ActivationHeatmap] = useState(null);
  const [c102CoordinateBarcodeHeatmap, setC102CoordinateBarcodeHeatmap] = useState(null);
  const [c104UpstreamRoleBarcodeHeatmap, setC104UpstreamRoleBarcodeHeatmap] = useState(null);
  const [c109RoleStateFieldAtlas, setC109RoleStateFieldAtlas] = useState(null);
  const [c157C166LocalFieldHeatmap, setC157C166LocalFieldHeatmap] = useState(null);
  const [c167C168RelationResidualHeatmap, setC167C168RelationResidualHeatmap] = useState(null);
  const [c170RoleCheckpointHeatmap, setC170RoleCheckpointHeatmap] = useState(null);
  const [c183NaturalResponseEcologyHeatmap, setC183NaturalResponseEcologyHeatmap] = useState(null);
  const [c189NewMaterialResponseScaffoldHeatmap, setC189NewMaterialResponseScaffoldHeatmap] = useState(null);
  const [c191ResponseEquivalenceAtlas, setC191ResponseEquivalenceAtlas] = useState(null);
  const [c193ProgramCenteredResidualHeatmap, setC193ProgramCenteredResidualHeatmap] = useState(null);
  const [c202SignedOperatorCampaignHeatmap, setC202SignedOperatorCampaignHeatmap] = useState(null);
  const [c215ResponseIntervalCompositionAtlas, setC215ResponseIntervalCompositionAtlas] = useState(null);
  const [c220ResponseStateMinimalityAtlas, setC220ResponseStateMinimalityAtlas] = useState(null);
  const [c222SurfaceConditionedResponseAtlas, setC222SurfaceConditionedResponseAtlas] = useState(null);
  const [c233SurfaceTransportCompositionAtlas, setC233SurfaceTransportCompositionAtlas] = useState(null);
  const [c243ConditionalEventAtlas, setC243ConditionalEventAtlas] = useState(null);
  const [c244IndependentEventReplication, setC244IndependentEventReplication] = useState(null);
  const [c245ConfirmedEventCore, setC245ConfirmedEventCore] = useState(null);
  const [c254TriMaterialEventAtlas, setC254TriMaterialEventAtlas] = useState(null);
  const [c260OutputPathCausalAtlas, setC260OutputPathCausalAtlas] = useState(null);
  const [c262GenerationSpecificityAtlas, setC262GenerationSpecificityAtlas] = useState(null);
  const [c272StateConditionedOperatorAtlas, setC272StateConditionedOperatorAtlas] = useState(null);
  const [c273ResponseEcologyAtlas, setC273ResponseEcologyAtlas] = useState(null);
  const [c275CrossRoleReuseAtlas, setC275CrossRoleReuseAtlas] = useState(null);
  const [c289JointResponseCampaignAtlas, setC289JointResponseCampaignAtlas] = useState(null);
  const [c308ConditionalHypergraphCampaignAtlas, setC308ConditionalHypergraphCampaignAtlas] = useState(null);
  const [c335DualAxisResponseAtlas, setC335DualAxisResponseAtlas] = useState(null);
  const [c360SingleSampleOperatorField, setC360SingleSampleOperatorField] = useState(null);
  const [c390LanguageOperationField, setC390LanguageOperationField] = useState(null);
  const [c398IndependentConstructionLockbox, setC398IndependentConstructionLockbox] = useState(null);
  const [c414OutputSensitiveLanguageField, setC414OutputSensitiveLanguageField] = useState(null);
  const [c433AxisLockboxField, setC433AxisLockboxField] = useState(null);
  const [c26801ResidualStateOperatorField, setC26801ResidualStateOperatorField] = useState(null);
  const [c32561LanguageEncodingField, setC32561LanguageEncodingField] = useState(null);
  const [c42641OutputConditionedCrossmodelField, setC42641OutputConditionedCrossmodelField] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    Promise.all([fetchJson(`${KERNEL_BASE}/manifest.json`), fetchJson(TRACE_MANIFEST)])
      .then(([kernel, traces]) => {
        if (!active) return;
        setKernelManifest(kernel);
        setTraceManifest(traces);
        setError('');
      })
      .catch((err) => {
        if (!active) return;
        setError(err?.message || 'research kernel unavailable');
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C262_GENERATION_SPECIFICITY_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C262_GENERATION_SPECIFICITY_ATLAS_ROUTE.sourceSchema) {
          setC262GenerationSpecificityAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC262GenerationSpecificityAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C272_STATE_CONDITIONED_OPERATOR_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C272_STATE_CONDITIONED_OPERATOR_ATLAS_ROUTE.sourceSchema) {
          setC272StateConditionedOperatorAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC272StateConditionedOperatorAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([
      fetchJson(C273_RESPONSE_ECOLOGY_ATLAS_ROUTE.sourcePath),
      fetchJson(C275_CROSS_ROLE_REUSE_ATLAS_ROUTE.sourcePath),
      fetchJson(C289_JOINT_RESPONSE_CAMPAIGN_ATLAS_ROUTE.sourcePath),
      fetchJson(C308_CONDITIONAL_HYPERGRAPH_CAMPAIGN_ATLAS_ROUTE.sourcePath),
      fetchJson(C335_DUAL_AXIS_RESPONSE_ATLAS_ROUTE.sourcePath),
      fetchJson(C360_SINGLE_SAMPLE_OPERATOR_FIELD_ROUTE.sourcePath),
      fetchJson(C390_LANGUAGE_OPERATION_FIELD_ROUTE.sourcePath),
      fetchJson(C398_INDEPENDENT_CONSTRUCTION_LOCKBOX_ROUTE.sourcePath),
      fetchJson(C414_OUTPUT_SENSITIVE_LANGUAGE_FIELD_ROUTE.sourcePath),
      fetchJson(C433_AXIS_LOCKBOX_FIELD_ROUTE.sourcePath),
      fetchJson(C26801_RESIDUAL_STATE_OPERATOR_FIELD_ROUTE.sourcePath),
      fetchJson(C32561_LANGUAGE_ENCODING_FIELD_ROUTE.sourcePath),
      fetchJson(C42641_OUTPUT_CONDITIONED_CROSSMODEL_FIELD_ROUTE.sourcePath),
    ])
      .then(([c273, c275, c289, c308, c335, c360, c390, c398, c414, c433, c26801, c32561, c42641]) => {
        if (!active) return;
        setC273ResponseEcologyAtlas(c273?.schema === C273_RESPONSE_ECOLOGY_ATLAS_ROUTE.sourceSchema ? c273 : null);
        setC275CrossRoleReuseAtlas(c275?.schema === C275_CROSS_ROLE_REUSE_ATLAS_ROUTE.sourceSchema ? c275 : null);
        setC289JointResponseCampaignAtlas(c289?.schema === C289_JOINT_RESPONSE_CAMPAIGN_ATLAS_ROUTE.sourceSchema ? c289 : null);
        setC308ConditionalHypergraphCampaignAtlas(c308?.schema === C308_CONDITIONAL_HYPERGRAPH_CAMPAIGN_ATLAS_ROUTE.sourceSchema ? c308 : null);
        setC335DualAxisResponseAtlas(c335?.schema === C335_DUAL_AXIS_RESPONSE_ATLAS_ROUTE.sourceSchema ? c335 : null);
        setC360SingleSampleOperatorField(c360?.schema === C360_SINGLE_SAMPLE_OPERATOR_FIELD_ROUTE.sourceSchema ? c360 : null);
        setC390LanguageOperationField(c390?.schema === C390_LANGUAGE_OPERATION_FIELD_ROUTE.sourceSchema ? c390 : null);
        setC398IndependentConstructionLockbox(c398?.schema === C398_INDEPENDENT_CONSTRUCTION_LOCKBOX_ROUTE.sourceSchema ? c398 : null);
        setC414OutputSensitiveLanguageField(c414?.schema === C414_OUTPUT_SENSITIVE_LANGUAGE_FIELD_ROUTE.sourceSchema ? c414 : null);
        setC433AxisLockboxField(c433?.schema === C433_AXIS_LOCKBOX_FIELD_ROUTE.sourceSchema ? c433 : null);
        setC26801ResidualStateOperatorField(c26801?.schema === C26801_RESIDUAL_STATE_OPERATOR_FIELD_ROUTE.sourceSchema ? c26801 : null);
        setC32561LanguageEncodingField(c32561?.schema === C32561_LANGUAGE_ENCODING_FIELD_ROUTE.sourceSchema ? c32561 : null);
        setC42641OutputConditionedCrossmodelField(c42641?.schema === C42641_OUTPUT_CONDITIONED_CROSSMODEL_FIELD_ROUTE.sourceSchema ? c42641 : null);
      })
      .catch(() => {
        if (!active) return;
        setC273ResponseEcologyAtlas(null);
        setC275CrossRoleReuseAtlas(null);
        setC289JointResponseCampaignAtlas(null);
        setC308ConditionalHypergraphCampaignAtlas(null);
        setC335DualAxisResponseAtlas(null);
        setC360SingleSampleOperatorField(null);
        setC390LanguageOperationField(null);
        setC398IndependentConstructionLockbox(null);
        setC414OutputSensitiveLanguageField(null);
        setC433AxisLockboxField(null);
        setC26801ResidualStateOperatorField(null);
        setC32561LanguageEncodingField(null);
        setC42641OutputConditionedCrossmodelField(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C157_C166_LOCAL_FIELD_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C157_C166_LOCAL_FIELD_HEATMAP_ROUTE.sourceSchema) {
          setC157C166LocalFieldHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC157C166LocalFieldHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C167_C168_RELATION_RESIDUAL_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C167_C168_RELATION_RESIDUAL_HEATMAP_ROUTE.sourceSchema) {
          setC167C168RelationResidualHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC167C168RelationResidualHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C183_NATURAL_RESPONSE_ECOLOGY_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C183_NATURAL_RESPONSE_ECOLOGY_HEATMAP_ROUTE.sourceSchema) {
          setC183NaturalResponseEcologyHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC183NaturalResponseEcologyHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C189_NEW_MATERIAL_RESPONSE_SCAFFOLD_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C189_NEW_MATERIAL_RESPONSE_SCAFFOLD_HEATMAP_ROUTE.sourceSchema) {
          setC189NewMaterialResponseScaffoldHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC189NewMaterialResponseScaffoldHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C191_RESPONSE_EQUIVALENCE_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C191_RESPONSE_EQUIVALENCE_ATLAS_ROUTE.sourceSchema) {
          setC191ResponseEquivalenceAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC191ResponseEquivalenceAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C193_PROGRAM_CENTERED_RESIDUAL_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C193_PROGRAM_CENTERED_RESIDUAL_HEATMAP_ROUTE.sourceSchema) {
          setC193ProgramCenteredResidualHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC193ProgramCenteredResidualHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C202_SIGNED_OPERATOR_CAMPAIGN_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C202_SIGNED_OPERATOR_CAMPAIGN_HEATMAP_ROUTE.sourceSchema) {
          setC202SignedOperatorCampaignHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC202SignedOperatorCampaignHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C215_RESPONSE_INTERVAL_COMPOSITION_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C215_RESPONSE_INTERVAL_COMPOSITION_ATLAS_ROUTE.sourceSchema) {
          setC215ResponseIntervalCompositionAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC215ResponseIntervalCompositionAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C220_RESPONSE_STATE_MINIMALITY_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C220_RESPONSE_STATE_MINIMALITY_ATLAS_ROUTE.sourceSchema) {
          setC220ResponseStateMinimalityAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC220ResponseStateMinimalityAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C222_SURFACE_CONDITIONED_RESPONSE_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C222_SURFACE_CONDITIONED_RESPONSE_ATLAS_ROUTE.sourceSchema) {
          setC222SurfaceConditionedResponseAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC222SurfaceConditionedResponseAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C233_SURFACE_TRANSPORT_COMPOSITION_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C233_SURFACE_TRANSPORT_COMPOSITION_ATLAS_ROUTE.sourceSchema) {
          setC233SurfaceTransportCompositionAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC233SurfaceTransportCompositionAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C243_CONDITIONAL_EVENT_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C243_CONDITIONAL_EVENT_ATLAS_ROUTE.sourceSchema) {
          setC243ConditionalEventAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC243ConditionalEventAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C244_INDEPENDENT_EVENT_REPLICATION_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C244_INDEPENDENT_EVENT_REPLICATION_ROUTE.sourceSchema) {
          setC244IndependentEventReplication(payload);
        }
      })
      .catch(() => {
        if (active) setC244IndependentEventReplication(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C245_CONFIRMED_EVENT_CORE_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C245_CONFIRMED_EVENT_CORE_ROUTE.sourceSchema) {
          setC245ConfirmedEventCore(payload);
        }
      })
      .catch(() => {
        if (active) setC245ConfirmedEventCore(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C254_TRI_MATERIAL_EVENT_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C254_TRI_MATERIAL_EVENT_ATLAS_ROUTE.sourceSchema) {
          setC254TriMaterialEventAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC254TriMaterialEventAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C260_OUTPUT_PATH_CAUSAL_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C260_OUTPUT_PATH_CAUSAL_ATLAS_ROUTE.sourceSchema) {
          setC260OutputPathCausalAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC260OutputPathCausalAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C170_ROLE_CHECKPOINT_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C170_ROLE_CHECKPOINT_HEATMAP_ROUTE.sourceSchema) {
          setC170RoleCheckpointHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC170RoleCheckpointHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C109_ROLE_STATE_FIELD_ATLAS_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C109_ROLE_STATE_FIELD_ATLAS_ROUTE.sourceSchema) {
          setC109RoleStateFieldAtlas(payload);
        }
      })
      .catch(() => {
        if (active) setC109RoleStateFieldAtlas(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C102_COORDINATE_BARCODE_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C102_COORDINATE_BARCODE_HEATMAP_ROUTE.sourceSchema) {
          setC102CoordinateBarcodeHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC102CoordinateBarcodeHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C104_UPSTREAM_ROLE_BARCODE_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C104_UPSTREAM_ROLE_BARCODE_HEATMAP_ROUTE.sourceSchema) {
          setC104UpstreamRoleBarcodeHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC104UpstreamRoleBarcodeHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(C101_ACTIVATION_COORDINATE_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === C101_ACTIVATION_COORDINATE_HEATMAP_ROUTE.sourceSchema) {
          setC101ActivationHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setC101ActivationHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(GRAPH_WALSH_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === GRAPH_WALSH_HEATMAP_ROUTE.sourceSchema) {
          setGraphWalshHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setGraphWalshHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchJson(RELATION_CONTRAST_HEATMAP_ROUTE.sourcePath)
      .then((payload) => {
        if (active && payload?.schema === RELATION_CONTRAST_HEATMAP_ROUTE.sourceSchema) {
          setRelationContrastHeatmap(payload);
        }
      })
      .catch(() => {
        if (active) setRelationContrastHeatmap(null);
      });
    return () => { active = false; };
  }, []);

  const traceItem = useMemo(
    () => (traceManifest?.items || []).find((item) => item.model === model) || null,
    [model, traceManifest]
  );
  const stableRun = useMemo(
    () => latestRunForModel(kernelManifest?.runs || [], model),
    [kernelManifest, model]
  );

  useEffect(() => {
    let active = true;
    if (!traceItem?.path) return () => { active = false; };
    const tracePath = traceItem.path.startsWith('/')
      ? traceItem.path
      : `${KERNEL_BASE}/${traceItem.path}`;
    fetchJson(tracePath)
      .then((payload) => { if (active) setTraceState({ model, payload }); })
      .catch((err) => { if (active) setError(err?.message || 'trace load failed'); });
    return () => { active = false; };
  }, [model, traceItem]);

  useEffect(() => {
    let active = true;
    const runId = traceState.model === model ? traceState.payload?.run_id : '';
    if (!runId) return () => { active = false; };
    fetchJson(`${KERNEL_BASE}/runs/${runId}/full_state_vectors.json`)
      .then((payload) => {
        if (active) setFullStateVectorState({ model, payload });
      })
      .catch(() => {
        if (active) setFullStateVectorState({ model, payload: null });
      });
    return () => { active = false; };
  }, [model, traceState]);

  useEffect(() => {
    let active = true;
    if (!stableRun?.unit_path) return () => { active = false; };
    fetchJsonl(`${KERNEL_BASE}/${stableRun.unit_path}`)
      .then((rows) => { if (active) setUnitState({ model, rows }); })
      .catch((err) => { if (active) setError(err?.message || 'unit evidence load failed'); });
    return () => { active = false; };
  }, [model, stableRun]);

  const trace = traceState.model === model ? traceState.payload : null;
  const fullStateVectors = fullStateVectorState.model === model ? fullStateVectorState.payload : null;
  const stableUnits = unitState.model === model ? unitState.rows : [];

  const currentEvent = useMemo(
    () => eventFor(trace?.events || [], fpCurrentLayer, subphaseId),
    [fpCurrentLayer, subphaseId, trace]
  );
  const forwardData = useMemo(() => buildForwardData(trace), [trace]);

  return {
    model,
    kernelManifest,
    traceManifest,
    traceItem,
    stableRun,
    trace,
    fullStateVectors,
    relationContrastHeatmap,
    graphWalshHeatmap,
    c101ActivationHeatmap,
    c102CoordinateBarcodeHeatmap,
    c104UpstreamRoleBarcodeHeatmap,
    c109RoleStateFieldAtlas,
    c157C166LocalFieldHeatmap,
    c167C168RelationResidualHeatmap,
    c170RoleCheckpointHeatmap,
    c183NaturalResponseEcologyHeatmap,
    c189NewMaterialResponseScaffoldHeatmap,
    c191ResponseEquivalenceAtlas,
    c193ProgramCenteredResidualHeatmap,
    c202SignedOperatorCampaignHeatmap,
    c215ResponseIntervalCompositionAtlas,
    c220ResponseStateMinimalityAtlas,
    c222SurfaceConditionedResponseAtlas,
    c233SurfaceTransportCompositionAtlas,
    c243ConditionalEventAtlas,
    c244IndependentEventReplication,
    c245ConfirmedEventCore,
    c254TriMaterialEventAtlas,
    c260OutputPathCausalAtlas,
    c262GenerationSpecificityAtlas,
    c272StateConditionedOperatorAtlas,
    c273ResponseEcologyAtlas,
    c275CrossRoleReuseAtlas,
    c289JointResponseCampaignAtlas,
    c308ConditionalHypergraphCampaignAtlas,
    c335DualAxisResponseAtlas,
    c360SingleSampleOperatorField,
    c390LanguageOperationField,
    c398IndependentConstructionLockbox,
    c414OutputSensitiveLanguageField,
    c433AxisLockboxField,
    c26801ResidualStateOperatorField,
    c32561LanguageEncodingField,
    c42641OutputConditionedCrossmodelField,
    stableUnits,
    currentEvent,
    forwardData,
    error,
    ready: Boolean(trace && forwardData),
  };
}

export { buildForwardData, eventFor, MODEL_KEY_MAP };
