/**
 * 逆向工程3D叠加效果容器
 * 在主Canvas中条件渲染，叠加在AppleNeuronSceneContent之上
 */
import { useMemo } from 'react';
import OrthogonalSubspaceOverlay from './OrthogonalSubspaceOverlay';
import BandFrequencyOverlay from './BandFrequencyOverlay';
import CausalFlowOverlay from './CausalFlowOverlay';
import EncodingEquationOverlay from './EncodingEquationOverlay';

export default function ReverseEngineeringOverlay({
  viewMode,
  selectedLanguageDims,
  selectedDNNFeature,
  selectedDNNCategory,
  nodes,
  links,
}) {
  // Compute layer positions from nodes for overlay placement
  const layerPositions = useMemo(() => {
    if (!nodes || nodes.length === 0) return [];
    const layers = {};
    nodes.forEach((node) => {
      if (node.layer !== undefined && node.position) {
        if (!layers[node.layer]) {
          layers[node.layer] = { x: 0, y: 0, z: 0, count: 0 };
        }
        layers[node.layer].x += node.position[0] || node.position.x || 0;
        layers[node.layer].y += node.position[1] || node.position.y || 0;
        layers[node.layer].z += node.position[2] || node.position.z || 0;
        layers[node.layer].count++;
      }
    });
    return Object.entries(layers).map(([layer, pos]) => ({
      layer: parseInt(layer),
      x: pos.x / pos.count,
      y: pos.y / pos.count,
      z: pos.z / pos.count,
    }));
  }, [nodes]);

  // Get active dimension list
  const activeDimensions = useMemo(() => {
    const dims = [];
    Object.entries(selectedLanguageDims || {}).forEach(([dimId, subs]) => {
      if (Object.values(subs).some(Boolean)) {
        dims.push(dimId);
      }
    });
    return dims;
  }, [selectedLanguageDims]);

  if (!viewMode || viewMode === 'structure') return null;

  return (
    <group>
      {viewMode === 'orthogonal' && (
        <OrthogonalSubspaceOverlay
          activeDimensions={activeDimensions}
          selectedDNNFeature={selectedDNNFeature}
          layerPositions={layerPositions}
        />
      )}
      {viewMode === 'spectral' && (
        <BandFrequencyOverlay
          activeDimensions={activeDimensions}
          selectedDNNFeature={selectedDNNFeature}
          layerPositions={layerPositions}
          nodes={nodes}
        />
      )}
      {viewMode === 'causal' && (
        <CausalFlowOverlay
          activeDimensions={activeDimensions}
          selectedDNNFeature={selectedDNNFeature}
          layerPositions={layerPositions}
          links={links}
        />
      )}
      {viewMode === 'encoding' && (
        <EncodingEquationOverlay
          activeDimensions={activeDimensions}
          selectedDNNFeature={selectedDNNFeature}
          layerPositions={layerPositions}
        />
      )}
    </group>
  );
}
