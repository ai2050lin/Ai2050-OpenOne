import { Html } from '@react-three/drei';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';

// --- Geometry Components ---

const Neuron = ({ pos, isFired, layer, id }) => {
  // Color code by layer
  const color = useMemo(() => {
    if (layer.includes("Shape")) return "#4ecdc4"; // Cyan
    if (layer.includes("Color")) return "#ff6b6b"; // Red
    return "#ffe66d"; // Yellow (Fiber)
  }, [layer]);

  // Glow effect when fired
  const scale = isFired ? 1.5 : 1.0;
  const emissive = isFired ? color : "#000000";
  const emissiveIntensity = isFired ? 2.0 : 0.0;

  return (
    <group position={pos}>
      <mesh scale={[scale, scale, scale]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshStandardMaterial 
            color={color} 
            emissive={emissive}
            emissiveIntensity={emissiveIntensity}
            roughness={0.3}
            metalness={0.8}
        />
      </mesh>
      {/* Label for key neurons */}
      {id.includes("12") && (
         <Html distanceFactor={10}>
            <div style={{ color: 'white', fontSize: '8px', background: 'rgba(0,0,0,0.5)', padding: '2px' }}>
              {layer.includes("Shape") ? "Round" : (layer.includes("Color") ? "Red" : "Concept")}
            </div>
         </Html>
      )}
    </group>
  );
};

const Axon = ({ start, end, isActive }) => {
    const ref = useRef()
    
    // Create geometry
    const points = useMemo(() => [
        new THREE.Vector3(...start),
        new THREE.Vector3(...end)
    ], [start, end])
    
    const lineGeometry = useMemo(() => {
        const geo = new THREE.BufferGeometry().setFromPoints(points)
        return geo
    }, [points])

    return (
        <line geometry={lineGeometry}>
            <lineBasicMaterial 
                color={isActive ? "#ffffff" : "#444444"} 
                transparent 
                opacity={isActive ? 0.8 : 0.2} 
                linewidth={1} 
            />
        </line>
    );
};

// --- Main Visualization Component ---

// --- Main Visualization Component ---

const BrainVis3D = ({ t, structure, activeSpikes }) => {
    // If no structure, render placeholder or nothing
    if (!structure) {
        return (
            <Html center>
                <div style={{ color: '#aaa', fontSize: '12px' }}>
                    {t ? t('snn.init_hint', 'Please initialize SNN') : 'Please initialize SNN'}
                </div>
            </Html>
        );
    }

    const { neurons, connections } = structure;
    console.log('[BrainVis3D] Rendering SNN', { neurons: neurons.length, connections: connections.length, activeSpikes });
    
    // Create an "isActive" map
    const activeMap = useMemo(() => {
        const map = {}; // neuronId -> boolean
        if (!activeSpikes) return map;
        
        // Helper to track indices per layer
        const layerIndices = {}; 
        
        neurons.forEach(n => {
            if (layerIndices[n.layer] === undefined) layerIndices[n.layer] = 0;
            const idx = layerIndices[n.layer]++;
            
            // Check if this neuron's index is in the active list for its layer
            if (activeSpikes[n.layer] && activeSpikes[n.layer].includes(idx)) {
                map[n.id] = true;
            }
        });
        return map;
    }, [neurons, activeSpikes]);


    // Parse connections for 3D lines
    const processedConnections = useMemo(() => {
        const posMap = {};
        neurons.forEach(n => posMap[n.id] = n.pos);
        
        return connections.map(c => ({
            ...c,
            start: posMap[c.srcId],
            end: posMap[c.tgtId]
        })).filter(c => c.start && c.end);
    }, [neurons, connections]);

    return (
        <group>
            {/* Visuals */}
            <group scale={[0.5, 0.5, 0.5]}> {/* Scale down to fit view */}
                {neurons.map(n => (
                    <Neuron 
                        key={n.id} 
                        {...n} 
                        isFired={activeMap[n.id]} 
                    />
                ))}
                {processedConnections.map((c, i) => (
                    <Axon 
                        key={i} 
                        start={c.start} 
                        end={c.end} 
                        isActive={activeMap[c.srcId]} 
                    />
                ))}
            </group>
        </group>
    );
};

export default BrainVis3D;
