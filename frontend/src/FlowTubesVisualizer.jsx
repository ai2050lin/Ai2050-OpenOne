
import { Text } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import axios from 'axios';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';

const API_BASE = 'http://localhost:5002';

function TubePath({ path, color, label, metrics, radius = 0.5, onHover }) {
  const curve = useMemo(() => {
    if (!path || path.length < 2) return null;
    const points = path.map(p => new THREE.Vector3(p[0] * 0.1, p[1] * 0.1, p[2] * 0.1)); // Scale down
    return new THREE.CatmullRomCurve3(points);
  }, [path]);

  const tubeRef = useRef();

  // Geometry with variable radius
  const geometry = useMemo(() => {
    if (!curve) return null;
    
    // Custom TubeGeometry to support variable radius
    // We can't easily do variable radius with standard TubeGeometry without modifying the generator.
    // However, we can use the 'scale' property in a shader or just use a standard tube for now
    // and mapped colors. 
    // To truly do variable radius, we'd need a custom mesh generation.
    // For this demonstration, let's stick to uniform radius but modulate COLOR along the tube if possible,
    // or just use the Mean Surprise to set the Global Radius of the tube.
    
    // Let's use MEAN Surprise to modulate the overall thickness of the tube.
    let meanSurprise = 0;
    if (metrics && metrics.surprise) {
        meanSurprise = metrics.surprise.reduce((a, b) => a + b, 0) / metrics.surprise.length;
    }
    
    // Base radius + Surprise factor
    // Normalize surprise roughly (assuming variance is small, e.g. 0.0 to 5.0)
    const effectiveRadius = radius * (1 + meanSurprise * 0.5); 
    
    return new THREE.TubeGeometry(curve, 64, effectiveRadius, 8, false);
  }, [curve, metrics, radius]);

  // Material with dynamic color based on Curvature
  // Since we can't easily do vertex colors on TubeGeometry without custom shaders or vertex attribute manipulation,
  // we will simply change the overall color of the tube based on its "Average Curvature".
  // High curvature -> Shift towards Red. Low curvature -> Keep original color.
  
  const effectiveColor = useMemo(() => {
      if (!metrics || !metrics.curvature) return color;
      
      const meanCurvature = metrics.curvature.reduce((a, b) => a + b, 0) / metrics.curvature.length;
      // Curvature is 0 to 2 (1 - cos). 
      // If curvature > 0.1, it's significant.
      
      const curvatureFactor = Math.min(meanCurvature * 5, 1); // Sensitivity
      
      const baseColor = new THREE.Color(color);
      const stressColor = new THREE.Color("#ff0000"); // Red for high curvature/stress
      
      return baseColor.lerp(stressColor, curvatureFactor);
  }, [color, metrics]);

  // Animation for "flow" effect (pulse)
  useFrame((state) => {
      if (tubeRef.current) {
          // Simple pulse effect on scale - modulate speed by Velocity if available?
          let speed = 2;
          if (metrics && metrics.velocity) {
              const meanVel = metrics.velocity.reduce((a, b) => a + b, 0) / metrics.velocity.length;
              speed = 2 + meanVel; // Faster velocity = Faster pulse
          }
          
          const scale = 1 + Math.sin(state.clock.elapsedTime * speed) * 0.02;
          tubeRef.current.scale.set(scale, scale, scale);
      }
  });

  if (!curve || !geometry) return null;

  return (
    <group>
      <mesh 
        ref={tubeRef}
        geometry={geometry}
        onPointerOver={(e) => {
            e.stopPropagation();
            onHover && onHover({ type: 'tube', label, path, metrics });
        }}
        onPointerOut={() => onHover && onHover(null)}
      >
        <meshPhysicalMaterial 
            color={effectiveColor} 
            transparent 
            opacity={0.6} 
            roughness={0.1}
            metalness={0.1}
            transmission={0.5}
            thickness={1.0}
            side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Start Label */}
      <Text
        position={curve.points[0]}
        fontSize={1}
        color={effectiveColor}
        anchorX="center"
        anchorY="bottom"
      >
        {label}
      </Text>
    </group>
  );
}

export default function FlowTubesVisualizer() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hovered, setHovered] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get(`${API_BASE}/nfb_flow_tubes`);
        console.log("Loaded Flow Tubes:", res.data); // Debug log
        
        // Transform incoming data if needed
        // The API returns { tubes: [...], layers: 13 }
        // The paths are raw coordinates. We might need to center/scale them.
        // Assuming coordinates are already reasonable, but maybe dividing by 10 as I did before is good to keep scene manageable.
        
        setData(res.data);
      } catch (err) {
        console.error("Error fetching flow tubes:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
        <Text position={[0, 0, 0]} color="white" fontSize={1} anchorX="center" anchorY="center">
            Loading Deep Dynamics...
        </Text>
    );
  }
  
  if (error) {
    return (
        <Text position={[0, 0, 0]} color="#ff4444" fontSize={1} anchorX="center" anchorY="center">
            Error: {error}
        </Text>
    );
  }

  if (!data || !data.tubes || data.tubes.length === 0) {
      return (
        <group>
             <Text position={[0, 0, 0]} color="orange" fontSize={1} anchorX="center" anchorY="center">
                No Flow Data. Run analysis first.
            </Text>
            <gridHelper args={[20, 20, 0x444444, 0x222222]} />
        </group>
      );
  }

  return (
    <group>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4488ff" />
        
        {data.tubes.map((tube, idx) => (
            <TubePath 
                key={idx}
                path={tube.path}
                color={tube.color}
                label={tube.label}
                metrics={tube.metrics} // Pass GUT metrics
                radius={tube.radius || 0.2}
                onHover={setHovered}
            />

        ))}

        {/* Dynamic Label on Hover */}
        {hovered && (
             <group position={[0, 5, 0]}>
                <Text fontSize={1.5} color="white" anchorX="center" anchorY="bottom" outlineWidth={0.05} outlineColor="black">
                    {hovered.label}
                </Text>
                {hovered.metrics && (
                    <Text position={[0, -1.5, 0]} fontSize={0.8} color="#aaaaaa" anchorX="center" anchorY="top" outlineWidth={0.02} outlineColor="black">
                        {`Surprise: ${(hovered.metrics.surprise?.reduce((a,b)=>a+b,0)/hovered.metrics.surprise?.length).toFixed(2)} | Curvature: ${(hovered.metrics.curvature?.reduce((a,b)=>a+b,0)/hovered.metrics.curvature?.length).toFixed(2)}`}
                    </Text>
                )}
            </group>
        )}
        
        {/* Helper Grid for visual context */}
        <gridHelper args={[50, 50, 0x222222, 0x111111]} position={[0, -5, 0]} />
        <axesHelper args={[5]} />
    </group>
  );
}
