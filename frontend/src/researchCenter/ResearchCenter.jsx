import { LoopEngineeringWorkspace } from './LoopEngineeringWorkspace';
import { ResearchWorkspace } from './ResearchWorkspace';

import './ResearchCenter.css';

/**
 * Single client entry for the two non-3D research workspaces.
 *
 * Theory is organized around accumulated research objects. AI R&D is organized
 * around evidence gates and agent roles. Neither branch changes the existing
 * Three.js/R3F LLM scene.
 */
export function ResearchCenter({ scope = 'theory', mode, onClose }) {
  if (scope === 'rnd') {
    return <LoopEngineeringWorkspace mode={mode || 'sidebar'} onClose={onClose} />;
  }
  return <ResearchWorkspace mode={mode || 'overlay'} onClose={onClose} />;
}
