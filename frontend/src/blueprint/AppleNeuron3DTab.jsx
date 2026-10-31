/**
 * AppleNeuron3DTab - 横向深度神经网络 3D 可视化入口
 *
 * 本文件仅作为 re-export 聚合层，所有实现已拆分到 appleNeuron/ 子目录：
 *   - constants.js     : 常量、配置数据
 *   - utils.js         : 工具函数
 *   - SceneComponents.jsx : 3D 场景组件
 *   - useAppleNeuronWorkspace.js : workspace hook
 *   - InfoPanels.jsx    : 信息面板组件
 */

// ---- 场景组件 ----
export { AppleNeuronSceneContent, AppleNeuronMainScene } from './appleNeuron/SceneComponents';

// ---- Workspace hook ----
export { useAppleNeuronWorkspace } from './appleNeuron/useAppleNeuronWorkspace';

// ---- 信息面板 ----
export {
  AppleNeuronEncodingInfoPanels,
  AppleNeuronResearchAssetInfoPanel,
  AppleNeuronSelectedLegendPanels,
  AppleNeuronCategoryComparePanel,
  AppleNeuronCompareFilterPanel,
  AppleNeuronGeneratedConceptSetsPanel,
  AppleNeuronMultidimSettingsPanel,
} from './appleNeuron/InfoPanels';

// ---- 常量（部分外部需要） ----
export {
  LAYER_COUNT,
  DFF,
  ROLE_COLORS,
  FRUIT_COLORS,
  ICSPB_THEORY_OBJECTS,
  ANALYSIS_MODE_OPTIONS,
  APPLE_ANIMATION_OPTIONS,
  MODE_VISUALS,
  DEFAULT_LANGUAGE_FOCUS,
} from './appleNeuron/constants';

// ---- 工具函数（部分外部需要） ----
export {
  toSafeNumber,
  neuronToPosition,
  buildConceptNeuronSet,
  buildFamilyPatchViewModel,
} from './appleNeuron/utils';

// ---- 主入口组件 ----
import LanguageResearchControlPanel from '../components/LanguageResearchControlPanel';
import { useAppleNeuronWorkspace } from './appleNeuron/useAppleNeuronWorkspace';
import { AppleNeuronMainScene } from './appleNeuron/SceneComponents';

export function AppleNeuron3DTab({ panelPosition = 'right', sceneHeight = '74vh', workspace: externalWorkspace } = {}) {
  const internalWorkspace = useAppleNeuronWorkspace();
  const workspace = externalWorkspace || internalWorkspace;
  const isPanelLeft = panelPosition === 'left';

  return (
    <div style={{ animation: 'roadmapFade 0.6s ease-out', display: 'grid', gridTemplateColumns: isPanelLeft ? '340px 1fr' : '1fr 340px', gap: 20 }}>
      {isPanelLeft ? (
        <>
          <LanguageResearchControlPanel workspace={workspace} structureTab="circuit" />
          <AppleNeuronMainScene workspace={workspace} sceneHeight={sceneHeight} />
        </>
      ) : (
        <>
          <AppleNeuronMainScene workspace={workspace} sceneHeight={sceneHeight} />
          <LanguageResearchControlPanel workspace={workspace} structureTab="circuit" />
        </>
      )}
    </div>
  );
}
