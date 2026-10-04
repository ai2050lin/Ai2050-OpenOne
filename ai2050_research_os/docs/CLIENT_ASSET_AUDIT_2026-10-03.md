# 客户端可视化资产审计 [2026-10-03]

范围：`frontend/src` 全部 .jsx/.js（M1c）。分档判据：组件是否有可回查数据来源。
本文件是工程审计记录，不是研究事实源，不进入 Registry。

## 总览

| 分档 | 数量 | 含义 |
| --- | --- | --- |
| A 已接 Canonical Snapshot | 10 |  |
| B 服务端 API 驱动 | 69 |  |
| C 概念示意/演示数据 | 175 |  |
| D 非可视化或待人工复核 | 103 |  |

## 入口

- main.jsx 导入：[["App", "App.jsx"], ["App", "AppNew.jsx"], ["ErrorBoundary", "ErrorBoundary.jsx"]]

## 明细

| 文件 | 行数 | 依赖 | Snapshot | API | 演示常量 | 分档 |
| --- | --- | --- | --- | --- | --- | --- |
| AGICentralCommand.jsx | 369 | - | - | Y | 0 | B 服务端 API 驱动 |
| AGIChatPanel.jsx | 271 | - | - | Y | 0 | B 服务端 API 驱动 |
| AGIProgressDashboard.jsx | 417 | - | - | Y | 0 | B 服务端 API 驱动 |
| AGIVisualizationApp.jsx | 15 | - | - | - | 0 | D 非可视化或待人工复核 |
| App.jsx | 5205 | 3d | - | Y | 2 | B 服务端 API 驱动 |
| AppNew.jsx | 109 | - | - | - | 0 | D 非可视化或待人工复核 |
| BrainVis3D.jsx | 147 | 3d | - | - | 0 | C 概念示意/演示数据 |
| ErrorBoundary.jsx | 66 | - | - | - | 0 | D 非可视化或待人工复核 |
| FlowTubesVisualizer.jsx | 258 | 3d | - | Y | 0 | B 服务端 API 驱动 |
| GlassMatrix3D.jsx | 502 | 3d | - | Y | 0 | B 服务端 API 驱动 |
| GlobalTopologyDashboard.jsx | 74 | - | - | - | 0 | D 非可视化或待人工复核 |
| HLAIBlueprint.jsx | 1234 | - | - | Y | 0 | B 服务端 API 驱动 |
| HeadAnalysisPanel.jsx | 276 | - | - | Y | 0 | B 服务端 API 驱动 |
| HolonomyLoopVisualizer.jsx | 145 | 3d | - | - | 1 | C 概念示意/演示数据 |
| LanguageValidityPanel.jsx | 215 | - | - | Y | 0 | B 服务端 API 驱动 |
| LayerFirstNeuronScene.jsx | 361 | 3d | - | - | 0 | C 概念示意/演示数据 |
| ParameterEncoding3D.jsx | 724 | 3d | - | Y | 0 | B 服务端 API 驱动 |
| ResonanceField3D.jsx | 154 | 3d | - | - | 0 | C 概念示意/演示数据 |
| SimplePanel.jsx | 93 | - | - | - | 0 | D 非可视化或待人工复核 |
| StructureAnalysisPanel.jsx | 2660 | 3d | - | Y | 0 | B 服务端 API 驱动 |
| TDAVisualization3D.jsx | 233 | 3d | - | - | 0 | C 概念示意/演示数据 |
| TrainingDynamics3D.jsx | 102 | 3d | - | Y | 0 | B 服务端 API 驱动 |
| TrainingMonitor.jsx | 250 | recharts | - | Y | 0 | B 服务端 API 驱动 |
| agi_visualization.jsx | 11 | - | - | - | 0 | D 非可视化或待人工复核 |
| locales.js | 436 | - | - | - | 0 | D 非可视化或待人工复核 |
| main.jsx | 27 | - | - | - | 0 | D 非可视化或待人工复核 |
| main_new.jsx | 18 | - | - | - | 0 | D 非可视化或待人工复核 |
| AIRnD/AIRnDCodeGenTab.jsx | 162 | - | - | Y | 0 | B 服务端 API 驱动 |
| AIRnD/AIRnDConfigTab.jsx | 541 | - | - | Y | 0 | B 服务端 API 驱动 |
| AIRnD/AIRnDConsoleTab.jsx | 361 | - | Y | - | 1 | A 已接 Canonical Snapshot |
| AIRnD/AIRnDFindingsTab.jsx | 135 | - | - | - | 0 | D 非可视化或待人工复核 |
| AIRnD/AIRnDLogTab.jsx | 157 | - | - | - | 0 | D 非可视化或待人工复核 |
| AIRnD/AIRnDOrchestratorTab.jsx | 139 | - | - | Y | 1 | B 服务端 API 驱动 |
| AIRnD/AIRnDOverlay.jsx | 439 | - | - | Y | 1 | B 服务端 API 驱动 |
| AIRnD/aiRnDConfig.js | 199 | - | - | Y | 4 | B 服务端 API 驱动 |
| annotation/AnnotationApp.jsx | 675 | - | - | Y | 2 | B 服务端 API 驱动 |
| annotation/main.jsx | 11 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/AGIUnifiedTheoryEngine.jsx | 175 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/AgiConceptS1ToS7Summary.jsx | 174 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/AgiMilestoneProgressDashboard.jsx | 222 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/AgiTaskBlockDashboard.jsx | 166 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/AnchorRelativeTopologyGraph.jsx | 192 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/AppleNeuron3DTab.jsx | 76 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/AppleOrthogonalityDashboard.jsx | 255 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/AtlasControlDashboard.jsx | 1970 | - | - | Y | 0 | B 服务端 API 驱动 |
| blueprint/AttentionAbstractionRouterDashboard.jsx | 353 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/BrainDRealCocalibratedTwoLayerLawDashboard.jsx | 209 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/BrainLearnableRankingTwoLayerLawDashboard.jsx | 210 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/CategorySubspaceGraph.jsx | 206 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/ConceptProtocolFieldMappingDashboard.jsx | 367 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/ConceptSimilarityGraph.jsx | 180 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/ConceptSubspaceNetwork.jsx | 266 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/ConceptVectorAlgebraGraph.jsx | 141 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/DProblemAtlasDashboard.jsx | 287 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/DRealTaskCocalibratedTwoLayerLawDashboard.jsx | 164 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/DeepAnalysisTab.jsx | 70 | - | Y | - | 1 | A 已接 Canonical Snapshot |
| blueprint/DeepManifoldEvolutionGraph.jsx | 167 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/DnnBrainPuzzleBridgeDashboard.jsx | 364 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/EPS_SNN_Dashboard.jsx | 128 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/EpisodicConsolidationDashboard.jsx | 189 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/EvidenceKernelDashboard.jsx | 696 | - | - | Y | 2 | B 服务端 API 驱动 |
| blueprint/FeatureEmergenceAnimation.jsx | 183 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/FirstPrinciplesTheoryDashboard.jsx | 119 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/GLM5Tab.jsx | 5 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/GLM5TheorySimplifiedDashboard.jsx | 233 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/GPT5Tab.jsx | 5 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/GammaSynchronyGraph.jsx | 179 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/GateLawDynamicsDashboard.jsx | 189 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/GateLawNonlinearDynamicsDashboard.jsx | 191 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/GeminiTab.jsx | 1824 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/GeneratorNetworkRealLayerBandBridgeDashboard.jsx | 193 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/GrokkingDynamicsDashboard.jsx | 169 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/HRRPhaseRigorousDashboard.jsx | 252 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/HippocampalReplayGraph.jsx | 161 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/HyperSpaceBindingGraph.jsx | 131 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/KnowledgeCascadeTreeGraph.jsx | 132 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LanguageAnalysisSectionHeader.jsx | 27 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/LanguageAnalysisTab.jsx | 53 | - | - | - | 1 | C 概念示意/演示数据 |
| blueprint/LanguageFeatureAnalysis.jsx | 70 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/LanguageResearchTimeline.jsx | 75 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/LanguageTraceAtlas.jsx | 231 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/LearnableRankingTwoLayerUnifiedLawDashboard.jsx | 201 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LearnableTwoLayerUnifiedLawDashboard.jsx | 199 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseEarlyCoreDecouplingDashboard.jsx | 277 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseEndToEndRegionFamilyGeneratorNetworkDashboard.jsx | 198 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseMidphaseCoreStabilizationDashboard.jsx | 278 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulsePhaseConditionedCausalAtlasDashboard.jsx | 268 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseRecoveryPhaseTrainingLawDashboard.jsx | 272 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseRegionDifferentiatedSelectorDashboard.jsx | 290 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseRegionFamilyGeneratorDashboard.jsx | 193 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseRegionFamilyGeneratorNetworkDashboard.jsx | 221 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseRegionHeterogeneityDashboard.jsx | 250 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseRegionParameterFamilyLearnerDashboard.jsx | 193 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseStageDecomposedTrainingLawDashboard.jsx | 272 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseThreeStageTrainingClosureDashboard.jsx | 186 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseTrainableRegionFamilyGeneratorDashboard.jsx | 194 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/LocalPulseUnifiedMultiobjectiveTrainingLawDashboard.jsx | 276 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/ManifoldStructureGraph.jsx | 266 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/MechanismAgiBridgeDashboard.jsx | 277 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/OpenWorldContinuousGroundingDashboard.jsx | 191 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/OpenWorldGroundingActionLoopDashboard.jsx | 168 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/OpenWorldGroundingGoalStateDashboard.jsx | 150 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/OpenWorldLongHorizonGoalDashboard.jsx | 168 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/OpenWorldSubgoalPlanningDashboard.jsx | 168 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/OpenWorldVariablePlanningTrainableDashboard.jsx | 186 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/ParameterizedSharedModalityLawDashboard.jsx | 184 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/PhaseGatedUnifiedLawDashboard.jsx | 188 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/PredictiveCodingGraph.jsx | 84 | - | - | - | 1 | C 概念示意/演示数据 |
| blueprint/ProjectRoadmapTab.jsx | 125 | - | Y | - | 2 | A 已接 Canonical Snapshot |
| blueprint/ProtocolFieldBoundaryAtlasDashboard.jsx | 250 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekAttentionTopologyAtlasDashboard.jsx | 228 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekAttentionTopologyDashboard.jsx | 224 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekConceptProtocolFieldMappingDashboard.jsx | 373 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekHardOnlineToolInterfaceDashboard.jsx | 212 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekMechanismBridgeDashboard.jsx | 266 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekOnlineRecoveryChainDashboard.jsx | 191 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekProtocolFieldBoundaryAtlasDashboard.jsx | 255 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekRealModelRecoveryProxyAtlasDashboard.jsx | 189 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekRealModelStructureAtlasDashboard.jsx | 233 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekRelationBoundaryAtlasDashboard.jsx | 231 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekRelationTopologyBridgeDashboard.jsx | 258 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekSharedLayerBandCausalOrientationDashboard.jsx | 273 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekSharedLayerBandTargetedAblationDashboard.jsx | 255 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Qwen3DeepSeekSharedSupportHeadBridgeDashboard.jsx | 273 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/QwenAblationReport.jsx | 214 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/RealModelChannelEditDashboard.jsx | 174 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepAgiClosureDashboard.jsx | 192 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepBetaScanDashboard.jsx | 240 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepDynamicTemperatureDashboard.jsx | 294 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepGateTemperatureDashboard.jsx | 256 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepLengthScanDashboard.jsx | 193 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepLongHorizonJointTemperatureDashboard.jsx | 321 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepMemoryBoostDashboard.jsx | 194 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepMemoryGatedMultiscaleDashboard.jsx | 259 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepMemoryMultiscaleDashboard.jsx | 257 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepMinimalControlBridgeDashboard.jsx | 274 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepSegmentSummaryDashboard.jsx | 284 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepUltraLongHorizonTemperatureDashboard.jsx | 316 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealMultistepUnifiedControlManifoldDashboard.jsx | 242 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RealTaskDrivenTwoLayerUnifiedLawDashboard.jsx | 200 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RelationBoundaryAtlasDashboard.jsx | 224 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RelationCouplingTraceDashboard.jsx | 296 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RelationProtocolHeadAtlasDashboard.jsx | 270 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RelationProtocolHeadCausalDashboard.jsx | 148 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RelationProtocolHeadGroupCausalDashboard.jsx | 148 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RelationProtocolMesofieldScaleDashboard.jsx | 320 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/RelationToolJointGeneratorNetworkUpgradeDashboard.jsx | 243 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/ResearchAuditTab.jsx | 498 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/ResearchProgressTab.jsx | 124 | - | Y | - | 2 | A 已接 Canonical Snapshot |
| blueprint/SNNBrainMappingGraph.jsx | 174 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/Semantic4DBrainAugmentationDashboard.jsx | 108 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/Semantic4DBrainCandidateCoverageDashboard.jsx | 160 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Semantic4DBrainConstraintExpansionDashboard.jsx | 177 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Semantic4DBrainConstraintSweepDashboard.jsx | 158 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Semantic4DConfidenceCrossDomainDashboard.jsx | 138 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Semantic4DDomainCorrectionDashboard.jsx | 138 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/Semantic4DVectorDomainCorrectionDashboard.jsx | 138 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedAtomCausalUnificationDashboard.jsx | 254 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopBasisShellFactorizationDashboard.jsx | 142 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopConfidenceDimensionDashboard.jsx | 142 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopConfidenceMinimizationDashboard.jsx | 144 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopConfidenceSemanticsDashboard.jsx | 144 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopFamilyShellFactorizationDashboard.jsx | 142 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopMinimalInterfaceStateDashboard.jsx | 142 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopModalityDashboard.jsx | 176 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopOutputShellFactorizationDashboard.jsx | 139 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopProtocolShellFactorizationDashboard.jsx | 139 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopShellDashboard.jsx | 144 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SharedCentralLoopShellLocalizationDashboard.jsx | 139 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/StateVariableUnifiedLawDashboard.jsx | 190 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/SystemStatusTab.jsx | 129 | - | Y | - | 2 | A 已接 Canonical Snapshot |
| blueprint/TheoryRouteDashboard.jsx | 63 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/ToolStageGeneratorNetworkUpgradeDashboard.jsx | 239 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/ToyGroundingCreditContinualDashboard.jsx | 220 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/TrajectoryCodebookGraph.jsx | 147 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/TwoLayerUnifiedLawDashboard.jsx | 199 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/UnifiedStructureCompressionDashboard.jsx | 194 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/UnifiedUpdateLawDBridgeDashboard.jsx | 197 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/UnifiedUpdateLawDashboard.jsx | 189 | recharts | - | - | 0 | C 概念示意/演示数据 |
| blueprint/UniversalManifoldGraph.jsx | 149 | - | - | - | 1 | C 概念示意/演示数据 |
| blueprint/appleNeuronInfoPanelsBridge.jsx | 9 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/appleNeuronSceneBridge.jsx | 2 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/appleNeuronWorkspaceBridge.js | 2 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/audit3dBridge.js | 122 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/blueprintConfig.jsx | 1567 | - | - | - | 2 | C 概念示意/演示数据 |
| blueprint/blueprintRuntimeUtils.js | 62 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/languageAnalysisData.js | 593 | - | - | - | 6 | C 概念示意/演示数据 |
| blueprint/theoryRouteLatestData.js | 84 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/useModelSystemEvidence.js | 70 | - | - | Y | 0 | B 服务端 API 驱动 |
| blueprint/appleNeuron/ComponentDetailPanel3D.jsx | 1869 | 3d | - | - | 2 | C 概念示意/演示数据 |
| blueprint/appleNeuron/InfoPanels.jsx | 1282 | 3d | - | - | 0 | C 概念示意/演示数据 |
| blueprint/appleNeuron/LayerDetailView.jsx | 391 | - | - | - | 1 | C 概念示意/演示数据 |
| blueprint/appleNeuron/LayerExplodedView3D.jsx | 774 | 3d | - | - | 2 | C 概念示意/演示数据 |
| blueprint/appleNeuron/SceneComponents.jsx | 2067 | 3d | - | - | 2 | C 概念示意/演示数据 |
| blueprint/appleNeuron/componentModelSpec.js | 149 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/appleNeuron/constants.js | 457 | - | - | - | 10 | C 概念示意/演示数据 |
| blueprint/appleNeuron/useAppleNeuronWorkspace.js | 1574 | - | - | Y | 0 | B 服务端 API 驱动 |
| blueprint/appleNeuron/utils.js | 1879 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/data/agi_3d_client_scene_v1.js | 405 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/data/agi_layer_raw_scene_v1.js | 184 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/data/layer_parameter_state_overlay_persisted_v1.js | 296 | - | - | - | 1 | C 概念示意/演示数据 |
| blueprint/data/layer_parameter_state_overlay_v1.js | 276 | - | - | - | 1 | C 概念示意/演示数据 |
| blueprint/data/persisted_data_catalog_v1.js | 168 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/data/persisted_entity_registry_v1.js | 87 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/data/persisted_mechanism_chain_index_v1.js | 39 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/data/persisted_puzzle_records_v1.js | 149 | - | - | - | 0 | D 非可视化或待人工复核 |
| blueprint/data/persisted_repair_replay_sample_slots_v1.js | 167 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/AGIVisualizationDashboard.jsx | 62 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/AgileVisualizationDashboard.jsx | 87 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/AppleNeuronCore3D.jsx | 293 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/BasicEncodingPanel.jsx | 287 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/DNNAnalysis3DVisualization.jsx | 504 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/DNNAnalysisControlPanel.jsx | 528 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/DataSourcePanel.jsx | 66 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/FiberNetPanel.jsx | 704 | - | - | Y | 1 | B 服务端 API 驱动 |
| components/LCSVisualization.jsx | 298 | 3d | - | - | 2 | C 概念示意/演示数据 |
| components/LanguageResearchControlPanel.jsx | 1296 | - | - | - | 1 | C 概念示意/演示数据 |
| components/LanguageResearchDataPanel.jsx | 287 | - | - | - | 1 | C 概念示意/演示数据 |
| components/MainVisualizationArea.jsx | 243 | plotly | - | Y | 0 | B 服务端 API 驱动 |
| components/MultiLayer3DVisualization.jsx | 582 | 3d | - | Y | 0 | B 服务端 API 驱动 |
| components/QuickSearchPanel.jsx | 36 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/SimplifiedControlPanel.jsx | 136 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/StatusBar.jsx | 53 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/UnifiedDataExplorer.jsx | 258 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/WorkbenchLayout.jsx | 278 | - | - | - | 1 | C 概念示意/演示数据 |
| components/analysis/CompareAnalysisView.jsx | 469 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/analysis/CorrelationView.jsx | 490 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/analysis/StructureExtractView.jsx | 429 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/analysis/index.jsx | 59 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/app/GlobalConfigPanel.jsx | 82 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/app/LegacyVisualization.jsx | 291 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/app/MechanismWorkspace.jsx | 317 | - | Y | - | 2 | A 已接 Canonical Snapshot |
| components/app/NativeAtlasHeatmap.jsx | 69 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/NativeMlpParameterInspector.jsx | 59 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/NativeOperationParameterInspector.jsx | 52 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/NativeParameterInspector.jsx | 82 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/NativePathParameterInspector.jsx | 57 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/NativePrecisionParameterInspector.jsx | 55 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/NativeQKVParameterInspector.jsx | 66 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/NativeSequenceParameterInspector.jsx | 54 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/NativeSourceParameterInspector.jsx | 71 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/PatternFamilyAtlasControls.jsx | 270 | - | - | - | 1 | C 概念示意/演示数据 |
| components/app/RdcBindingAtlas.jsx | 111 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcConstructionAtlas.jsx | 193 | - | - | Y | 1 | B 服务端 API 驱动 |
| components/app/RdcFeatureAtlas.jsx | 259 | 3d | - | Y | 1 | B 服务端 API 驱动 |
| components/app/RdcFormationAtlas.jsx | 45 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcFormationEvidence.jsx | 61 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcFormationProgramEvidence.jsx | 37 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcJointAtlas.jsx | 205 | 3d | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcLawAtlas.jsx | 101 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcNaturalAnalyses.jsx | 66 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcNaturalQuestions.jsx | 140 | - | - | Y | 1 | B 服务端 API 驱动 |
| components/app/RdcOperatorAtlas.jsx | 163 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcPrefixAtlas.jsx | 111 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcQueryAtlas.jsx | 155 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcRelationStudy.jsx | 89 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcRuntimeAtlas.jsx | 87 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcSourceCoupling.jsx | 35 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/RdcUpdateAtlas.jsx | 142 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/app/ResearchEvidenceCockpit.jsx | 106 | - | Y | - | 0 | A 已接 Canonical Snapshot |
| components/app/ResearchHeatmapRoute.jsx | 3341 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/app/ResearchPlaybackPanel.jsx | 144 | - | - | - | 2 | C 概念示意/演示数据 |
| components/app/ResearchSpaceOverlay.jsx | 154 | 3d | Y | - | 0 | A 已接 Canonical Snapshot |
| components/app/ResearchStatusSummary.jsx | 140 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/app/SNNResearchDashboard.jsx | 172 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/evaluation/BenchmarkView.jsx | 442 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/evaluation/DNNAnalysisPanel.jsx | 445 | - | - | - | 3 | C 概念示意/演示数据 |
| components/evaluation/GeometricTestView.jsx | 387 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/evaluation/MilestoneProgressPanel.jsx | 169 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/evaluation/ProgressRiskDualAxis.jsx | 154 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/evaluation/ProgressTracker.jsx | 484 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/evaluation/RouteABComparePanel.jsx | 141 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/evaluation/RouteScoreTrendPanel.jsx | 160 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/evaluation/RouteTimelineBoard.jsx | 334 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/evaluation/StageSwimlaneBoard.jsx | 160 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/evaluation/TheoryAuditPanel.jsx | 140 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/evaluation/WeeklyReportPanel.jsx | 148 | - | - | Y | 0 | B 服务端 API 驱动 |
| components/evaluation/index.jsx | 59 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/intervention/ActivationIntervention.jsx | 462 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/intervention/GeometricIntervention.jsx | 542 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/intervention/SafetyIntervention.jsx | 463 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/intervention/index.jsx | 59 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/observation/ActivationView.jsx | 225 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/observation/GeometryView.jsx | 298 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/observation/LayerView.jsx | 191 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/observation/index.jsx | 60 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/BandFrequencyOverlay.jsx | 91 | 3d | - | - | 1 | C 概念示意/演示数据 |
| components/reverse/CausalFlowOverlay.jsx | 136 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/reverse/CrossDimensionMatrix.jsx | 108 | - | - | - | 3 | C 概念示意/演示数据 |
| components/reverse/DNNFeatureTabs.jsx | 118 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/DimensionGroup.jsx | 55 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/EncodingEquationOverlay.jsx | 138 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/reverse/FeatureDetailView.jsx | 193 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/LanguageDimensionSelector.jsx | 101 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/ModelComparisonView.jsx | 66 | - | - | - | 1 | C 概念示意/演示数据 |
| components/reverse/OrthogonalSubspaceOverlay.jsx | 88 | 3d | - | - | 0 | C 概念示意/演示数据 |
| components/reverse/PuzzleProgressView.jsx | 56 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/QuickTestPresets.jsx | 52 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/ReverseEngineeringDataPanel.jsx | 187 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/ReverseEngineeringOperationPanel.jsx | 172 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/ReverseEngineeringOverlay.jsx | 90 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/reverse/ViewModeSelect.jsx | 52 | - | - | - | 1 | C 概念示意/演示数据 |
| components/shared/DataComparisonView.jsx | 356 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/shared/DataDisplayTemplates.jsx | 304 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/shared/LoadingSpinner.jsx | 68 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/shared/MetricCard.jsx | 93 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/shared/OperationHistory.jsx | 284 | - | - | - | 0 | D 非可视化或待人工复核 |
| components/shared/SafeResponsiveContainer.jsx | 60 | recharts | - | - | 0 | C 概念示意/演示数据 |
| components/shared/index.js | 7 | - | - | - | 0 | D 非可视化或待人工复核 |
| config/api.js | 88 | - | - | Y | 0 | B 服务端 API 驱动 |
| config/dnnFeatures.js | 108 | - | - | - | 0 | D 非可视化或待人工复核 |
| config/languageDimensions.js | 133 | - | - | - | 0 | D 非可视化或待人工复核 |
| config/panels.js | 730 | - | - | - | 2 | C 概念示意/演示数据 |
| config/researchAssets.js | 28 | - | - | Y | 0 | B 服务端 API 驱动 |
| config/reverseColorMaps.js | 149 | - | - | - | 0 | D 非可视化或待人工复核 |
| config/testPresets.js | 138 | - | - | - | 1 | C 概念示意/演示数据 |
| config/app/controlPanelBlueprint.js | 31 | - | - | - | 0 | D 非可视化或待人工复核 |
| neural_vis/dataSourceAdapters.js | 524 | - | - | - | 0 | D 非可视化或待人工复核 |
| neural_vis/index.jsx | 879 | 3d | - | - | 1 | C 概念示意/演示数据 |
| neural_vis/main.jsx | 14 | - | - | - | 0 | D 非可视化或待人工复核 |
| neural_vis/components/HoverTooltip.jsx | 55 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/components/PuzzlePanel.jsx | 742 | - | - | - | 2 | C 概念示意/演示数据 |
| neural_vis/components/SceneHelpers.jsx | 30 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/hooks/useVisData.js | 218 | - | - | Y | 0 | B 服务端 API 驱动 |
| neural_vis/renderers/AtlasGraphRenderer.jsx | 270 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/CausalChainRenderer.jsx | 164 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/DarkMatterFlowRenderer.jsx | 195 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/FlowRenderer.jsx | 60 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/ForceLineRenderer.jsx | 130 | 3d | - | - | 1 | C 概念示意/演示数据 |
| neural_vis/renderers/GrammarRoleMatrixRenderer.jsx | 196 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/Heatmap3DRenderer.jsx | 51 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/LayerStackRenderer.jsx | 106 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/PatternFamilyNeuronAtlasRenderer.jsx | 779 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/PointCloudRenderer.jsx | 51 | - | - | - | 0 | D 非可视化或待人工复核 |
| neural_vis/renderers/RealUnitTraceRenderer.jsx | 150 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/SubspaceRenderer.jsx | 153 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/renderers/TrajectoryRenderer.jsx | 96 | 3d | - | - | 0 | C 概念示意/演示数据 |
| neural_vis/utils/constants.js | 476 | - | - | - | 0 | D 非可视化或待人工复核 |
| plugins/researchPlugins.js | 221 | - | - | - | 1 | C 概念示意/演示数据 |
| researchCenter/LanguageEncodingExplorer.jsx | 294 | - | - | Y | 0 | B 服务端 API 驱动 |
| researchCenter/LoopEngineeringWorkspace.jsx | 617 | - | - | Y | 2 | B 服务端 API 驱动 |
| researchCenter/ResearchCenter.jsx | 19 | - | - | - | 0 | D 非可视化或待人工复核 |
| researchCenter/ResearchWorkspace.jsx | 257 | - | - | - | 1 | C 概念示意/演示数据 |
| researchCenter/useLanguageEncodingCatalog.js | 24 | - | Y | Y | 0 | A 已接 Canonical Snapshot |
| researchCenter/useResearchWorkspace.js | 72 | - | - | Y | 0 | B 服务端 API 驱动 |
| researchKernel/heatmapResearchRoute.js | 474 | - | - | - | 1 | C 概念示意/演示数据 |
| researchKernel/patternAtlasEvidence.js | 144 | - | - | - | 0 | D 非可视化或待人工复核 |
| researchKernel/snnResearchState.js | 134 | - | - | - | 0 | D 非可视化或待人工复核 |
| researchKernel/snnRuntime.js | 21 | - | - | - | 0 | D 非可视化或待人工复核 |
| researchKernel/useLiveModelHeatmap.js | 164 | - | - | Y | 0 | B 服务端 API 驱动 |
| researchKernel/usePatternFamilyNeuronAtlas.js | 98 | - | - | Y | 0 | B 服务端 API 驱动 |
| researchKernel/useResearchKernel.js | 721 | - | - | Y | 0 | B 服务端 API 驱动 |
| researchKernel/useResearchSnapshot.js | 44 | - | Y | Y | 0 | A 已接 Canonical Snapshot |
| researchKernel/useResearchWorkspace.js | 211 | - | - | Y | 0 | B 服务端 API 驱动 |
| utils/backendAvailability.js | 31 | - | - | - | 0 | D 非可视化或待人工复核 |
| utils/colors.js | 124 | - | - | - | 0 | D 非可视化或待人工复核 |
| utils/runtimeClient.js | 69 | - | - | Y | 0 | B 服务端 API 驱动 |
