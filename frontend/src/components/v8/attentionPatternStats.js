// Shared with the file:// design preview. Only accept complete attention rows
// (or a complete partition of one row), allowing BF16 rounding in the sum.
(() => {
  const sumTolerance = 0.005;

  function summarize(weights, {topK = 5} = {}) {
    if (!(Array.isArray(weights) || ArrayBuffer.isView(weights)) || !Number.isSafeInteger(weights.length) || !weights.length) {
      return {ok:false, reason:'缺少当前头在该查询位置的完整注意力行。'};
    }
    if (!Number.isSafeInteger(topK) || topK < 1) {
      return {ok:false, reason:'累计来源数须为正整数。'};
    }
    let total = 0;
    for (const weight of weights) {
      if (!Number.isFinite(weight) || weight < 0 || weight > 1) {
        return {ok:false, reason:'注意力行包含缺失、非有限或超出 [0, 1] 的权重。'};
      }
      total += weight;
    }
    if (Math.abs(total - 1) > sumTolerance) {
      return {ok:false, reason:'权重总和为 '+total.toFixed(5)+'，超出 1 ± 0.005。请提供所有来源，分组时须包含其余 token。'};
    }
    let entropyBits = 0;
    const ranked = Array.from(weights, (weight,index) => {
      const probability = weight / total;
      if (probability > 0) entropyBits -= probability * Math.log2(probability);
      return {index,weight,probability};
    }).sort((a,b) => b.weight - a.weight || a.index - b.index);
    const k = Math.min(topK, ranked.length);
    const topMass = ranked.slice(0,k).reduce((mass,source) => mass + source.probability,0);
    return {ok:true, count:ranked.length, total, entropyBits, effectiveSources:2 ** entropyBits, ranked, topK:k, topMass};
  }

  globalThis.AI2050AttentionStats = Object.freeze({summarize});
})();
