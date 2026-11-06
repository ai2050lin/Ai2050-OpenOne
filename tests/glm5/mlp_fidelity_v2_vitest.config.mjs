import base from './mlp_atlas_vitest.config.mjs';
export default {...base,test:{...base.test,include:[...base.test.include,'../tests/glm5/mlp_fidelity_v2_client.test.jsx']}};
