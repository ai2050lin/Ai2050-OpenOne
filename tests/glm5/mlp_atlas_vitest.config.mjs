import base from './native_mlp_function_vitest.config.mjs';
export default {...base,test:{...base.test,include:[...base.test.include,'../tests/glm5/mlp_atlas_client.test.jsx','../tests/glm5/mlp_atlas_audit_loading.test.jsx']}};
