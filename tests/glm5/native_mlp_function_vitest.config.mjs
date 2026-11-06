import base from './native_mlp_long_vitest.config.mjs';
export default {...base,test:{...base.test,include:[...base.test.include,'../tests/glm5/native_mlp_function_space.test.jsx','../tests/glm5/native_mlp_function_report.test.jsx','../tests/glm5/native_mlp_function_stream.test.jsx']}};
