import { defineConfig } from '../../frontend/node_modules/vitest/dist/config.js';
import { fileURLToPath } from 'node:url';
export default defineConfig({
  esbuild:{jsx:'automatic'},
  resolve:{alias:{react:fileURLToPath(new URL('../../frontend/node_modules/react',import.meta.url)),vitest:fileURLToPath(new URL('../../frontend/node_modules/vitest/dist/index.js',import.meta.url))}},
  test:{environment:'happy-dom',env:{VITE_API_BASE:'http://localhost:5001'},include:['../tests/glm5/phase2764_client.test.jsx']},
});
