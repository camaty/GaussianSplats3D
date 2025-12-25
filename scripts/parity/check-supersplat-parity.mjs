import fs from 'node:fs';
import path from 'node:path';

import { SplatMaterial3D } from '../../src/splatmesh/SplatMaterial3D.js';

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../..');

const supersplatRoot = process.env.SUPERSPLAT_ROOT
  ? path.resolve(process.env.SUPERSPLAT_ROOT)
  : null;

const candidatePlayCanvasBuildPaths = [
  supersplatRoot ? path.resolve(supersplatRoot, 'node_modules/playcanvas/build/playcanvas.mjs') : null,
  // Default workspace layout in this environment
  path.resolve('/root/supersplat/node_modules/playcanvas/build/playcanvas.mjs'),
  // Fallback: if repos are adjacent
  path.resolve(repoRoot, '../../supersplat/node_modules/playcanvas/build/playcanvas.mjs')
].filter(Boolean);

const playcanvasBuildPath = candidatePlayCanvasBuildPaths.find((p) => fs.existsSync(p));

function fail(msg) {
  console.error(`FAIL: ${msg}`);
  process.exitCode = 1;
}

function ok(msg) {
  console.log(`OK: ${msg}`);
}

function mustContain(haystack, needle, label) {
  if (!haystack.includes(needle)) {
    fail(`${label} missing expected snippet:\n${needle}`);
  } else {
    ok(label);
  }
}

function extractBacktickVar(source, varName) {
  const re = new RegExp('var\\s+' + varName + '\\s*=\\s*`([\\s\\S]*?)`;');
  const m = source.match(re);
  if (!m) {
    throw new Error(`Could not extract ${varName} from playcanvas build`);
  }
  return m[1];
}

function extractFunctionSource(source, fnName) {
  const idx = source.indexOf(`function ${fnName}(`);
  if (idx < 0) {
    throw new Error(`Could not find function ${fnName}() in source`);
  }
  const braceStart = source.indexOf('{', idx);
  if (braceStart < 0) {
    throw new Error(`Could not find opening brace for function ${fnName}()`);
  }

  // Lightweight brace-matching that skips strings and comments.
  let i = braceStart;
  let depth = 0;
  let inSingle = false;
  let inDouble = false;
  let inTemplate = false;
  let inLineComment = false;
  let inBlockComment = false;
  let escaped = false;

  for (; i < source.length; i++) {
    const ch = source[i];
    const next = i + 1 < source.length ? source[i + 1] : '';

    if (inLineComment) {
      if (ch === '\n') inLineComment = false;
      continue;
    }
    if (inBlockComment) {
      if (ch === '*' && next === '/') {
        inBlockComment = false;
        i++;
      }
      continue;
    }

    if (inSingle) {
      if (!escaped && ch === "'") inSingle = false;
      escaped = !escaped && ch === '\\';
      continue;
    }
    if (inDouble) {
      if (!escaped && ch === '"') inDouble = false;
      escaped = !escaped && ch === '\\';
      continue;
    }
    if (inTemplate) {
      if (!escaped && ch === '`') inTemplate = false;
      escaped = !escaped && ch === '\\';
      continue;
    }

    escaped = false;

    if (ch === '/' && next === '/') {
      inLineComment = true;
      i++;
      continue;
    }
    if (ch === '/' && next === '*') {
      inBlockComment = true;
      i++;
      continue;
    }

    if (ch === "'") {
      inSingle = true;
      continue;
    }
    if (ch === '"') {
      inDouble = true;
      continue;
    }
    if (ch === '`') {
      inTemplate = true;
      continue;
    }

    if (ch === '{') {
      depth++;
    } else if (ch === '}') {
      depth--;
      if (depth === 0) {
        return source.slice(idx, i + 1);
      }
    }
  }

  throw new Error(`Could not match braces for function ${fnName}()`);
}

function normalize(s) {
  return s.replace(/\s+/g, ' ').trim();
}

function main() {
  if (!playcanvasBuildPath) {
    throw new Error(
      'PlayCanvas build not found. Tried:\n' + candidatePlayCanvasBuildPaths.map((p) => `- ${p}`).join('\n') +
      '\n\nSet SUPERSPLAT_ROOT to your /root/supersplat path if needed.'
    );
  }

  const pc = fs.readFileSync(playcanvasBuildPath, 'utf8');
  const pcCorner = extractBacktickVar(pc, 'gsplatCornerVS\\$1');
  const pcPS = extractBacktickVar(pc, 'gsplatPS\\$1');
  const pcSortWorker = extractFunctionSource(pc, 'SortWorker');

  // Build GaussianSplats3D shader sources (3D mode, with/without AA and dithering).
  const matNoAA = SplatMaterial3D.build(false, false, false, 2048, 1.0, false, 0, 0.3, false);
  const matAA = SplatMaterial3D.build(false, false, true, 2048, 1.0, false, 0, 0.3, false);
  const matDither = SplatMaterial3D.build(false, false, false, 2048, 1.0, false, 0, 0.3, true);

  const vs = matNoAA.vertexShader;
  const fsSrc = matNoAA.fragmentShader;
  const vsAA = matAA.vertexShader;
  const fsDither = matDither.fragmentShader;

  // --- Vertex stage parity (gsplatCornerVS) ---
  // We do targeted snippet checks because naming/structs differ between engines.

  mustContain(
    normalize(vs),
    normalize('float focalPx = viewport.x * projectionMatrix[0][0];'),
    'VS: focal scalar (viewport.x * proj[0][0])'
  );

  mustContain(
    normalize(vs),
    normalize('float J1 = focalPx / v.z;'),
    'VS: J1 computation'
  );

  mustContain(
    normalize(vs),
    normalize('vec2 J2 = -J1 / v.z * v.xy;'),
    'VS: J2 computation'
  );

  mustContain(
    normalize(vs),
    normalize('mat3 W = transpose(mat3(transformModelViewMatrix));'),
    'VS: W = transpose(mat3(modelView))'
  );

  mustContain(
    normalize(vs),
    normalize('mat3 cov2Dm = transpose(T) * Vrk * T;'),
    'VS: projected covariance (transpose(T) * Vrk * T)'
  );

  // Clip-space offset add (this is the key difference vs NDC-based implementations).
  mustContain(
    normalize(vs),
    normalize('gl_Position = clipCenter + vec4(cornerOffset, 0.0, 0.0);'),
    'VS: gl_Position uses clipCenter + offset'
  );

  // --- AA stage parity (GSPLAT_AA) ---
  // PlayCanvas uses detOrig on cov, detBlur on (cov + 0.3). Our AA path matches via kernel2DSize default=0.3.
  mustContain(
    normalize(vsAA),
    normalize('float detOrig = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];'),
    'AA: detOrig'
  );
  mustContain(
    normalize(vsAA),
    normalize('vColor.a *= sqrt(max(detOrig / detBlur, 0.0));'),
    'AA: alpha *= sqrt(detOrig/detBlur)'
  );

  // --- Fragment stage parity (gsplatPS) ---
  mustContain(
    normalize(fsSrc),
    normalize('float A8 = dot(vPosition, vPosition);'),
    'FS: uses vPosition dot'
  );
  mustContain(
    normalize(fsSrc),
    normalize('if (alpha < 1.0 / 255.0) discard;'),
    'FS: alpha discard 1/255'
  );
  mustContain(
    normalize(fsSrc),
    normalize('gl_FragColor = vec4(color * alpha, alpha);'),
    'FS: premultiplied output'
  );

  // --- Dither stage parity (seed) ---
  // PlayCanvas uses opacityDither(alpha, id * 0.013)
  if (!normalize(pcPS).includes(normalize('opacityDither(alpha, id * 0.013)'))) {
    fail('PlayCanvas reference: expected opacityDither(alpha, id * 0.013) not found');
  } else {
    ok('PlayCanvas reference: opacityDither(alpha, id * 0.013) present');
  }

  mustContain(
    normalize(fsDither),
    normalize('ignNoise(gl_FragCoord.xy, ditherJitter, vDitherId * 0.013)'),
    'FS(dither): seed uses vDitherId*0.013'
  );

  // Sanity: confirm PlayCanvas corner chunk contains the culling/offset shape we mirror.
  if (!normalize(pcCorner).includes(normalize('corner.offset = (source.cornerUV.x * v1 + source.cornerUV.y * v2) * c;'))) {
    fail('PlayCanvas reference: expected corner.offset formula not found');
  } else {
    ok('PlayCanvas reference: corner.offset formula present');
  }

  // --- Sorting stage parity (GSplatSorter / SortWorker) ---
  // PlayCanvas sorts by projected depth along camera direction in model-space:
  // key ~ dot(center, cameraDirection) (cameraPosition subtraction is constant for ordering).
  mustContain(
    normalize(pcSortWorker),
    normalize('const d = (x * dx + y * dy + z * dz - minDist) / binRange;'),
    'Sort(PlayCanvas): per-splat depth uses dot(center, cameraDir)'
  );
  mustContain(
    normalize(pcSortWorker),
    normalize('const cameraDist = px * dx + py * dy + pz * dz;'),
    'Sort(PlayCanvas): cameraDist = dot(cameraPos, cameraDir)'
  );
  mustContain(
    normalize(pcSortWorker),
    normalize('return centers[o++] * dx + centers[o++] * dy + centers[o] * dz - cameraDist;'),
    'Sort(PlayCanvas): front/back test uses dot(center, dir) - cameraDist'
  );

  // GaussianSplats3D uses the clip-space z row of MVP as its sort distance (monotonic with view depth).
  // Verify that the implementation uses MVP elements [2, 6, 10, 14] consistently, and passes the MVP to WASM.
  const gsViewerPath = path.resolve(repoRoot, 'src/Viewer.js');
  const gsSplatMeshPath = path.resolve(repoRoot, 'src/splatmesh/SplatMesh.js');
  const gsSortWorkerPath = path.resolve(repoRoot, 'src/worker/SortWorker.js');

  const gsViewer = fs.readFileSync(gsViewerPath, 'utf8');
  const gsSplatMesh = fs.readFileSync(gsSplatMeshPath, 'utf8');
  const gsSortWorker = fs.readFileSync(gsSortWorkerPath, 'utf8');

  mustContain(
    normalize(gsViewer),
    normalize('mvpMatrix.copy(this.camera.matrixWorld).invert();'),
    'Sort(GS3D): MVP starts from inverse(camera.matrixWorld)'
  );
  mustContain(
    normalize(gsViewer),
    normalize('mvpMatrix.premultiply(mvpCamera.projectionMatrix);'),
    'Sort(GS3D): MVP premultiplies projectionMatrix'
  );
  mustContain(
    normalize(gsViewer),
    normalize("'modelViewProj': mvpMatrix.elements"),
    'Sort(GS3D): sends MVP elements to sorter'
  );

  mustContain(
    normalize(gsSplatMesh),
    normalize('const viewProj = [modelViewProjMatrix.elements[2], modelViewProjMatrix.elements[6], modelViewProjMatrix.elements[10], modelViewProjMatrix.elements[14]];'),
    'Sort(GS3D/GPU): uses MVP z-row [2,6,10,14]'
  );
  mustContain(
    normalize(gsSplatMesh),
    normalize('distance = center.x * modelViewProj.x + center.y * modelViewProj.y + center.z * modelViewProj.z + modelViewProj.w * center.w;'),
    'Sort(GS3D/GPU): distance is dot(center, MVP.zRow)'
  );
  mustContain(
    normalize(gsSortWorker),
    normalize('new Float32Array(wasmMemory, modelViewProjOffset, 16).set(modelViewProj);'),
    'Sort(GS3D/WASM): uploads MVP to WASM memory'
  );
  mustContain(
    normalize(gsSortWorker),
    normalize('wasmInstance.exports.sortIndexes(indexesToSortOffset, centersOffset, precomputedDistancesOffset,'),
    'Sort(GS3D/WASM): calls WASM sorter'
  );

  if (process.exitCode) {
    console.error('\nParity check failed. Inspect shader changes or update checks for intentional deviations.');
    process.exit(process.exitCode);
  }

  console.log('\nAll parity checks passed.');
}

main();
