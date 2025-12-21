import * as THREE from 'three';
import { SplatMaterial } from './SplatMaterial.js';

export class SplatMaterial3D {

    /**
     * @param {boolean} antialiased High-Fidelityレンダリング（Mip-Splatting）を有効にするか
     */
    static build(dynamicMode = false, enableOptionalEffects = false, antialiased = false, maxScreenSpaceSplatSize = 2048,
                 splatScale = 1.0, pointCloudModeEnabled = false, maxSphericalHarmonicsDegree = 0, kernel2DSize = 0.3,
                 ditherEnabled = false) {

        const customVertexVars = `
            uniform vec2 covariancesTextureSize;
            uniform highp sampler2D covariancesTexture;
            uniform highp usampler2D covariancesTextureHalfFloat;
            uniform int covariancesAreHalfFloat;

            void fromCovarianceHalfFloatV4(uvec4 val, out vec4 first, out vec4 second) {
                vec2 r = unpackHalf2x16(val.r);
                vec2 g = unpackHalf2x16(val.g);
                vec2 b = unpackHalf2x16(val.b);

                first = vec4(r.x, r.y, g.x, g.y);
                second = vec4(b.x, b.y, 0.0, 0.0);
            }
        `;

        let vertexShaderSource = SplatMaterial.buildVertexShaderBase(dynamicMode, enableOptionalEffects,
                                                                     maxSphericalHarmonicsDegree, customVertexVars);
        vertexShaderSource += SplatMaterial3D.buildVertexShaderProjection(antialiased, enableOptionalEffects,
                                                                          maxScreenSpaceSplatSize, kernel2DSize);
        const fragmentShaderSource = SplatMaterial3D.buildFragmentShader();

        const uniforms = SplatMaterial.getUniforms(dynamicMode, enableOptionalEffects,
                                                   maxSphericalHarmonicsDegree, splatScale, pointCloudModeEnabled);

        uniforms['covariancesTextureSize'] = { 'type': 'v2', 'value': new THREE.Vector2(1024, 1024) };
        uniforms['covariancesTexture'] = { 'type': 't', 'value': null };
        uniforms['covariancesTextureHalfFloat'] = { 'type': 't', 'value': null };
        uniforms['covariancesAreHalfFloat'] = { 'type': 'i', 'value': 0 };

        // Supersplat parity uniforms
        uniforms['renderMode'] = { 'type': 'i', 'value': 0 };
        uniforms['ringSize'] = { 'type': 'f', 'value': 0.0 };
        uniforms['ditherMode'] = { 'type': 'i', 'value': ditherEnabled ? 1 : 0 };
        uniforms['ditherJitter'] = { 'type': 'v2', 'value': new THREE.Vector2(0, 0) };
        uniforms['toneMapMode'] = { 'type': 'i', 'value': 0 };
        uniforms['gammaMode'] = { 'type': 'i', 'value': 1 };

        const material = new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexShader: vertexShaderSource,
            fragmentShader: fragmentShaderSource,
            transparent: true,
            alphaTest: 1.0,
            blending: ditherEnabled ? THREE.NoBlending : THREE.CustomBlending,
            blendSrc: THREE.OneFactor,
            blendDst: THREE.OneMinusSrcAlphaFactor,
            blendEquation: THREE.AddEquation,
            depthTest: true,
            depthWrite: !!ditherEnabled,
            side: THREE.DoubleSide
        });

        return material;
    }

    static buildVertexShaderProjection(antialiased, enableOptionalEffects, maxScreenSpaceSplatSize, kernel2DSize) {
        let vertexShaderSource = `
            vec4 sampledCovarianceA;
            vec4 sampledCovarianceB;
            vec3 cov3D_M11_M12_M13;
            vec3 cov3D_M22_M23_M33;
            if (covariancesAreHalfFloat == 0) {
                sampledCovarianceA = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset, covariancesTextureSize));
                sampledCovarianceB = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset + uint(1), covariancesTextureSize));
                cov3D_M11_M12_M13 = vec3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) + vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) + vec3(sampledCovarianceB.gba) * fOddOffset;
            } else {
                uvec4 sampledCovarianceU = texture(covariancesTextureHalfFloat, getDataUV(1, 0, covariancesTextureSize));
                fromCovarianceHalfFloatV4(sampledCovarianceU, sampledCovarianceA, sampledCovarianceB);
                cov3D_M11_M12_M13 = sampledCovarianceA.rgb;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg);
            }
        
            mat3 Vrk = mat3(
                cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z,
                cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x, cov3D_M22_M23_M33.y,
                cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z
            );

            mat3 J;
            if (orthographicMode == 1) {
                J = transpose(mat3(orthoZoom, 0.0, 0.0, 0.0, orthoZoom, 0.0, 0.0, 0.0, 0.0));
            } else {
                float s = 1.0 / (viewCenter.z * viewCenter.z);
                J = mat3(
                    focal.x / viewCenter.z, 0., -(focal.x * viewCenter.x) * s,
                    0., focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s,
                    0., 0., 0.
                );
            }

            mat3 W = transpose(mat3(transformModelViewMatrix));
            mat3 T = W * J;
            mat3 cov2Dm = transpose(T) * Vrk * T;
        `;

        // --- High Fidelity: Mip-Splatting Implementation ---
        if (antialiased) {
            vertexShaderSource += `
                float detOrig = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
                cov2Dm[0][0] += ${kernel2DSize};
                cov2Dm[1][1] += ${kernel2DSize};
                float detBlur = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
                vColor.a *= sqrt(max(detOrig / detBlur, 0.0));
                if (vColor.a < minAlpha) return;
            `;
        } else {
            vertexShaderSource += `
                cov2Dm[0][0] += ${kernel2DSize};
                cov2Dm[1][1] += ${kernel2DSize};
            `;
        }
        // ---------------------------------------------------

        vertexShaderSource += `
            vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);
            float a = cov2Dv.x;
            float d = cov2Dv.z;
            float b = cov2Dv.y;
            float D = a * d - b * b;
            float trace = a + d;
            float traceOver2 = 0.5 * trace;
            float term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
            float eigenValue1 = traceOver2 + term2;
            float eigenValue2 = traceOver2 - term2;

            if (pointCloudModeEnabled == 1) {
                eigenValue1 = eigenValue2 = 0.2;
            }

            if (eigenValue2 <= 0.0) return;

            vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
            vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

            vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), ${parseInt(maxScreenSpaceSplatSize)}.0);
            vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), ${parseInt(maxScreenSpaceSplatSize)}.0);

            if (length(basisVector1) < 2.0 && length(basisVector2) < 2.0) return;
        `;

        if (enableOptionalEffects) {
            vertexShaderSource += `
                vColor.a *= splatOpacityFromScene;
            `;
        }

        vertexShaderSource += `
            float clipFactor = 1.0;
            if (vColor.a > 0.0) {
                clipFactor = min(1.0, sqrt(-log(1.0 / (255.0 * vColor.a))) / 2.0);
            }
            vPosition *= clipFactor;

            vec2 ndcOffset = vec2(vPosition.x * basisVector1 + vPosition.y * basisVector2) *
                             basisViewport * 2.0 * inverseFocalAdjustment;

            vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
            gl_Position = quadPos;
            vPosition *= sqrt8;
        `;

        vertexShaderSource += SplatMaterial.getVertexShaderFadeIn();
        vertexShaderSource += `}`;

        return vertexShaderSource;
    }

    static buildFragmentShader() {
        return `
            precision highp float;
            #include <common>
 
            uniform vec3 debugColor;
            uniform int renderMode;
            uniform float ringSize;
            uniform int ditherMode;
            uniform vec2 ditherJitter;
            uniform int toneMapMode;
            uniform int gammaMode;

            varying vec4 vColor;
            varying vec2 vUv;
            varying vec2 vPosition;
            varying vec3 vPickColor;

            // --- FIXED: Helpers moved outside main() ---
            const float EXP4 = 0.01831563888873418;
            const float INV_EXP4 = 1.018657360363774;
            const float MIN_ALPHA = 0.00392156862745098;

            float normExp(float x) {
                return (exp(x * -4.0) - EXP4) * INV_EXP4;
            }

            float ignNoise(vec2 fragCoord, vec2 jitter, float idSeed) {
                vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
                float noise = fract(magic.z * fract(dot(fragCoord + jitter + vec2(idSeed), magic.xy)));
                return pow(noise, 2.2);
            }

            vec3 decodeGamma(vec3 c) {
                return pow(c, vec3(2.2));
            }
            vec3 gammaCorrectOutput(vec3 c) {
                return pow(c + 1e-7, vec3(1.0 / 2.2));
            }
            vec3 toneMapAces(vec3 x) {
                const float a = 2.51;
                const float b = 0.03;
                const float c = 2.43;
                const float d = 0.59;
                const float e = 0.14;
                return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
            }
            vec3 prepareOutputFromGamma(vec3 gammaColor, int gammaMode, int toneMapMode) {
                vec3 lin = (gammaMode == 0) ? decodeGamma(gammaColor) : gammaColor;
                if (toneMapMode == 0) return lin;
                return gammaCorrectOutput(toneMapAces(lin));
            }

            void main () {
                float A8 = dot(vPosition, vPosition);
                float r2 = A8 / 8.0;
                if (r2 > 1.0) discard;

                if (renderMode == 1) {
                    gl_FragColor = vec4(vPickColor, 1.0);
                    return;
                }
                if (renderMode == 2) {
                    float oa = exp(-r2 * 4.0) * vColor.a;
                    gl_FragColor = vec4(1.0, 1.0, 1.0, oa);
                    return;
                }

                float alpha = normExp(r2) * vColor.a;
                if (alpha < MIN_ALPHA) discard;

                if (renderMode == 3 && ringSize > 0.0) {
                    if (r2 < 1.0 - ringSize) {
                        alpha = max(0.05, alpha);
                    } else {
                        alpha = 0.6;
                    }
                }

                if (ditherMode != 0) {
                    float noise = ignNoise(gl_FragCoord.xy, ditherJitter, vPickColor.x * 255.0 * 0.013);
                    if (alpha < noise) discard;
                }

                vec3 color = prepareOutputFromGamma(max(vColor.rgb, 0.0), gammaMode, toneMapMode);
                gl_FragColor = vec4(color * alpha, alpha);
            }
        `;
    }
}