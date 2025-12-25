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
        // Match PlayCanvas/supersplat defaults: TONEMAP=LINEAR, GAMMA=SRGB, exposure=1.
        // These settings output gamma-space color (no implicit sRGB framebuffer conversion).
        uniforms['toneMapMode'] = { 'type': 'i', 'value': 1 };
        uniforms['gammaMode'] = { 'type': 'i', 'value': 1 };
        uniforms['exposure'] = { 'type': 'f', 'value': 1.0 };

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
                sampledCovarianceB = texture(
                    covariancesTexture,
                    getDataUVF(nearestEvenIndex, 1.5, oddOffset + uint(1), covariancesTextureSize)
                );
                cov3D_M11_M12_M13 = vec3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceB.gba) * fOddOffset;
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

            // PlayCanvas gsplatCornerVS parity: use scalar focal in pixels derived from viewport.x and projection[0][0].
            // This intentionally uses the same J1 for both X/Y to match engine behavior.
            float focalPx = viewport.x * projectionMatrix[0][0];
            vec3 v = (orthographicMode == 1) ? vec3(0.0, 0.0, 1.0) : (viewCenter.xyz / viewCenter.w);
            float J1 = focalPx / v.z;
            vec2 J2 = -J1 / v.z * v.xy;
            mat3 J = mat3(
                J1, 0.0, J2.x,
                0.0, J1, J2.y,
                0.0, 0.0, 0.0
            );

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
            `;
        } else {
            vertexShaderSource += `
                cov2Dm[0][0] += ${kernel2DSize};
                cov2Dm[1][1] += ${kernel2DSize};
            `;
        }
        // ---------------------------------------------------

        vertexShaderSource += `
            // Supersplat/PlayCanvas parity: compute screen-space ellipse axes from projected covariance.
            // This avoids subtle scaling differences that show up as blur/softness.
            float diagonal1 = cov2Dm[0][0];
            float offDiagonal = cov2Dm[0][1];
            float diagonal2 = cov2Dm[1][1];

            float mid = 0.5 * (diagonal1 + diagonal2);
            float radius = length(vec2((diagonal1 - diagonal2) / 2.0, offDiagonal));
            float lambda1 = mid + radius;
            float lambda2 = max(mid - radius, 0.1);

            float vmin = min(1024.0, min(viewport.x, viewport.y));

            float l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin);
            float l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin);

            if (pointCloudModeEnabled == 1) {
                l1 = l2 = 2.0;
            }

            // Apply global splatScale the same way as the previous basis-vector approach.
            l1 *= splatScale;
            l2 *= splatScale;

            // Early-out gaussians smaller than 2 pixels.
            if (l1 < 2.0 && l2 < 2.0) {
                gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                return;
            }

            // Frustum culling in clip space (PlayCanvas-style).
            vec2 c = clipCenter.ww / viewport;
            if (any(greaterThan(abs(clipCenter.xy) - vec2(max(l1, l2)) * c, clipCenter.ww))) {
                gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                return;
            }

            vec2 diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1));
            vec2 v1 = l1 * diagonalVector;
            vec2 v2 = l2 * vec2(diagonalVector.y, -diagonalVector.x);
        `;

        if (enableOptionalEffects) {
            vertexShaderSource += `
                vColor.a *= splatOpacityFromScene;
            `;
        }

        vertexShaderSource += `
            // PlayCanvas gsplatCornerVS parity: offset is computed in clip space and added to clipCenter.
            vec2 cornerOffset = (vPosition.x * v1 + vPosition.y * v2) * c;
            gl_Position = clipCenter + vec4(cornerOffset, 0.0, 0.0);
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
            uniform float exposure;

            varying vec4 vColor;
            varying vec2 vUv;
            varying vec2 vPosition;
            varying vec3 vPickColor;
            varying float vDitherId;

            // --- FIXED: Helpers moved outside main() ---
            const float EXP4 = 0.01831563888873418;
            const float INV_EXP4 = 1.018657360363774;
            // supersplat does not discard based on low alpha here.

            float normExp(float x) {
                return (exp(x * -4.0) - EXP4) * INV_EXP4;
            }

            float ignNoise(vec2 fragCoord, vec2 jitter, float idSeed) {
                vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
                float noise = fract(magic.z * fract(dot(fragCoord + jitter + vec2(idSeed), magic.xy)));
                return pow(noise, 2.2);
            }

            // PlayCanvas shader chunk parity
            vec3 decodeGamma(vec3 raw) {
                return pow(raw, vec3(2.2));
            }

            vec3 gammaCorrectOutput(vec3 color, int gammaMode) {
                // gammaMode: 1 = SRGB, 0 = NONE
                return (gammaMode == 1) ? pow(color + 1e-7, vec3(1.0 / 2.2)) : color;
            }

            vec3 toneMapLinear(vec3 color, float exposure) {
                return color * exposure;
            }

            vec3 toneMapAces(vec3 color, float exposure) {
                const float a = 2.51;
                const float b = 0.03;
                const float c = 2.43;
                const float d = 0.59;
                const float e = 0.14;
                vec3 x = color * exposure;
                return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
            }

            vec3 toneMap(vec3 linearColor, int toneMapMode, float exposure) {
                // toneMapMode: 0=NONE, 1=LINEAR, 2=ACES
                if (toneMapMode == 1) return toneMapLinear(linearColor, exposure);
                if (toneMapMode == 2) return toneMapAces(linearColor, exposure);
                return linearColor;
            }

            vec3 prepareOutputFromGamma(vec3 gammaColor, int gammaMode, int toneMapMode, float exposure) {
                // Mirrors PlayCanvas gsplatOutputVS: output is either linear or gamma depending on GAMMA and TONEMAP.
                if (toneMapMode == 0) {
                    // TONEMAP == NONE
                    if (gammaMode == 0) {
                        // GAMMA == NONE -> convert to linear
                        return decodeGamma(gammaColor);
                    }
                    // GAMMA == SRGB -> output gamma color directly
                    return gammaColor;
                }

                // TONEMAP != NONE -> tonemap in linear then output linear or gamma
                return gammaCorrectOutput(toneMap(decodeGamma(gammaColor), toneMapMode, exposure), gammaMode);
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

                // PlayCanvas/supersplat parity: discard below 1/255.
                if (alpha < 1.0 / 255.0) discard;

                if (renderMode == 3 && ringSize > 0.0) {
                    if (r2 < 1.0 - ringSize) {
                        alpha = max(0.05, alpha);
                    } else {
                        alpha = 0.6;
                    }
                }

                if (ditherMode != 0) {
                    float noise = ignNoise(gl_FragCoord.xy, ditherJitter, vDitherId * 0.013);
                    if (alpha < noise) discard;
                }

                vec3 color = prepareOutputFromGamma(max(vColor.rgb, 0.0), gammaMode, toneMapMode, exposure);
                gl_FragColor = vec4(color * alpha, alpha);
            }
        `;
    }
}
