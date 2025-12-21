import * as THREE from 'three';
import { SplatMaterial } from './SplatMaterial.js';

export class SplatMaterial3D {

    /**
     * Build the Three.js material that is used to render the splats.
     * @param {number} dynamicMode If true, it means the scene geometry represented by this splat mesh is not stationary or
     *                             that the splat count might change
     * @param {boolean} enableOptionalEffects When true, allows for usage of extra properties and attributes in the shader for effects
     *                                        such as opacity adjustment. Default is false for performance reasons.
     * @param {boolean} antialiased If true, calculate compensation factor to deal with gaussians being rendered at a significantly
     *                              different resolution than that of their training
     * @param {number} maxScreenSpaceSplatSize The maximum clip space splat size
     * @param {number} splatScale Value by which all splats are scaled in screen-space (default is 1.0)
     * @param {number} pointCloudModeEnabled Render all splats as screen-space circles
     * @param {number} maxSphericalHarmonicsDegree Degree of spherical harmonics to utilize in rendering splats
     * @return {THREE.ShaderMaterial}
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

        uniforms['covariancesTextureSize'] = {
            'type': 'v2',
            'value': new THREE.Vector2(1024, 1024)
        };
        uniforms['covariancesTexture'] = {
            'type': 't',
            'value': null
        };
        uniforms['covariancesTextureHalfFloat'] = {
            'type': 't',
            'value': null
        };
        uniforms['covariancesAreHalfFloat'] = {
            'type': 'i',
            'value': 0
        };

        // --- supersplat parity knobs (no UI here; caller may set uniforms) ---
        // renderMode: 0=forward (default), 1=pick (RGB encodes splat id), 2=outline, 3=rings
        uniforms['renderMode'] = { 'type': 'i', 'value': 0 };
        // ringSize: like supersplat's ringSize (0 disables)
        uniforms['ringSize'] = { 'type': 'f', 'value': 0.0 };
        // ditherMode: 0 disables, 1 enables IGN-noise opacity dither
        uniforms['ditherMode'] = { 'type': 'i', 'value': ditherEnabled ? 1 : 0 };
        // jitter for dither (temporal or per-frame). Default 0.
        uniforms['ditherJitter'] = { 'type': 'v2', 'value': new THREE.Vector2(0, 0) };
        // toneMapMode / gammaMode are a small hook to mirror PlayCanvas' prepareOutputFromGamma.
        // 0 = identity (keep existing Three.js pipeline), 1 = ACES-ish tonemap + sRGB output.
        uniforms['toneMapMode'] = { 'type': 'i', 'value': 0 };
        // 0 = decode input gamma (pow 2.2), 1 = pass-through
        uniforms['gammaMode'] = { 'type': 'i', 'value': 1 };

        const material = new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexShader: vertexShaderSource,
            fragmentShader: fragmentShaderSource,
            transparent: true,
            alphaTest: 1.0,
            // supersplat/PlayCanvas GSplat uses premultiplied alpha output.
            // In Three.js we express that as (ONE, ONE_MINUS_SRC_ALPHA) blending.
            blending: ditherEnabled ? THREE.NoBlending : THREE.CustomBlending,
            blendSrc: THREE.OneFactor,
            blendDst: THREE.OneMinusSrcAlphaFactor,
            blendEquation: THREE.AddEquation,
            depthTest: true,
            // Dithered-opacity mode wants depthWrite so we can avoid sorting at the cost of noise.
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
                sampledCovarianceA = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset,
                                                                            covariancesTextureSize));
                sampledCovarianceB = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset + uint(1),
                                                                            covariancesTextureSize));

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
        
            // Construct the 3D covariance matrix
            mat3 Vrk = mat3(
                cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z,
                cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x, cov3D_M22_M23_M33.y,
                cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z
            );

            mat3 J;
            if (orthographicMode == 1) {
                // Since the projection is linear, we don't need an approximation
                J = transpose(mat3(orthoZoom, 0.0, 0.0,
                                0.0, orthoZoom, 0.0,
                                0.0, 0.0, 0.0));
            } else {
                // Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
                // 3D covariance matrix instead of using the actual projection matrix because that transformation would
                // require a non-linear component (perspective division) which would yield a non-gaussian result.
                float s = 1.0 / (viewCenter.z * viewCenter.z);
                J = mat3(
                    focal.x / viewCenter.z, 0., -(focal.x * viewCenter.x) * s,
                    0., focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s,
                    0., 0., 0.
                );
            }

            // Concatenate the projection approximation with the model-view transformation
            mat3 W = transpose(mat3(transformModelViewMatrix));
            mat3 T = W * J;

            // Transform the 3D covariance matrix (Vrk) to compute the 2D covariance matrix
            mat3 cov2Dm = transpose(T) * Vrk * T;
            `;

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

        vertexShaderSource += `

            // We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because
            // we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0],
            // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
            // need cov2Dm[1][0] because it is a symetric matrix.
            vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

            // We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix
            // so that we can determine the 2D basis for the splat. This is done using the method described
            // here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
            // After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
            // by normalizing the eigen-vectors and then multiplying them by (sqrt(8) * sqrt(eigen-value)), which is
            // equal to scaling them by sqrt(8) standard deviations.
            //
            // This is a different approach than in the original work at INRIA. In that work they compute the
            // max extents of the projected splat in screen space to form a screen-space aligned bounding rectangle
            // which forms the geometry that is actually rasterized. The dimensions of that bounding box are 3.0
            // times the square root of the maximum eigen-value, or 3 standard deviations. They then use the inverse
            // 2D covariance matrix (called 'conic') in the CUDA rendering thread to determine fragment opacity by
            // calculating the full gaussian: exp(-0.5 * (X - mean) * conic * (X - mean)) * splat opacity
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
            // since the eigen vectors are orthogonal, we derive the second one from the first
            vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

            // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
            vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), ${parseInt(maxScreenSpaceSplatSize)}.0);
            vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), ${parseInt(maxScreenSpaceSplatSize)}.0);

            // supersplat/PlayCanvas: early discard very small splats (saves fill, reduces sparkle).
            // The PlayCanvas check is based on eigenvalue-derived extents; here we approximate using basis lengths in pixels.
            if (length(basisVector1) < 2.0 && length(basisVector2) < 2.0) return;
            `;

        if (enableOptionalEffects) {
            vertexShaderSource += `
                vColor.a *= splatOpacityFromScene;
            `;
        }

        vertexShaderSource += `
            // supersplat/PlayCanvas: clipCorner() shrinks the quad to exclude regions where
            // the gaussian would be below 1/255 alpha. This reduces halo and saves fill.
            // We apply the same shrink by scaling the quad corner (vPosition) before projection.
            float clipFactor = 1.0;
            if (vColor.a > 0.0) {
                // clip = min(1, sqrt(-log(1/(255*alpha)))/2)
                clipFactor = min(1.0, sqrt(-log(1.0 / (255.0 * vColor.a))) / 2.0);
            }
            vPosition *= clipFactor;
        `;

        vertexShaderSource += `
            vec2 ndcOffset = vec2(vPosition.x * basisVector1 + vPosition.y * basisVector2) *
                             basisViewport * 2.0 * inverseFocalAdjustment;

            vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
            gl_Position = quadPos;

            // Scale the position data we send to the fragment shader
            vPosition *= sqrt8;
        `;

        vertexShaderSource += SplatMaterial.getVertexShaderFadeIn();
        vertexShaderSource += `}`;

        return vertexShaderSource;
    }

    static buildFragmentShader() {
        let fragmentShaderSource = `
            precision highp float;
            #include <common>
 
            uniform vec3 debugColor;

            // supersplat parity controls (all optional; caller can ignore)
            uniform int renderMode;     // 0=forward, 1=pick, 2=outline, 3=rings
            uniform float ringSize;     // rings thickness (0 disables)
            uniform int ditherMode;     // 0=off, 1=IGN opacity dither
            uniform vec2 ditherJitter;  // jitter for dither (optional)
            uniform int toneMapMode;    // 0=identity, 1=ACES-ish + sRGB
            uniform int gammaMode;      // 0=decodeGamma(pow2.2), 1=pass-through

            varying vec4 vColor;
            varying vec2 vUv;
            varying vec2 vPosition;
            varying vec3 vPickColor;

            // --- helpers (must be outside main() for GLSL ES) ---
            const float EXP4 = 0.01831563888873418; // exp(-4)
            const float INV_EXP4 = 1.018657360363774; // 1/(1-exp(-4))
            const float MIN_ALPHA = 0.00392156862745098; // 1/255

            // supersplat normalizes exp() so alpha hits 0 at quad edge (r2==1).
            float normExp(float x) {
                return (exp(x * -4.0) - EXP4) * INV_EXP4;
            }

            // Opacity dither (IGN). Mirrors supersplat's opacityDither concept.
            float ignNoise(vec2 fragCoord, vec2 jitter, float idSeed) {
                vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
                float noise = fract(magic.z * fract(dot(fragCoord + jitter + vec2(idSeed), magic.xy)));
                return pow(noise, 2.2);
            }

            // prepareOutputFromGamma: minimal hook mirroring PlayCanvas.
            vec3 decodeGamma(vec3 c) {
                return pow(c, vec3(2.2));
            }
            vec3 gammaCorrectOutput(vec3 c) {
                return pow(c + 1e-7, vec3(1.0 / 2.2));
            }
            vec3 toneMapAces(vec3 x) {
                // A small ACES approximation (good enough for parity testing).
                const float a = 2.51;
                const float b = 0.03;
                const float c = 2.43;
                const float d = 0.59;
                const float e = 0.14;
                return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
            }
            vec3 prepareOutputFromGamma(vec3 gammaColor, int gammaMode, int toneMapMode) {
                vec3 lin = (gammaMode == 0) ? decodeGamma(gammaColor) : gammaColor;
                if (toneMapMode == 0) {
                    return lin;
                }
                return gammaCorrectOutput(toneMapAces(lin));
            }
        `;

        fragmentShaderSource += `
            void main () {
                // --- supersplat/PlayCanvas gsplat parity ---
                // NOTE: vPosition has been scaled by sqrt(8) in the vertex shader.
                // Let r2 = dot(uv,uv) in [-1,1]^2 space. Then dot(vPosition,vPosition) = 8*r2.
                float A8 = dot(vPosition, vPosition);
                float r2 = A8 / 8.0;
                if (r2 > 1.0) discard;

                // Pick pass: output encoded splat id (24-bit RGB).
                if (renderMode == 1) {
                    gl_FragColor = vec4(vPickColor, 1.0);
                    return;
                }

                // Outline pass: mimic supersplat outline alpha curve.
                if (renderMode == 2) {
                    float oa = exp(-r2 * 4.0) * vColor.a;
                    gl_FragColor = vec4(1.0, 1.0, 1.0, oa);
                    return;
                }

                float alpha = normExp(r2) * vColor.a;
                if (alpha < MIN_ALPHA) discard;

                // Rings mode: replicate supersplat's debug rings (alpha remap).
                if (renderMode == 3 && ringSize > 0.0) {
                    if (r2 < 1.0 - ringSize) {
                        alpha = max(0.05, alpha);
                    } else {
                        alpha = 0.6;
                    }
                }

                // Opacity dither: supersplat uses opacityDither(alpha, id*0.013) with IGN noise option.
                // We implement IGN (no texture) for portability.
                if (ditherMode != 0) {
                    float noise = ignNoise(gl_FragCoord.xy, ditherJitter, vPickColor.x * 255.0 * 0.013);
                    if (alpha < noise) discard;
                }

                // Default keeps legacy Three.js output pipeline unless toneMapMode != 0.
                vec3 color = prepareOutputFromGamma(max(vColor.rgb, 0.0), gammaMode, toneMapMode);

                // Premultiplied output (supersplat/PlayCanvas GSplat).
                gl_FragColor = vec4(color * alpha, alpha);
            }
        `;

        return fragmentShaderSource;
    }

}
