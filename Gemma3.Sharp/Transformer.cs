using System;
using System.Runtime.InteropServices;
using System.Numerics;

namespace Gemma3.Sharp
{
    public struct Gemma3Config
    {
        public int VocabSize;
        public int HiddenSize;
        public int IntermediateSize;
        public int NumLayers;
        public int NumHeads;
        public int NumKVHeads;
        public int HeadDim;
        public int MaxContext;
        public int SlidingWindow;
        public float RmsNormEps;
        public float RopeThetaLocal;
        public float RopeThetaGlobal;

        public static Gemma3Config Default => new Gemma3Config
        {
            VocabSize = 262208,
            HiddenSize = 2560,
            IntermediateSize = 10240,
            NumLayers = 34,
            NumHeads = 8,
            NumKVHeads = 4,
            HeadDim = 256,
            MaxContext = 131072, // 128K
            SlidingWindow = 1024,
            RmsNormEps = 1e-6f,
            RopeThetaLocal = 10000.0f,
            RopeThetaGlobal = 1000000.0f
        };

        public static bool IsGlobalLayer(int layerIdx)
        {
            // Global every 6th layer: layers 5, 11, 17, 23, 29 (0-indexed)
            return ((layerIdx + 1) % 6 == 0);
        }
    }

    public unsafe class KVCache : IDisposable
    {
        public struct LayerCache
        {
            public float* K;
            public float* V;
            public int Pos;
            public int Size; // Max seq len for this layer
        }

        public LayerCache[]? Layers;
        public int CurrentPos;
        private int _numLayers;

        public KVCache(Gemma3Config cfg, int maxSeq)
        {
            _numLayers = cfg.NumLayers;
            Layers = new LayerCache[_numLayers];
            int kvSize = cfg.NumKVHeads * cfg.HeadDim;

            for (int l = 0; l < _numLayers; l++)
            {
                int layerMaxSeq = Gemma3Config.IsGlobalLayer(l) ? maxSeq : cfg.SlidingWindow;

                // Allocate unmanaged memory (Aligned to 64 bytes)
                Layers[l].K = (float*)NativeMemory.AlignedAlloc((nuint)(layerMaxSeq * kvSize * sizeof(float)), 64);
                Layers[l].V = (float*)NativeMemory.AlignedAlloc((nuint)(layerMaxSeq * kvSize * sizeof(float)), 64);

                // Zero initialize
                NativeMemory.Clear(Layers[l].K, (nuint)(layerMaxSeq * kvSize * sizeof(float)));
                NativeMemory.Clear(Layers[l].V, (nuint)(layerMaxSeq * kvSize * sizeof(float)));

                Layers[l].Size = layerMaxSeq;
                Layers[l].Pos = 0;
            }
        }

        public void Reset()
        {
            CurrentPos = 0;
            for (int l = 0; l < _numLayers; l++)
            {
                Layers[l].Pos = 0;
            }
        }

        public void Dispose()
        {
            if (Layers != null)
            {
                for (int l = 0; l < _numLayers; l++)
                {
                    if (Layers[l].K != null) NativeMemory.AlignedFree(Layers[l].K);
                    if (Layers[l].V != null) NativeMemory.AlignedFree(Layers[l].V);
                }
                Layers = null;
            }
        }
    }

    public unsafe class ActivationBuffers : IDisposable
    {
        public float* X;
        public float* XNorm;
        public float* Q;
        public float* K;
        public float* V;
        public float* AttnOut;
        public float* ProjOut;
        public float* MlpGate;
        public float* MlpUp;
        public float* MlpOut;
        public float* Logits;
        public float* Mask;

        public ActivationBuffers(Gemma3Config cfg)
        {
            X = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.HiddenSize * sizeof(float)), 64);
            XNorm = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.HiddenSize * sizeof(float)), 64);
            Q = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.NumHeads * cfg.HeadDim * sizeof(float)), 64);
            K = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.NumKVHeads * cfg.HeadDim * sizeof(float)), 64);
            V = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.NumKVHeads * cfg.HeadDim * sizeof(float)), 64);
            AttnOut = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.NumHeads * cfg.HeadDim * sizeof(float)), 64);
            ProjOut = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.HiddenSize * sizeof(float)), 64);
            MlpGate = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.IntermediateSize * sizeof(float)), 64);
            MlpUp = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.IntermediateSize * sizeof(float)), 64);
            MlpOut = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.HiddenSize * sizeof(float)), 64);
            Logits = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.VocabSize * sizeof(float)), 64);
            Mask = (float*)NativeMemory.AlignedAlloc((nuint)(cfg.MaxContext * sizeof(float)), 64);
        }

        public void Dispose()
        {
             if(X != null) NativeMemory.AlignedFree(X);
             if(XNorm != null) NativeMemory.AlignedFree(XNorm);
             if(Q != null) NativeMemory.AlignedFree(Q);
             if(K != null) NativeMemory.AlignedFree(K);
             if(V != null) NativeMemory.AlignedFree(V);
             if(AttnOut != null) NativeMemory.AlignedFree(AttnOut);
             if(ProjOut != null) NativeMemory.AlignedFree(ProjOut);
             if(MlpGate != null) NativeMemory.AlignedFree(MlpGate);
             if(MlpUp != null) NativeMemory.AlignedFree(MlpUp);
             if(MlpOut != null) NativeMemory.AlignedFree(MlpOut);
             if(Logits != null) NativeMemory.AlignedFree(Logits);
             if(Mask != null) NativeMemory.AlignedFree(Mask);
        }
    }

    public unsafe class Gemma3Transformer : IDisposable
    {
        public Gemma3Config Config { get; }
        public Gemma3Weights Weights { get; }
        public KVCache Cache { get; }
        public ActivationBuffers Buffers { get; }

        public Gemma3Transformer(Gemma3Weights weights, Gemma3Config config, int maxContext = 8192)
        {
            Weights = weights;
            Config = config;
            Cache = new KVCache(config, maxContext);
            Buffers = new ActivationBuffers(config);
        }

        public void Forward(float* logits, int tokenId, int pos)
        {
            int hiddenSize = Config.HiddenSize;
            int vocabSize = Config.VocabSize;

            // Embedding
            // Note: Weights are ReadOnlySpan<ushort>. We need pointer.
            // But Span cannot be converted to pointer easily unless fixed.
            // But the Weights class uses MappedFile pointers internally but exposes Spans.
            // We should modify Weights to expose pointers or use fixed on spans (which works if they are backed by unmanaged memory, but Spans from MappedViewAccessor are).
            // Actually, the easiest way is to modify Kernels to accept ReadOnlySpan or just unsafe pointers.
            // But Kernels use pointers.
            // So we need to get pointers from Weights.

            // Let's assume we can get pointers. I'll use fixed statement on Spans.
            // Wait, you can only use fixed on managed arrays or strings or fixed size buffers, OR on spans that refer to pinned memory.
            // The spans from SafeTensors are backed by unmanaged memory (pointers).
            // So `fixed` might not work directly or it might simply work if I use `GetPinnableReference`?
            // Actually `fixed (ushort* ptr = span)` works for unmanaged-backed spans too in recent C#.

            // Embed
            fixed (ushort* wEmbed = Weights.EmbedTokens)
            {
                // We implement specialized kernel for embedding row copy
                ushort* row = wEmbed + tokenId * hiddenSize;
                float* x = Buffers.X;
                for(int i=0; i<hiddenSize; i++) x[i] = Kernels.BF16ToFloat(row[i]);
            }

            // Scale
            float scale = MathF.Sqrt(hiddenSize);
            for(int i=0; i<hiddenSize; i++) Buffers.X[i] *= scale;

            // Layers
            for(int l=0; l<Config.NumLayers; l++)
            {
                var layer = Weights.Layers[l];

                // Pre-Attn RMSNorm
                fixed (ushort* wInputLn = layer.InputLayernorm)
                {
                    Kernels.RMSNorm(Buffers.XNorm, Buffers.X, wInputLn, hiddenSize, Config.RmsNormEps);
                }

                // Attention
                fixed (ushort* wQ = layer.QProj)
                fixed (ushort* wK = layer.KProj)
                fixed (ushort* wV = layer.VProj)
                fixed (ushort* wO = layer.OProj)
                fixed (ushort* wQNorm = layer.QNorm)
                fixed (ushort* wKNorm = layer.KNorm)
                {
                    LayerAttention(l, pos, wQ, wK, wV, wO, wQNorm, wKNorm);
                }

                // Post-Attn Norm
                fixed (ushort* wPostAttnLn = layer.PostAttentionLayernorm)
                {
                     // In-place RMSNorm
                     // Kernels.RMSNorm(Buffers.ProjOut, Buffers.ProjOut, wPostAttnLn, ...)
                     // Wait, C code does inplace on proj_out.
                     // But we have ProjOut separately.
                     // C: gemma3_rmsnorm_bf16_inplace(buf->proj_out, ...)

                     // We can implement inplace RMSNorm in Kernels or just use separate buffer.
                     // Let's just use XNorm buffer as temp if needed, or implement inplace.
                     // I'll implement explicit inplace call here:
                     float* ptr = Buffers.ProjOut;
                     // Manual inline or helper
                     Kernels.RMSNorm(ptr, ptr, wPostAttnLn, hiddenSize, Config.RmsNormEps);
                }

                // Residual X += ProjOut
                for(int i=0; i<hiddenSize; i++) Buffers.X[i] += Buffers.ProjOut[i];

                // MLP Block
                // Pre-FF Norm
                fixed (ushort* wPreFfLn = layer.PreFeedforwardLayernorm)
                {
                     Kernels.RMSNorm(Buffers.XNorm, Buffers.X, wPreFfLn, hiddenSize, Config.RmsNormEps);
                }

                fixed (ushort* wGate = layer.GateProj)
                fixed (ushort* wUp = layer.UpProj)
                fixed (ushort* wDown = layer.DownProj)
                {
                    LayerMlp(l, wGate, wUp, wDown);
                }

                // Post-FF Norm
                fixed (ushort* wPostFfLn = layer.PostFeedforwardLayernorm)
                {
                    Kernels.RMSNorm(Buffers.MlpOut, Buffers.MlpOut, wPostFfLn, hiddenSize, Config.RmsNormEps);
                }

                // Residual X += MlpOut
                for(int i=0; i<hiddenSize; i++) Buffers.X[i] += Buffers.MlpOut[i];
            }

            // Final Norm
            fixed (ushort* wNorm = Weights.Norm)
            {
                Kernels.RMSNorm(Buffers.XNorm, Buffers.X, wNorm, hiddenSize, Config.RmsNormEps);
            }

            // Logits
            fixed (ushort* wEmbed = Weights.EmbedTokens)
            {
                Kernels.MatVec(logits, wEmbed, Buffers.XNorm, vocabSize, hiddenSize);
            }
        }

        private void LayerAttention(int l, int pos, ushort* wQ, ushort* wK, ushort* wV, ushort* wO, ushort* wQNorm, ushort* wKNorm)
        {
             int hiddenSize = Config.HiddenSize;
             int numHeads = Config.NumHeads;
             int numKVHeads = Config.NumKVHeads;
             int headDim = Config.HeadDim;
             int qSize = numHeads * headDim;
             int kvSize = numKVHeads * headDim;

             // Projections
             Kernels.MatVec(Buffers.Q, wQ, Buffers.XNorm, qSize, hiddenSize);
             Kernels.MatVec(Buffers.K, wK, Buffers.XNorm, kvSize, hiddenSize);
             Kernels.MatVec(Buffers.V, wV, Buffers.XNorm, kvSize, hiddenSize);

             // QK Norm
             for(int h=0; h<numHeads; h++)
                 Kernels.RMSNorm(Buffers.Q + h*headDim, Buffers.Q + h*headDim, wQNorm, headDim, Config.RmsNormEps);
             for(int h=0; h<numKVHeads; h++)
                 Kernels.RMSNorm(Buffers.K + h*headDim, Buffers.K + h*headDim, wKNorm, headDim, Config.RmsNormEps);

             // RoPE
             bool isGlobal = Gemma3Config.IsGlobalLayer(l);
             float theta = isGlobal ? Config.RopeThetaGlobal : Config.RopeThetaLocal;
             Kernels.RoPE(Buffers.Q, Buffers.K, numHeads, numKVHeads, headDim, pos, theta);

             // Cache KV
             CacheKV(l, Buffers.K, Buffers.V, kvSize, isGlobal, pos);

             // Attention
             float* kCache = Cache.Layers[l].K;
             float* vCache = Cache.Layers[l].V;

             // Determine seq len and mask
             int seqLen;
             if (isGlobal)
             {
                 seqLen = pos + 1;
                 // Causal mask
                 for(int i=0; i<seqLen; i++) Buffers.Mask[i] = 0.0f; // Simplified causal mask (all prev tokens valid)
                 // Note: Actual causal mask requires -inf for future tokens if we compute for multiple tokens.
                 // But here we compute for 1 token 'pos'. It attends to 0..pos.
                 // So the mask is all zeros for 0..pos.
             }
             else
             {
                 int window = Config.SlidingWindow;
                 seqLen = (pos < window) ? pos + 1 : window;
                 // Sliding window mask: all valid in window
                 for(int i=0; i<seqLen; i++) Buffers.Mask[i] = 0.0f;
             }

             // GQA
             float scale = 1.0f / MathF.Sqrt(headDim);

             // We need to implement GQA in Kernels or here.
             // C implementation has `gemma3_gqa`.
             // I'll implement it inline here using loops and helper from Kernels?
             // Or better, add GQA to Kernels.
             // For now, I'll implement it here using loop.

             int headsPerGroup = numHeads / numKVHeads;

             for(int h=0; h<numHeads; h++)
             {
                 int kvHead = h / headsPerGroup;
                 float* qHead = Buffers.Q + h * headDim;
                 float* outHead = Buffers.AttnOut + h * headDim;

                 // Scores
                 // kCache is [seqLen, kvHeads, headDim] (interleaved)
                 // Or [layer][pos * kvSize + kvHead * headDim] ?
                 // My CacheKV impl puts it linearly:
                 // cache_pos * kvSize + ...
                 // So stride between positions is kvSize.

                 for(int t=0; t<seqLen; t++)
                 {
                     float* kPos = kCache + t * kvSize + kvHead * headDim;

                     // Dot product qHead . kPos
                     // Simple float dot
                     float score = 0.0f;
                     for(int d=0; d<headDim; d++) score += qHead[d] * kPos[d];
                     score *= scale;

                     // Mask (always 0 here for valid positions)
                     Buffers.Mask[t] += score; // reusing mask buffer for scores? No, separate buffer needed.
                     // The C code reuses mask buffer or allocs scores buffer.
                     // "float *scores = (float *)malloc(seq_len * sizeof(float));"
                     // I should use a temp buffer. I have Buffers.Mask but that's for mask values.
                     // I can use Buffers.Mask as temp score buffer since I just computed it to be 0.
                     // So Buffers.Mask[t] = score.

                     Buffers.Mask[t] = score;
                 }

                 // Softmax on Mask (which now holds scores)
                 Kernels.Softmax(Buffers.Mask, seqLen);

                 // Weighted sum
                 for(int d=0; d<headDim; d++) outHead[d] = 0.0f;

                 for(int t=0; t<seqLen; t++)
                 {
                     float w = Buffers.Mask[t];
                     float* vPos = vCache + t * kvSize + kvHead * headDim;
                     for(int d=0; d<headDim; d++) outHead[d] += w * vPos[d];
                 }
             }

             // Output Proj
             Kernels.MatVec(Buffers.ProjOut, wO, Buffers.AttnOut, hiddenSize, qSize);
        }

        private void CacheKV(int l, float* k, float* v, int kvSize, bool isGlobal, int pos)
        {
            int cachePos = isGlobal ? pos : pos % Config.SlidingWindow;

            float* kDst = Cache.Layers[l].K + cachePos * kvSize;
            float* vDst = Cache.Layers[l].V + cachePos * kvSize;

            Buffer.MemoryCopy(k, kDst, kvSize * sizeof(float), kvSize * sizeof(float));
            Buffer.MemoryCopy(v, vDst, kvSize * sizeof(float), kvSize * sizeof(float));

            Cache.Layers[l].Pos = pos + 1;
        }

        private void LayerMlp(int l, ushort* wGate, ushort* wUp, ushort* wDown)
        {
            int hiddenSize = Config.HiddenSize;
            int intermediateSize = Config.IntermediateSize;

            Kernels.MatVec(Buffers.MlpGate, wGate, Buffers.XNorm, intermediateSize, hiddenSize);
            Kernels.MatVec(Buffers.MlpUp, wUp, Buffers.XNorm, intermediateSize, hiddenSize);

            // SwiGLU: GELU(gate) * up
            Kernels.GELU(Buffers.MlpGate, intermediateSize);
            for(int i=0; i<intermediateSize; i++) Buffers.MlpGate[i] *= Buffers.MlpUp[i];

            Kernels.MatVec(Buffers.MlpOut, wDown, Buffers.MlpGate, hiddenSize, intermediateSize);
        }

        public void Dispose()
        {
             Cache.Dispose();
             Buffers.Dispose();
        }
    }
}
