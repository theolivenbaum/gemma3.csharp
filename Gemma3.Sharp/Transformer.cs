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

            // Embed
            fixed (ushort* wEmbed = Weights.EmbedTokens)
            {
                ushort* row = wEmbed + tokenId * hiddenSize;
                float* x = Buffers.X;
                for(int i=0; i<hiddenSize; i++) x[i] = Kernels.BF16ToFloat(row[i]);
            }

            // Scale
            float scale = MathF.Sqrt(hiddenSize);
            Kernels.VecScale(Buffers.X, Buffers.X, scale, hiddenSize);

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

                // Post-Attn Norm (In-place on ProjOut)
                fixed (ushort* wPostAttnLn = layer.PostAttentionLayernorm)
                {
                     Kernels.RMSNorm(Buffers.ProjOut, Buffers.ProjOut, wPostAttnLn, hiddenSize, Config.RmsNormEps);
                }

                // Residual X += ProjOut
                Kernels.VecAdd(Buffers.X, Buffers.X, Buffers.ProjOut, hiddenSize);

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

                // Post-FF Norm (In-place on MlpOut)
                fixed (ushort* wPostFfLn = layer.PostFeedforwardLayernorm)
                {
                    Kernels.RMSNorm(Buffers.MlpOut, Buffers.MlpOut, wPostFfLn, hiddenSize, Config.RmsNormEps);
                }

                // Residual X += MlpOut
                Kernels.VecAdd(Buffers.X, Buffers.X, Buffers.MlpOut, hiddenSize);
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

        public void Prefill(float* logits, int[] tokens, int startPos)
        {
            // Process tokens sequentially for now
            // We use Buffers.Logits as temp space if needed, but for the last token we output to logits ptr.
            // Actually, we can just call Forward for each token.
            // If caller provides 'logits', they usually want logits for the *last* token only (next token prediction).
            // But we can optimize to not compute full logits for intermediate tokens if we want,
            // but Forward currently always computes them.
            // To optimize, Forward should take a flag or we should have a separate method.
            // The C implementation computes logits for intermediate tokens into buf->logits (temp) but ignores them.

            for (int i = 0; i < tokens.Length; i++)
            {
                int pos = startPos + i;
                bool isLast = (i == tokens.Length - 1);

                if (isLast)
                {
                    Forward(logits, tokens[i], pos);
                }
                else
                {
                    // Compute into temp buffer (Buffers.Logits) to avoid writing to output logits if it points to same place
                    // But Forward writes to 'logits' arg.
                    // We can reuse Buffers.Logits as scratch.
                    Forward(Buffers.Logits, tokens[i], pos);
                }
            }
            Cache.CurrentPos = startPos + tokens.Length;
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
                 // Causal mask: 0 for all valid positions
                 Kernels.VecZero(Buffers.Mask, seqLen);
             }
             else
             {
                 int window = Config.SlidingWindow;
                 seqLen = (pos < window) ? pos + 1 : window;
                 // Sliding window mask: 0 for all valid positions
                 Kernels.VecZero(Buffers.Mask, seqLen);
             }

             // GQA
             float scale = 1.0f / MathF.Sqrt(headDim);

             Kernels.GQA(
                 Buffers.AttnOut,
                 Buffers.Q,
                 kCache,
                 vCache,
                 Buffers.Mask, // scores_buf. Scratch space for scores.
                               // Note: GQA takes 'scores_buf' as scratch, and 'mask' as optional input.
                               // Since for single token generation, the mask is effectively all zeros (all context is valid).
                               // So we can pass mask=null to GQA.
                               // And use Buffers.Mask as 'scores_buf'.
                 numHeads,
                 numKVHeads,
                 headDim,
                 seqLen,
                 scale,
                 null // mask is all zeros, so we can pass null to optimize
             );

             // Output Proj
             Kernels.MatVec(Buffers.ProjOut, wO, Buffers.AttnOut, hiddenSize, qSize);
        }

        private void CacheKV(int l, float* k, float* v, int kvSize, bool isGlobal, int pos)
        {
            int cachePos = isGlobal ? pos : pos % Config.SlidingWindow;

            float* kDst = Cache.Layers[l].K + cachePos * kvSize;
            float* vDst = Cache.Layers[l].V + cachePos * kvSize;

            Kernels.VecCopy(kDst, k, kvSize);
            Kernels.VecCopy(vDst, v, kvSize);

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
            Kernels.VecMul(Buffers.MlpGate, Buffers.MlpGate, Buffers.MlpUp, intermediateSize);

            Kernels.MatVec(Buffers.MlpOut, wDown, Buffers.MlpGate, hiddenSize, intermediateSize);
        }

        public void Dispose()
        {
             Cache.Dispose();
             Buffers.Dispose();
        }
    }
}
