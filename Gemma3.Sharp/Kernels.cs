using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Numerics;

namespace Gemma3.Sharp
{
    public unsafe static class Kernels
    {
        // Helper: BF16 to Float
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float BF16ToFloat(ushort bf16)
        {
            uint bits = (uint)bf16 << 16;
            return BitConverter.UInt32BitsToSingle(bits);
        }

        // MatVec: y = A * x
        // A is [M, K] in BF16
        // x is [K] in Float32
        // y is [M] in Float32
        public static void MatVec(float* y, ushort* A, float* x, int M, int K)
        {
            // Parallelize over M (rows)
            // For simplicity in this port, we'll use a simple loop, but in production Parallel.For is better.
            // But let's stick to single threaded first to match C loop structure,
            // the C code doesn't use OpenMP. It relies on the caller or just runs single core?
            // "make fast -march=native". The C code loop is serial.

            for (int i = 0; i < M; i++)
            {
                y[i] = DotProductBF16(A + i * K, x, K);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float DotProductBF16(ushort* a, float* b, int n)
        {
            float sum = 0.0f;
            int i = 0;

            if (Avx2.IsSupported)
            {
                Vector256<float> sumVec = Vector256<float>.Zero;

                // Process 16 elements at a time (256 bits of BF16 -> 512 bits of Float)
                // We need 2 float vectors for 16 BF16s
                for (; i <= n - 16; i += 16)
                {
                    // Load 16 BF16 values (unaligned load)
                    Vector256<ushort> va_bf16 = Vector256.Load((ushort*)(a + i));

                    // Unpack to 32-bit integers (shift left 16)
                    // First we need to split into two 128-bit lanes or widen.
                    // Avx2.ConvertToVector256Int32 extends 8 ushorts to 8 ints? No, usually 8 shorts/ushorts.
                    // Vector256<ushort> has 16 elements.
                    // We need to split into low 8 and high 8.

                    Vector128<ushort> va_low = va_bf16.GetLower();
                    Vector128<ushort> va_high = va_bf16.GetUpper();

                    // Extend to uint (zerofill) then shift left 16
                    Vector256<uint> va_low_32 = Avx2.ConvertToVector256Int32(va_low).AsUInt32();
                    Vector256<uint> va_high_32 = Avx2.ConvertToVector256Int32(va_high).AsUInt32();

                    va_low_32 = Avx2.ShiftLeftLogical(va_low_32, 16);
                    va_high_32 = Avx2.ShiftLeftLogical(va_high_32, 16);

                    // Cast to float
                    Vector256<float> vf_low = va_low_32.AsSingle();
                    Vector256<float> vf_high = va_high_32.AsSingle();

                    // Load x
                    Vector256<float> vb_low = Vector256.Load(b + i);
                    Vector256<float> vb_high = Vector256.Load(b + i + 8);

                    // FMA
                    sumVec = Fma.MultiplyAdd(vf_low, vb_low, sumVec);
                    sumVec = Fma.MultiplyAdd(vf_high, vb_high, sumVec);
                }

                // Sum the vector elements
                // There isn't a single horizontal sum for AVX2 easily, so we can do dot product style reduction
                // or just store and sum.
                // Or better: Vector256.Sum (Available in .NET 7+)
                sum = Vector256.Sum(sumVec);
            }

            // Scalar cleanup
            for (; i < n; i++)
            {
                sum += BF16ToFloat(a[i]) * b[i];
            }

            return sum;
        }

        public static void RMSNorm(float* y, float* x, ushort* w, int n, float eps)
        {
            float ss = 0.0f;
            for (int i = 0; i < n; i++)
            {
                ss += x[i] * x[i];
            }
            ss = ss / n + eps;
            float rsqrt_ss = 1.0f / MathF.Sqrt(ss);

            for (int i = 0; i < n; i++)
            {
                // Gemma uses (1.0 + weight)
                float weight = 1.0f + BF16ToFloat(w[i]);
                y[i] = x[i] * rsqrt_ss * weight;
            }
        }

        public static void Softmax(float* x, int n)
        {
            float maxVal = x[0];
            for (int i = 1; i < n; i++)
            {
                if (x[i] > maxVal) maxVal = x[i];
            }

            float sum = 0.0f;
            for (int i = 0; i < n; i++)
            {
                x[i] = MathF.Exp(x[i] - maxVal);
                sum += x[i];
            }

            float invSum = 1.0f / sum;
            for (int i = 0; i < n; i++)
            {
                x[i] *= invSum;
            }
        }

        public static void SiLU(float* x, int n)
        {
            for (int i = 0; i < n; i++)
            {
                float val = x[i];
                x[i] = val / (1.0f + MathF.Exp(-val));
            }
        }

        public static void GELU(float* x, int n)
        {
            const float sqrt_2_over_pi = 0.7978845608028654f;
            const float coeff = 0.044715f;

            for (int i = 0; i < n; i++)
            {
                float xi = x[i];
                float x3 = xi * xi * xi;
                float inner = sqrt_2_over_pi * (xi + coeff * x3);
                x[i] = 0.5f * xi * (1.0f + MathF.Tanh(inner));
            }
        }

        public static void RoPE(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos, float theta)
        {
            for (int h = 0; h < n_heads; h++)
            {
                ApplyRotaryEmbedding(q + h * head_dim, head_dim, pos, theta);
            }
            for (int h = 0; h < n_kv_heads; h++)
            {
                ApplyRotaryEmbedding(k + h * head_dim, head_dim, pos, theta);
            }
        }

        private static void ApplyRotaryEmbedding(float* x, int head_dim, int pos, float theta)
        {
            int half_dim = head_dim / 2;
            for (int i = 0; i < half_dim; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, (float)(2 * i) / head_dim);
                float angle = pos * freq;
                float cos = MathF.Cos(angle);
                float sin = MathF.Sin(angle);

                float x0 = x[i];
                float x1 = x[i + half_dim];
                x[i] = x0 * cos - x1 * sin;
                x[i + half_dim] = x0 * sin + x1 * cos;
            }
        }

        // Sampling helpers
        public static void SampleTopK(float* logits, int vocabSize, int k)
        {
             if (k <= 0 || k >= vocabSize) return;

             // We need to find the k-th largest element.
             // For simplicity, we can do a partial sort or just sort indices.
             // Since vocab is large (256k), full sort is slow.
             // But K is usually small (50).
             // We can scan and keep top K.

             // Quick implementation: just scan linear or use a heap if strict performance needed.
             // Or C code uses qsort on an indexed array.

             var indexed = new (float Val, int Idx)[vocabSize];
             for(int i=0; i<vocabSize; i++) indexed[i] = (logits[i], i);

             Array.Sort(indexed, (a, b) => b.Val.CompareTo(a.Val)); // Descending

             float threshold = indexed[k-1].Val;
             for(int i=0; i<vocabSize; i++)
             {
                 if (logits[i] < threshold) logits[i] = float.NegativeInfinity;
             }
        }

        public static void SampleTopP(float* logits, int vocabSize, float p)
        {
            // Softmax first? No, caller usually does softmax before TopP but AFTER TopK?
            // C code:
            // gemma3_topp_filter:
            //   Softmax(probs, logits)
            //   Sort probs
            //   Cumsum
            //   Set logits to -inf

            var probs = new float[vocabSize];
            fixed(float* pProbs = probs)
            {
                // Copy logits
                Buffer.MemoryCopy(logits, pProbs, vocabSize * sizeof(float), vocabSize * sizeof(float));
                Softmax(pProbs, vocabSize);

                var indexed = new (float Prob, int Idx)[vocabSize];
                for(int i=0; i<vocabSize; i++) indexed[i] = (probs[i], i);

                Array.Sort(indexed, (a, b) => b.Prob.CompareTo(a.Prob));

                float cumsum = 0.0f;
                int cutoff = vocabSize;
                for(int i=0; i<vocabSize; i++)
                {
                    cumsum += indexed[i].Prob;
                    if (cumsum > p)
                    {
                        cutoff = i + 1;
                        break;
                    }
                }

                // Create a keep set
                var keep = new bool[vocabSize];
                for(int i=0; i<cutoff; i++) keep[indexed[i].Idx] = true;

                for(int i=0; i<vocabSize; i++)
                {
                    if (!keep[i]) logits[i] = float.NegativeInfinity;
                }
            }
        }

        public static int Sample(float* probs, int vocabSize)
        {
            float r = Random.Shared.NextSingle();
            float cumsum = 0.0f;
            for(int i=0; i<vocabSize; i++)
            {
                cumsum += probs[i];
                if (r < cumsum) return i;
            }
            return vocabSize - 1;
        }

        public static int ArgMax(float* x, int n)
        {
            int maxIdx = 0;
            float maxVal = x[0];
            for (int i = 1; i < n; i++)
            {
                if (x[i] > maxVal)
                {
                    maxVal = x[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }
    }
}
