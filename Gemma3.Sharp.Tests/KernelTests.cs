using System;
using Xunit;
using Gemma3.Sharp;

namespace Gemma3.Sharp.Tests
{
    public unsafe class KernelTests
    {
        private ushort FloatToBF16(float f)
        {
            uint bits = BitConverter.SingleToUInt32Bits(f);
            return (ushort)(bits >> 16);
        }

        [Fact]
        public void TestBF16Conversion()
        {
            float f = 1.0f;
            ushort bf16 = FloatToBF16(f);
            float f2 = Kernels.BF16ToFloat(bf16);
            Assert.Equal(f, f2);

            f = -2.5f;
            bf16 = FloatToBF16(f);
            f2 = Kernels.BF16ToFloat(bf16);
            Assert.Equal(f, f2);
        }

        [Fact]
        public void TestMatVec()
        {
            int M = 2;
            int K = 4;

            // A = [[1, 2, 3, 4], [5, 6, 7, 8]]
            ushort[] A = new ushort[M * K];
            A[0] = FloatToBF16(1.0f); A[1] = FloatToBF16(2.0f); A[2] = FloatToBF16(3.0f); A[3] = FloatToBF16(4.0f);
            A[4] = FloatToBF16(5.0f); A[5] = FloatToBF16(6.0f); A[6] = FloatToBF16(7.0f); A[7] = FloatToBF16(8.0f);

            // x = [0.5, 0.5, 0.5, 0.5]
            float[] x = new float[] { 0.5f, 0.5f, 0.5f, 0.5f };
            float[] y = new float[M];

            fixed (ushort* pA = A)
            fixed (float* px = x)
            fixed (float* py = y)
            {
                Kernels.MatVec(py, pA, px, M, K);
            }

            // y[0] = 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 = 0.5 + 1 + 1.5 + 2 = 5.0
            // y[1] = 5*0.5 + 6*0.5 + 7*0.5 + 8*0.5 = 2.5 + 3 + 3.5 + 4 = 13.0

            Assert.Equal(5.0f, y[0], 0.01f);
            Assert.Equal(13.0f, y[1], 0.01f);
        }

        [Fact]
        public void TestMatVecLarge()
        {
            // Test AVX path with > 16 elements
            int M = 1;
            int K = 32;
            ushort[] A = new ushort[K];
            float[] x = new float[K];

            float expectedSum = 0;
            for(int i=0; i<K; i++)
            {
                A[i] = FloatToBF16(1.0f);
                x[i] = 1.0f;
                expectedSum += 1.0f;
            }

            float[] y = new float[M];
            fixed (ushort* pA = A)
            fixed (float* px = x)
            fixed (float* py = y)
            {
                Kernels.MatVec(py, pA, px, M, K);
            }

            Assert.Equal(expectedSum, y[0], 0.01f);
        }

        [Fact]
        public void TestRMSNorm()
        {
            int n = 4;
            float[] x = { 1.0f, 2.0f, 3.0f, 4.0f };
            // Weights are 0 (so 1.0 + 0 = 1.0 scaling)
            ushort[] w = new ushort[n];
            // 0.0f in BF16 is 0

            float[] y = new float[n];

            fixed (float* px = x)
            fixed (ushort* pw = w)
            fixed (float* py = y)
            {
                Kernels.RMSNorm(py, px, pw, n, 1e-6f);
            }

            // SS = (1+4+9+16)/4 = 30/4 = 7.5
            // RMS = sqrt(7.5) = 2.7386
            // y = x / 2.7386

            float rms = MathF.Sqrt(7.5f);
            Assert.Equal(1.0f / rms, y[0], 0.01f);
            Assert.Equal(4.0f / rms, y[3], 0.01f);
        }

        [Fact]
        public void TestSoftmax()
        {
            float[] x = { 1.0f, 2.0f, 3.0f };
            fixed (float* px = x)
            {
                Kernels.Softmax(px, 3);
            }

            // exp(1-3) + exp(2-3) + exp(3-3) = e^-2 + e^-1 + 1 = 0.135 + 0.367 + 1 = 1.503
            // p[2] = 1 / 1.503 = 0.665

            Assert.True(x[2] > x[1]);
            Assert.True(x[1] > x[0]);
            Assert.Equal(1.0f, x[0] + x[1] + x[2], 0.001f);
        }
    }
}
