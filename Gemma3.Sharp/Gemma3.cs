using System;
using System.IO;
using System.Collections.Generic;
using System.Text;

namespace Gemma3.Sharp
{
    public class Gemma3GenParams
    {
        public int MaxTokens { get; set; } = 512;
        public float Temperature { get; set; } = 0.7f;
        public int TopK { get; set; } = 50;
        public float TopP { get; set; } = 0.9f;
        public int Seed { get; set; } = -1;
        public bool Verbose { get; set; } = false;
    }

    public class Gemma3 : IDisposable
    {
        private SafeTensorsContext _ctx;
        private Gemma3Weights _weights;
        private Gemma3Tokenizer _tokenizer;
        private Gemma3Transformer _transformer;

        public Gemma3Tokenizer Tokenizer => _tokenizer;

        private Gemma3(string modelDir, int maxContext)
        {
            string tokenizerPath = Path.Combine(modelDir, "tokenizer.model");
            if (!File.Exists(tokenizerPath))
                throw new FileNotFoundException("Tokenizer model not found", tokenizerPath);

            _tokenizer = Gemma3Tokenizer.Load(tokenizerPath);
            _ctx = SafeTensorsContext.Load(modelDir);
            _weights = new Gemma3Weights(_ctx);
            _transformer = new Gemma3Transformer(_weights, Gemma3Config.Default, maxContext);
        }

        public static Gemma3 Load(string modelDir, int maxContext = 8192)
        {
            return new Gemma3(modelDir, maxContext);
        }

        public unsafe string Generate(string prompt, Gemma3GenParams? parameters = null, Action<string>? onToken = null)
        {
            var params_ = parameters ?? new Gemma3GenParams();
            int[] tokens = _tokenizer.Encode(prompt, true, false);

            _transformer.Cache.Reset();

            var sb = new StringBuilder();
            float* logits = _transformer.Buffers.Logits;
            int vocabSize = _transformer.Config.VocabSize;

            if (tokens.Length == 0) return "";

            // Prefill full prompt
            _transformer.Prefill(logits, tokens, 0);
            int pos = _transformer.Cache.CurrentPos;

            int nextToken = 0;

            for (int i = 0; i < params_.MaxTokens; i++)
            {
                // Sampling Pipeline
                // 1. Temp
                if (params_.Temperature > 0)
                {
                    float invTemp = 1.0f / params_.Temperature;
                    Kernels.VecScale(logits, logits, invTemp, vocabSize);
                }

                // 2. TopK
                if (params_.TopK > 0) Kernels.SampleTopK(logits, vocabSize, params_.TopK);

                // 3. TopP
                if (params_.TopP < 1.0f) Kernels.SampleTopP(logits, vocabSize, params_.TopP);

                // 4. Softmax (Logits -> Probs)
                Kernels.Softmax(logits, vocabSize);

                // 5. Sample
                if (params_.Temperature <= 0)
                    nextToken = Kernels.ArgMax(logits, vocabSize);
                else
                    nextToken = Kernels.Sample(logits, vocabSize);

                if (nextToken == _tokenizer.EosId || nextToken == _tokenizer.EndTurnId) break;

                string piece = _tokenizer.Decode(new[] { nextToken });
                sb.Append(piece);
                onToken?.Invoke(piece);

                // Forward next token
                if (i < params_.MaxTokens - 1)
                {
                    _transformer.Forward(logits, nextToken, pos++);
                }
            }

            return sb.ToString();
        }

        public void Dispose()
        {
            _transformer.Dispose();
            _ctx.Dispose();
        }
    }
}
