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

            // Prefill
            float* logits = _transformer.Buffers.Logits;
            int pos = 0;

            for (int i = 0; i < tokens.Length - 1; i++)
            {
                _transformer.Forward(logits, tokens[i], pos++);
            }

            int nextToken = tokens[tokens.Length - 1];

            for (int i = 0; i < params_.MaxTokens; i++)
            {
                _transformer.Forward(logits, nextToken, pos++);

                // Sample
                // Apply temp
                if (params_.Temperature > 0)
                {
                    for(int j=0; j<Gemma3Config.Default.VocabSize; j++) logits[j] /= params_.Temperature;
                }

                // Softmax
                Kernels.Softmax(logits, Gemma3Config.Default.VocabSize);

                // TopK
                if (params_.TopK > 0) Kernels.SampleTopK(logits, Gemma3Config.Default.VocabSize, params_.TopK);

                // TopP
                if (params_.TopP < 1.0f) Kernels.SampleTopP(logits, Gemma3Config.Default.VocabSize, params_.TopP);

                // Sample
                if (params_.Temperature <= 0)
                    nextToken = Kernels.ArgMax(logits, Gemma3Config.Default.VocabSize);
                else
                    nextToken = Kernels.Sample(logits, Gemma3Config.Default.VocabSize);

                if (nextToken == _tokenizer.EosId || nextToken == _tokenizer.EndTurnId) break;

                string piece = _tokenizer.Decode(new[] { nextToken });
                sb.Append(piece);
                onToken?.Invoke(piece);
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
