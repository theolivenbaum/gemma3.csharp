using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Collections.Generic;
using Xunit;
using Gemma3.Sharp;

namespace Gemma3.Sharp.Tests
{
    public class IntegrationTests
    {
        private string CreateDummyTokenizer()
        {
            string path = Path.Combine(Path.GetTempPath(), "tokenizer.model");
            // Use the logic from TokenizerTests, duplicated here for simplicity or make helper public
            // I'll just write a minimal one.
            using (var fs = File.Create(path))
            {
                // Write minimal valid SentencePiece model
                // Needs at least special tokens to avoid crash in Load
                 void WriteVarint(Stream s, ulong val)
                {
                    while (val >= 0x80)
                    {
                        s.WriteByte((byte)((val & 0x7F) | 0x80));
                        val >>= 7;
                    }
                    s.WriteByte((byte)val);
                }
                void WriteString(Stream s, int field, string str)
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(str);
                    ulong tag = ((ulong)field << 3) | 2;
                    WriteVarint(s, tag);
                    WriteVarint(s, (ulong)bytes.Length);
                    s.Write(bytes, 0, bytes.Length);
                }
                void WritePiece(Stream s, string piece)
                {
                     using (var ms = new MemoryStream())
                    {
                        WriteString(ms, 1, piece);
                        // WriteFloat(ms, 2, 0f);
                        // WriteType(ms, 1);
                        byte[] payload = ms.ToArray();
                        ulong tag = (1UL << 3) | 2;
                        WriteVarint(s, tag);
                        WriteVarint(s, (ulong)payload.Length);
                        s.Write(payload, 0, payload.Length);
                    }
                }

                WritePiece(fs, "<pad>");
                WritePiece(fs, "<eos>");
                WritePiece(fs, "<bos>");
                WritePiece(fs, "<unk>");
                WritePiece(fs, "Hello");
                // Add enough dummy tokens to reach StartTurn/EndTurn logic if needed
                // But Tokenizer.Load looks for specific strings.
                WritePiece(fs, "<start_of_turn>");
                WritePiece(fs, "<end_of_turn>");
            }
            return path;
        }

        private void CreateDummySafetensors(string dir)
        {
            string path = Path.Combine(dir, "model.safetensors");

            // Create a small data buffer (e.g. 1MB) filled with zeros (or random)
            long dataSize = 1024 * 1024;

            // Generate header
            var header = new Dictionary<string, object>();

            // Helper to add tensor entry
            void AddTensor(string name)
            {
                header[name] = new
                {
                    dtype = "BF16",
                    shape = new[] { 1, 1 }, // Dummy shape
                    data_offsets = new[] { 0, dataSize } // Point to entire buffer
                };
            }

            AddTensor("language_model.model.embed_tokens.weight");
            AddTensor("language_model.model.norm.weight");

            for (int l = 0; l < 34; l++)
            {
                string prefix = $"language_model.model.layers.{l}.";
                AddTensor(prefix + "input_layernorm.weight");
                AddTensor(prefix + "self_attn.q_proj.weight");
                AddTensor(prefix + "self_attn.k_proj.weight");
                AddTensor(prefix + "self_attn.v_proj.weight");
                AddTensor(prefix + "self_attn.o_proj.weight");
                AddTensor(prefix + "self_attn.q_norm.weight");
                AddTensor(prefix + "self_attn.k_norm.weight");
                AddTensor(prefix + "post_attention_layernorm.weight");
                AddTensor(prefix + "mlp.gate_proj.weight");
                AddTensor(prefix + "mlp.up_proj.weight");
                AddTensor(prefix + "mlp.down_proj.weight");
                AddTensor(prefix + "pre_feedforward_layernorm.weight");
                AddTensor(prefix + "post_feedforward_layernorm.weight");
            }

            string headerJson = JsonSerializer.Serialize(header);
            byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);
            ulong headerLen = (ulong)headerBytes.Length;

            using (var fs = File.Create(path))
            {
                fs.Write(BitConverter.GetBytes(headerLen)); // 8 bytes
                fs.Write(headerBytes);

                // Create sparse file large enough for the biggest tensor (Embeddings = 1.3GB)
                // We'll set it to 2GB to be safe.
                fs.SetLength(2L * 1024 * 1024 * 1024);
            }
        }

        [Fact]
        public void TestFullFlow()
        {
            string modelDir = Path.Combine(Path.GetTempPath(), "gemma3_test_" + Guid.NewGuid());
            Directory.CreateDirectory(modelDir);

            try
            {
                CreateDummyTokenizer();
                // Move tokenizer to modelDir
                File.Move(Path.Combine(Path.GetTempPath(), "tokenizer.model"), Path.Combine(modelDir, "tokenizer.model"));

                CreateDummySafetensors(modelDir);

                using (var gemma = Gemma3.Load(modelDir, maxContext: 128))
                {
                    Assert.NotNull(gemma);

                    // Generate
                    // This will produce garbage because weights are zero.
                    // But it shouldn't crash.
                    // "Hello" will tokenize to <unk> probably or "Hello" if present.
                    // All weights 0 -> logits 0 -> probabilities uniform.
                    // Sampling might pick anything.

                    string output = gemma.Generate("Hello", new Gemma3GenParams { MaxTokens = 5 });
                    Assert.NotNull(output);
                    // Assert.NotEmpty(output); // Might be empty if EOS selected immediately
                }
            }
            finally
            {
                if (Directory.Exists(modelDir)) Directory.Delete(modelDir, true);
            }
        }
    }
}
