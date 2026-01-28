using System;
using System.IO;
using System.Text;
using Xunit;
using Gemma3.Sharp;

namespace Gemma3.Sharp.Tests
{
    public class TokenizerTests
    {
        private string CreateDummyModel()
        {
            string path = Path.GetTempFileName();
            using (var fs = File.OpenWrite(path))
            {
                // Helper to write varint
                void WriteVarint(Stream s, ulong val)
                {
                    while (val >= 0x80)
                    {
                        s.WriteByte((byte)((val & 0x7F) | 0x80));
                        val >>= 7;
                    }
                    s.WriteByte((byte)val);
                }

                // Helper to write string field
                void WriteString(Stream s, int field, string str)
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(str);
                    ulong tag = ((ulong)field << 3) | 2;
                    WriteVarint(s, tag);
                    WriteVarint(s, (ulong)bytes.Length);
                    s.Write(bytes, 0, bytes.Length);
                }

                // Helper to write float field
                void WriteFloat(Stream s, int field, float val)
                {
                    ulong tag = ((ulong)field << 3) | 5;
                    WriteVarint(s, tag);
                    s.Write(BitConverter.GetBytes(val), 0, 4);
                }

                // Helper to write type
                void WriteType(Stream s, int type)
                {
                    ulong tag = (3UL << 3) | 0;
                    WriteVarint(s, tag);
                    WriteVarint(s, (ulong)type);
                }

                // Write a SentencePiece message
                void WritePiece(Stream s, string piece, float score, int type)
                {
                    using (var ms = new MemoryStream())
                    {
                        WriteString(ms, 1, piece);
                        WriteFloat(ms, 2, score);
                        WriteType(ms, type);

                        byte[] payload = ms.ToArray();
                        ulong tag = (1UL << 3) | 2; // Field 1, wire 2
                        WriteVarint(s, tag);
                        WriteVarint(s, (ulong)payload.Length);
                        s.Write(payload, 0, payload.Length);
                    }
                }

                // Add standard tokens
                WritePiece(fs, "<pad>", 0, 3); // 0
                WritePiece(fs, "<eos>", 0, 3); // 1
                WritePiece(fs, "<bos>", 0, 3); // 2
                WritePiece(fs, "<unk>", 0, 3); // 3

                // Add some vocab
                // "Hello" -> " He", "ll", "o"
                WritePiece(fs, "\u2581H", -1.0f, 1);  // 4
                WritePiece(fs, "e", -1.0f, 1);        // 5
                WritePiece(fs, "l", -1.0f, 1);        // 6
                WritePiece(fs, "o", -1.0f, 1);        // 7

                WritePiece(fs, "\u2581He", 5.0f, 1);  // 8
                WritePiece(fs, "ll", 5.0f, 1);        // 9
                WritePiece(fs, "\u2581Hell", 8.0f, 1); // 10
                WritePiece(fs, "\u2581Hello", 10.0f, 1); // 11

                // Add special chat tokens
                WritePiece(fs, "<start_of_turn>", 0, 3);
                WritePiece(fs, "<end_of_turn>", 0, 3);
            }
            return path;
        }

        [Fact]
        public void TestLoadAndTokenize()
        {
            string path = CreateDummyModel();
            try
            {
                var tok = Gemma3Tokenizer.Load(path);
                Assert.NotNull(tok);
                Assert.Equal(0, tok.PadId);
                Assert.Equal(1, tok.EosId);
                Assert.Equal(2, tok.BosId);

                // Test "Hello" -> should use the merged token ID 11
                // Note: The logic handles spaces. "Hello" at start -> "\u2581Hello"
                var tokens = tok.Encode("Hello", false, false);
                Assert.Single(tokens);
                Assert.Equal(11, tokens[0]);

                // Test decode
                string decoded = tok.Decode(tokens);
                Assert.Equal("Hello", decoded);
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }
    }
}
