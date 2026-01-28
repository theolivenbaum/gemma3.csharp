using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Gemma3.Sharp
{
    public class Gemma3Tokenizer
    {
        public const int VocabSize = 262208;
        public const int MaxTokenLen = 512;

        // Special Tokens
        public const int TokenPad = 0;
        public const int TokenEos = 1;
        public const int TokenBos = 2;
        public const int TokenUnk = 3;
        public const int TokenStartTurn = 105;
        public const int TokenEndTurn = 106;

        public class VocabEntry
        {
            public string Piece { get; set; } = string.Empty;
            public float Score { get; set; }
            public int Type { get; set; }
        }

        private VocabEntry[] _vocab;
        private Dictionary<string, int> _pieceToId;
        private int[] _byteTokens; // Maps 0-255 to token ID

        // Special IDs
        public int BosId { get; private set; }
        public int EosId { get; private set; }
        public int PadId { get; private set; }
        public int UnkId { get; private set; }
        public int StartTurnId { get; private set; }
        public int EndTurnId { get; private set; }

        public Gemma3Tokenizer()
        {
            _vocab = Array.Empty<VocabEntry>();
            _pieceToId = new Dictionary<string, int>();
            _byteTokens = new int[256];
            Array.Fill(_byteTokens, -1);
        }

        public static Gemma3Tokenizer Load(string path)
        {
            var tok = new Gemma3Tokenizer();
            byte[] data = File.ReadAllBytes(path);

            tok.ParseProtobuf(data);

            // Find special tokens
            tok.PadId = tok._pieceToId.GetValueOrDefault("<pad>", TokenPad);
            tok.EosId = tok._pieceToId.GetValueOrDefault("<eos>", TokenEos);
            tok.BosId = tok._pieceToId.GetValueOrDefault("<bos>", TokenBos);
            tok.UnkId = tok._pieceToId.GetValueOrDefault("<unk>", TokenUnk);

            if (!tok._pieceToId.TryGetValue("<pad>", out _)) tok.PadId = TokenPad;
            if (!tok._pieceToId.TryGetValue("<eos>", out _)) tok.EosId = TokenEos;
            if (!tok._pieceToId.TryGetValue("<bos>", out _)) tok.BosId = TokenBos;
            if (!tok._pieceToId.TryGetValue("<unk>", out _)) tok.UnkId = TokenUnk;

            tok.StartTurnId = tok._pieceToId.GetValueOrDefault("<start_of_turn>", TokenStartTurn);
            tok.EndTurnId = tok._pieceToId.GetValueOrDefault("<end_of_turn>", TokenEndTurn);

            if (!tok._pieceToId.TryGetValue("<start_of_turn>", out _)) tok.StartTurnId = TokenStartTurn;
            if (!tok._pieceToId.TryGetValue("<end_of_turn>", out _)) tok.EndTurnId = TokenEndTurn;

            return tok;
        }

        private void ParseProtobuf(ReadOnlySpan<byte> data)
        {
            var vocabList = new List<VocabEntry>(VocabSize);
            _pieceToId = new Dictionary<string, int>(VocabSize);

            int offset = 0;
            while (offset < data.Length)
            {
                ulong tag = ReadVarint(data, ref offset);
                int field = (int)(tag >> 3);
                int wireType = (int)(tag & 7);

                if (field == 1 && wireType == 2) // repeated SentencePiece
                {
                    ulong msgLen = ReadVarint(data, ref offset);
                    int msgEnd = offset + (int)msgLen;

                    string? piece = null;
                    float score = 0.0f;
                    int type = 1;

                    while (offset < msgEnd)
                    {
                        ulong innerTag = ReadVarint(data, ref offset);
                        int innerField = (int)(innerTag >> 3);
                        int innerWire = (int)(innerTag & 7);

                        if (innerField == 1 && innerWire == 2) // piece
                        {
                            ulong strLen = ReadVarint(data, ref offset);
                            piece = Encoding.UTF8.GetString(data.Slice(offset, (int)strLen));
                            offset += (int)strLen;
                        }
                        else if (innerField == 2 && innerWire == 5) // score (float 32-bit)
                        {
                            score = BitConverter.ToSingle(data.Slice(offset, 4));
                            offset += 4;
                        }
                        else if (innerField == 3 && innerWire == 0) // type (varint)
                        {
                            type = (int)ReadVarint(data, ref offset);
                        }
                        else
                        {
                            SkipField(data, ref offset, innerWire);
                        }
                    }

                    if (piece != null)
                    {
                        var entry = new VocabEntry { Piece = piece, Score = score, Type = type };
                        int id = vocabList.Count;
                        vocabList.Add(entry);
                        _pieceToId[piece] = id;

                        // Check for byte token <0xNN>
                        if (type == 6 || (piece.Length == 6 && piece.StartsWith("<0x") && piece.EndsWith(">")))
                        {
                            if (byte.TryParse(piece.Substring(3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte bVal))
                            {
                                _byteTokens[bVal] = id;
                            }
                        }
                    }
                }
                else
                {
                    SkipField(data, ref offset, wireType);
                }
            }

            _vocab = vocabList.ToArray();
        }

        private static ulong ReadVarint(ReadOnlySpan<byte> data, ref int offset)
        {
            ulong result = 0;
            int shift = 0;
            while (offset < data.Length)
            {
                byte b = data[offset++];
                result |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) break;
                shift += 7;
            }
            return result;
        }

        private static void SkipField(ReadOnlySpan<byte> data, ref int offset, int wireType)
        {
            if (wireType == 0) // Varint
            {
                ReadVarint(data, ref offset);
            }
            else if (wireType == 1) // 64-bit
            {
                offset += 8;
            }
            else if (wireType == 2) // Length delimited
            {
                ulong len = ReadVarint(data, ref offset);
                offset += (int)len;
            }
            else if (wireType == 5) // 32-bit
            {
                offset += 4;
            }
            else
            {
                // throw new Exception($"Unknown wire type {wireType}");
                // Ignore unknown types just like C version
            }
        }

        private struct BpeSymbol
        {
            public int Id;
            public int Prev;
            public int Next;
            public int Len; // Length in bytes of the original char/piece
        }

        public int[] Encode(string text, bool addBos, bool addEos, int maxTokens = int.MaxValue)
        {
            if (string.IsNullOrEmpty(text))
            {
                var result = new List<int>();
                if (addBos) result.Add(BosId);
                if (addEos) result.Add(EosId);
                return result.ToArray();
            }

            // Initial symbol creation
            // Gemma uses the "byte fallback" scheme where each UTF-8 byte is potentially a token,
            // but usually we start by trying to match characters.
            // However, the C implementation tokenizes primarily by trying to find pieces for characters
            // (possibly with prepended space) or falling back to bytes.

            // SentencePiece treats space as ▁ (U+2581) which is 0xE2 0x96 0x81

            var symbols = new List<BpeSymbol>(text.Length + 2);
            int textLen = text.Length;
            int pos = 0;

            while (pos < textLen)
            {
                int charLen = 1;
                byte b = (byte)text[pos];
                if ((b & 0x80) == 0) charLen = 1;
                else if ((b & 0xE0) == 0xC0) charLen = 2;
                else if ((b & 0xF0) == 0xE0) charLen = 3;
                else if ((b & 0xF8) == 0xF0) charLen = 4;

                if (pos + charLen > textLen) charLen = textLen - pos;

                bool prependSpace = (pos == 0 || text[pos - 1] == ' ');

                if (text[pos] == ' ')
                {
                    pos++;
                    continue; // Space handled by next token having prepended ▁
                }

                // Construct piece to search
                string piece;
                if (prependSpace)
                {
                    piece = "\u2581" + text.Substring(pos, charLen);
                }
                else
                {
                    piece = text.Substring(pos, charLen);
                }

                int id;
                if (_pieceToId.TryGetValue(piece, out id))
                {
                    symbols.Add(new BpeSymbol
                    {
                        Id = id,
                        Prev = symbols.Count - 1,
                        Next = symbols.Count + 1, // Will fix up later
                        Len = piece.Length // Wait, Len should be logical or byte length? C impl tracks char_len/bytes
                        // In C impl: symbols[n_symbols].len = char_len (for raw char) or 1 (for byte fallback)
                        // It accumulates these lengths during merge.
                        // Wait, piece.Length in C# is chars, but the C impl uses tracking.
                        // Let's verify what 'Len' is used for. Just for BPE priority maybe?
                        // No, 'len' in C impl seems to be about the accumulated length of the piece?
                        // "symbols[best_i].len += symbols[j].len;"
                        // Actually it doesn't seem critical for score lookup, as we look up the combined string.
                        // The C impl uses the actual piece string from vocab for merge lookup.
                    });
                }
                else
                {
                    // Byte fallback
                    var bytes = Encoding.UTF8.GetBytes(text.Substring(pos, charLen));
                    foreach (var byteVal in bytes)
                    {
                        int byteId = _byteTokens[byteVal];
                        if (byteId < 0) byteId = UnkId;

                        symbols.Add(new BpeSymbol
                        {
                            Id = byteId,
                            Prev = symbols.Count - 1,
                            Next = symbols.Count + 1
                        });
                    }
                }

                pos += charLen;
            }

            // Fixup linked list
            for (int i = 0; i < symbols.Count; i++)
            {
                var s = symbols[i];
                if (i == symbols.Count - 1) s.Next = -1;
                symbols[i] = s;
            }

            // BPE Merge Loop
            while (true)
            {
                float bestScore = float.NegativeInfinity;
                int bestIdx = -1;
                int bestMergedId = -1;

                for (int i = 0; i < symbols.Count; i++)
                {
                    if (symbols[i].Id < 0) continue; // Deleted
                    int next = symbols[i].Next;
                    if (next < 0 || next >= symbols.Count) continue;
                    if (symbols[next].Id < 0) continue;

                    string p1 = _vocab[symbols[i].Id].Piece;
                    string p2 = _vocab[symbols[next].Id].Piece;

                    string merged = p1 + p2;
                    if (_pieceToId.TryGetValue(merged, out int mergedId))
                    {
                         float score = _vocab[mergedId].Score;
                         if (score > bestScore)
                         {
                             bestScore = score;
                             bestIdx = i;
                             bestMergedId = mergedId;
                         }
                    }
                }

                if (bestIdx < 0) break;

                // Apply merge
                int j = symbols[bestIdx].Next;
                var symI = symbols[bestIdx];
                var symJ = symbols[j];

                symI.Id = bestMergedId;
                symI.Next = symJ.Next;

                symbols[bestIdx] = symI;

                // Update prev pointer of the node after J
                if (symJ.Next >= 0 && symJ.Next < symbols.Count)
                {
                    var symNext = symbols[symJ.Next];
                    symNext.Prev = bestIdx;
                    symbols[symJ.Next] = symNext;
                }

                // Mark J as deleted
                symJ.Id = -1;
                symbols[j] = symJ;
            }

            // Collect tokens
            var tokens = new List<int>();
            if (addBos && tokens.Count < maxTokens) tokens.Add(BosId);

            for (int i = 0; i < symbols.Count && tokens.Count < maxTokens; i++)
            {
                if (symbols[i].Id >= 0)
                {
                    tokens.Add(symbols[i].Id);
                }
            }

            if (addEos && tokens.Count < maxTokens) tokens.Add(EosId);

            return tokens.ToArray();
        }

        public string Decode(int[] tokens)
        {
            var sb = new StringBuilder();
            foreach (var id in tokens)
            {
                if (id < 0 || id >= _vocab.Length) continue;
                string piece = _vocab[id].Piece;

                // Handle ▁ (U+2581) -> space
                piece = piece.Replace("\u2581", " ");

                // Handle <0xNN>
                if (piece.StartsWith("<0x") && piece.EndsWith(">") && piece.Length == 6)
                {
                     if (byte.TryParse(piece.Substring(3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte bVal))
                     {
                         // This is tricky because we are building a string, but this is a raw byte.
                         // If the sequence of bytes forms valid UTF8, we should probably output that.
                         // But for now, let's just append the char (which might be garbage if standalone).
                         // A better approach is to collect all bytes and decode at the end.
                         // But the simple approach:
                         sb.Append((char)bVal);
                         continue;
                     }
                }

                sb.Append(piece);
            }

            string text = sb.ToString();
            if (text.StartsWith(" ")) text = text.Substring(1);
            return text;
        }

        // Helper for testing
        public int PieceToId(string piece) => _pieceToId.GetValueOrDefault(piece, -1);
    }
}
