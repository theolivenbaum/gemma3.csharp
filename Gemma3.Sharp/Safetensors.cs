using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text.Json;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace Gemma3.Sharp
{
    public unsafe class SafeTensorsContext : IDisposable
    {
        private class MappedFile : IDisposable
        {
            public MemoryMappedFile File { get; }
            public MemoryMappedViewAccessor Accessor { get; }
            public byte* Pointer { get; }
            public long HeaderSize { get; }
            public long DataOffset { get; }

            public MappedFile(string path)
            {
                File = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
                Accessor = File.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
                byte* ptr = null;
                Accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
                Pointer = ptr;

                // Read header size (8 bytes, LE uint64)
                ulong headerSize = *(ulong*)ptr;
                HeaderSize = (long)headerSize;
                DataOffset = 8 + HeaderSize;
            }

            public void Dispose()
            {
                Accessor.SafeMemoryMappedViewHandle.ReleasePointer();
                Accessor.Dispose();
                File.Dispose();
            }
        }

        private readonly List<MappedFile> _files = new();
        private readonly Dictionary<string, TensorInfo> _tensors = new();

        public struct TensorInfo
        {
            public string Name;
            public string Dtype;
            public long[] Shape;
            public long DataOffset;
            public long DataSize;
            public int FileIndex;
        }

        public static SafeTensorsContext Load(string modelDir)
        {
            var ctx = new SafeTensorsContext();
            try
            {
                var files = Directory.GetFiles(modelDir, "*.safetensors");
                Array.Sort(files); // Ensure consistent order

                for (int i = 0; i < files.Length; i++)
                {
                    var mf = new MappedFile(files[i]);
                    ctx._files.Add(mf);

                    // Parse header
                    var headerSpan = new ReadOnlySpan<byte>(mf.Pointer + 8, (int)mf.HeaderSize);

                    // Copy header to managed array for JsonDocument
                    byte[] headerBytes = headerSpan.ToArray();

                    using (var doc = JsonDocument.Parse(headerBytes))
                    {
                        foreach (var prop in doc.RootElement.EnumerateObject())
                        {
                            if (prop.Name == "__metadata__") continue;

                            var info = new TensorInfo
                            {
                                Name = prop.Name,
                                FileIndex = i
                            };

                            if (prop.Value.TryGetProperty("dtype", out var dtypeEl))
                                info.Dtype = dtypeEl.GetString() ?? "unknown";

                            if (prop.Value.TryGetProperty("shape", out var shapeEl))
                            {
                                var shapeList = new List<long>();
                                foreach (var dim in shapeEl.EnumerateArray())
                                    shapeList.Add(dim.GetInt64());
                                info.Shape = shapeList.ToArray();
                            }

                            if (prop.Value.TryGetProperty("data_offsets", out var offsetsEl))
                            {
                                long start = offsetsEl[0].GetInt64();
                                long end = offsetsEl[1].GetInt64();
                                info.DataOffset = start;
                                info.DataSize = end - start;
                            }

                            ctx._tensors[info.Name] = info;
                        }
                    }
                }
            }
            catch
            {
                ctx.Dispose();
                throw;
            }
            return ctx;
        }

        public ReadOnlySpan<byte> GetTensorData(string name)
        {
            if (!_tensors.TryGetValue(name, out var info))
                return ReadOnlySpan<byte>.Empty;

            var file = _files[info.FileIndex];
            // Data offset in JSON is relative to the end of header
            byte* start = file.Pointer + file.DataOffset + info.DataOffset;

            // Check bounds (sanity check)
            // long maxLen = file.Accessor.Capacity - (file.DataOffset + info.DataOffset);
            // if (info.DataSize > maxLen) throw new Exception("Tensor out of bounds");

            return new ReadOnlySpan<byte>(start, (int)info.DataSize);
        }

        public ReadOnlySpan<ushort> GetTensorBF16(string name)
        {
             var span = GetTensorData(name);
             if (span.IsEmpty) return ReadOnlySpan<ushort>.Empty;
             return MemoryMarshal.Cast<byte, ushort>(span);
        }

        public void Dispose()
        {
            foreach (var f in _files) f.Dispose();
            _files.Clear();
        }
    }

    public unsafe class Gemma3Weights
    {
        // Pointers to BF16 data
        public ReadOnlySpan<ushort> EmbedTokens => _ctx.GetTensorBF16("language_model.model.embed_tokens.weight");
        public ReadOnlySpan<ushort> Norm => _ctx.GetTensorBF16("language_model.model.norm.weight");

        public struct Layer
        {
            private SafeTensorsContext _ctx;
            private int _idx;

            public Layer(SafeTensorsContext ctx, int idx) { _ctx = ctx; _idx = idx; }

            private ReadOnlySpan<ushort> Get(string suffix) => _ctx.GetTensorBF16($"language_model.model.layers.{_idx}.{suffix}.weight");

            public ReadOnlySpan<ushort> InputLayernorm => Get("input_layernorm");
            public ReadOnlySpan<ushort> QProj => Get("self_attn.q_proj");
            public ReadOnlySpan<ushort> KProj => Get("self_attn.k_proj");
            public ReadOnlySpan<ushort> VProj => Get("self_attn.v_proj");
            public ReadOnlySpan<ushort> OProj => Get("self_attn.o_proj");
            public ReadOnlySpan<ushort> QNorm => Get("self_attn.q_norm");
            public ReadOnlySpan<ushort> KNorm => Get("self_attn.k_norm");
            public ReadOnlySpan<ushort> PostAttentionLayernorm => Get("post_attention_layernorm");
            public ReadOnlySpan<ushort> GateProj => Get("mlp.gate_proj");
            public ReadOnlySpan<ushort> UpProj => Get("mlp.up_proj");
            public ReadOnlySpan<ushort> DownProj => Get("mlp.down_proj");
            public ReadOnlySpan<ushort> PreFeedforwardLayernorm => Get("pre_feedforward_layernorm");
            public ReadOnlySpan<ushort> PostFeedforwardLayernorm => Get("post_feedforward_layernorm");
        }

        public Layer[] Layers { get; }
        private SafeTensorsContext _ctx;

        public Gemma3Weights(SafeTensorsContext ctx)
        {
            _ctx = ctx;
            Layers = new Layer[34]; // Gemma 3 4B has 34 layers
            for (int i = 0; i < 34; i++)
            {
                Layers[i] = new Layer(ctx, i);
            }
        }
    }
}
