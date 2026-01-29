using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading.Tasks;

namespace Gemma3.Sharp
{
    public class ModelDownloader
    {
        private readonly HttpClient _client;
        private readonly string _token;

        public ModelDownloader(string? token = null)
        {
            _client = new HttpClient();
            _token = token ?? Environment.GetEnvironmentVariable("HF_TOKEN") ?? "";

            if (!string.IsNullOrEmpty(_token))
            {
                _client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _token);
            }
        }

        public async Task DownloadModelAsync(string repoId, string outputDir)
        {
            Console.WriteLine($"Downloading {repoId} to {outputDir}...");
            Directory.CreateDirectory(outputDir);

            // 1. Get file list
            var files = await GetRepoFilesAsync(repoId);

            foreach (var file in files)
            {
                if (ShouldDownload(file))
                {
                    await DownloadFileAsync(repoId, file, outputDir);
                }
            }

            Console.WriteLine("Download complete.");
        }

        private bool ShouldDownload(string filename)
        {
            // Filter files we care about
            if (filename.EndsWith(".safetensors")) return true;
            if (filename.EndsWith("tokenizer.model")) return true;
            if (filename.EndsWith("config.json")) return true;
            if (filename.EndsWith(".index.json")) return true;
            return false;
        }

        private async Task<List<string>> GetRepoFilesAsync(string repoId)
        {
            // Use HF API to list files
            // https://huggingface.co/api/models/google/gemma-3-4b-it/tree/main
            string url = $"https://huggingface.co/api/models/{repoId}/tree/main?recursive=true";
            var response = await _client.GetAsync(url);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var files = new List<string>();

            using (var doc = JsonDocument.Parse(json))
            {
                foreach (var item in doc.RootElement.EnumerateArray())
                {
                    if (item.TryGetProperty("path", out var pathEl))
                    {
                        files.Add(pathEl.GetString() ?? "");
                    }
                }
            }

            return files;
        }

        private async Task DownloadFileAsync(string repoId, string filename, string outputDir)
        {
            string url = $"https://huggingface.co/{repoId}/resolve/main/{filename}";
            string localPath = Path.Combine(outputDir, filename);

            string? dir = Path.GetDirectoryName(localPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            if (File.Exists(localPath))
            {
                Console.WriteLine($"Skipping {filename} (already exists)");
                // In a real implementation we would check size/sha
                return;
            }

            Console.WriteLine($"Downloading {filename}...");

            using (var response = await _client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead))
            {
                response.EnsureSuccessStatusCode();
                using (var stream = await response.Content.ReadAsStreamAsync())
                using (var fileStream = new FileStream(localPath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    await stream.CopyToAsync(fileStream);
                }
            }
        }
    }
}
