using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

/// Renders a domain-randomised training set from the SurgSeg organ scene: every sample gets a
/// random camera view, lighting, tissue colours/textures, backdrop, organ pose and instrument
/// pose, plus an exact label mask (unified 22-class ids, as SurgSegCapture writes them).
/// The point is to teach the model anatomy and context, not one simulator's look, so it keeps
/// working on any scene. Output: SurgSegCapture/<name>/{images,masks}/000000.png ...
///
/// Menu: SurgSeg > Generate Randomized Dataset
/// Batch: Unity.exe -batchmode -projectPath <proj> -executeMethod SurgSegDatasetGenerator.GenerateBatch -quit
///        (optional env SURGSEG_N=800, SURGSEG_W=640, SURGSEG_H=512, SURGSEG_NAME=randomized_v1)
public static class SurgSegDatasetGenerator
{
    [MenuItem("SurgSeg/Generate Randomized Dataset")]
    public static void GenerateMenu() => Generate(800, 640, 512, "randomized_v1", 1234);

    public static void GenerateBatch()
    {
        int Env(string k, int d) => int.TryParse(System.Environment.GetEnvironmentVariable(k), out var v) ? v : d;
        string name = System.Environment.GetEnvironmentVariable("SURGSEG_NAME");
        Generate(Env("SURGSEG_N", 800), Env("SURGSEG_W", 640), Env("SURGSEG_H", 512),
                 string.IsNullOrEmpty(name) ? "randomized_v1" : name, Env("SURGSEG_SEED", 1234));
    }

    static System.Random rng;
    static float R(float a, float b) => a + (float)rng.NextDouble() * (b - a);
    static bool P(float p) => rng.NextDouble() < p;

    public static void Generate(int count, int width, int height, string name, int seed)
    {
        rng = new System.Random(seed);
        SurgSegOrganSceneBuilder.Build();                       // fresh scene, saved before we touch it

        var cam = GameObject.Find("SurgSeg Endoscope").GetComponent<Camera>();
        var spot = GameObject.Find("Endoscope Light").GetComponent<Light>();
        var fill = GameObject.Find("Fill Light").GetComponent<Light>();
        var tool = GameObject.Find("PSM1");
        var back = GameObject.Find("fat_backdrop");
        var anatomy = GameObject.Find("SurgSeg Anatomy (mirrored to anatomical view)") ?? GameObject.Find("SurgSeg Anatomy");
        Vector3 focus = new Vector3(0f, 1.01f, 0f);
        Vector3 calot = Centre("cystic_duct", focus) * 0.5f + Centre("cystic_artery", focus) * 0.5f;
        Vector3 baseOffset = new Vector3(0f, -0.07f, 0.17f).normalized;

        // in-memory material copies: randomising never touches the saved assets
        var renderers = new List<Renderer>();
        var baseColor = new Dictionary<Renderer, Color>();
        foreach (var r in Object.FindObjectsOfType<Renderer>())
        {
            var mats = r.sharedMaterials;
            for (int i = 0; i < mats.Length; i++) if (mats[i] != null) mats[i] = new Material(mats[i]);
            r.sharedMaterials = mats;
            renderers.Add(r);
            baseColor[r] = mats.Length > 0 && mats[0] != null && mats[0].HasProperty("_Color") ? mats[0].color : Color.gray;
        }
        var noise = new List<Texture2D>();
        for (int i = 0; i < 12; i++) noise.Add(NoiseTexture(256, i));

        var idShader = Shader.Find("Hidden/SurgSegId");
        var idMats = new Dictionary<int, Material>();
        var rgbRT = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB) { antiAliasing = 4 };
        var idRT = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear) { antiAliasing = 1, filterMode = FilterMode.Point };
        var rgbTex = new Texture2D(width, height, TextureFormat.RGB24, false);
        var idTex = new Texture2D(width, height, TextureFormat.RGBA32, false, true);

        string dir = Path.Combine(Path.GetDirectoryName(Application.dataPath), "SurgSegCapture", name);
        Directory.CreateDirectory(Path.Combine(dir, "images"));
        Directory.CreateDirectory(Path.Combine(dir, "masks"));
        var anatomyPos = anatomy.transform.position; var anatomyRot = anatomy.transform.rotation; var anatomyScale = anatomy.transform.localScale;
        var classPixels = new long[256];
        int withVessel = 0;

        for (int n = 0; n < count; n++)
        {
            EditorUtility.DisplayProgressBar("SurgSeg dataset", $"{n}/{count}", (float)n / count);

            // organ group pose jitter, so positions cannot be memorised
            anatomy.transform.SetPositionAndRotation(anatomyPos, anatomyRot);
            anatomy.transform.localScale = anatomyScale * R(0.9f, 1.1f);
            anatomy.transform.RotateAround(focus, Vector3.up, R(-15f, 15f));
            anatomy.transform.RotateAround(focus, Vector3.right, R(-12f, 12f));
            anatomy.transform.RotateAround(focus, Vector3.forward, R(-12f, 12f));
            foreach (var r in renderers)
            {
                string on = r.gameObject.name.ToLowerInvariant();
                bool optional = on.Contains("stomach") || on.Contains("duodenum") || on.Contains("hepatic");
                r.enabled = !optional || P(0.8f);
            }

            // tissue appearance: half near the original colour, half anything at all
            foreach (var r in renderers)
            {
                if (r.transform.IsChildOf(tool.transform)) continue;
                foreach (var m in r.sharedMaterials)
                {
                    if (m == null || !m.HasProperty("_Color")) continue;
                    Color.RGBToHSV(baseColor[r], out var h, out var s, out var v);
                    m.color = P(0.5f)
                        ? Color.HSVToRGB(Mathf.Repeat(h + R(-0.06f, 0.06f), 1f), Mathf.Clamp01(s + R(-0.25f, 0.2f)), Mathf.Clamp01(v + R(-0.3f, 0.2f)))
                        : Color.HSVToRGB(R(0f, 1f), R(0.1f, 0.9f), R(0.25f, 1f));
                    if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", R(0.05f, 0.9f));
                    if (m.HasProperty("_MainTex"))
                    {
                        m.mainTexture = P(0.6f) ? noise[rng.Next(noise.Count)] : null;
                        m.mainTextureScale = Vector2.one * R(1f, 8f);
                        m.mainTextureOffset = new Vector2(R(0f, 1f), R(0f, 1f));
                    }
                }
            }
            back.SetActive(P(0.85f));
            back.transform.localScale = Vector3.one * R(0.4f, 1.2f);

            // camera: mostly on the hepatocystic triangle, sometimes anywhere in the field
            Vector3 target = P(0.5f) ? calot : P(0.6f) ? focus : focus + new Vector3(R(-0.05f, 0.05f), R(-0.04f, 0.05f), R(-0.02f, 0.02f));
            Vector3 dirToCam = Quaternion.Euler(R(-35f, 30f), R(-60f, 60f), 0f) * baseOffset;
            cam.transform.position = target + dirToCam * R(0.06f, 0.24f);
            cam.transform.LookAt(target, Vector3.up);
            cam.transform.Rotate(0f, 0f, R(-25f, 25f), Space.Self);
            cam.fieldOfView = R(50f, 85f);
            cam.backgroundColor = Color.HSVToRGB(R(0f, 1f), R(0f, 0.6f), R(0f, 0.25f));

            // lighting
            spot.intensity = R(0.5f, 2.2f);
            spot.color = Color.HSVToRGB(R(0.02f, 0.15f), R(0f, 0.25f), 1f);
            spot.transform.SetPositionAndRotation(cam.transform.position + cam.transform.right * R(-0.01f, 0.01f), cam.transform.rotation);
            fill.intensity = R(0f, 0.6f);
            fill.transform.rotation = Quaternion.Euler(R(10f, 80f), R(0f, 360f), 0f);
            RenderSettings.ambientLight = Color.HSVToRGB(R(0f, 0.1f), R(0.1f, 0.4f), R(0.05f, 0.35f));

            // instrument
            tool.SetActive(P(0.7f));
            if (tool.activeSelf)
            {
                tool.transform.position = target + cam.transform.right * R(-0.04f, 0.04f) + cam.transform.up * R(-0.03f, 0.03f) + cam.transform.forward * R(-0.02f, 0.01f);
                tool.transform.rotation = Quaternion.LookRotation(-(cam.transform.forward + cam.transform.right * R(-1f, 1f) + cam.transform.up * R(-0.5f, 0.5f)), cam.transform.up) * Quaternion.Euler(90f, 0f, 0f);
            }

            Render(cam, rgbRT, rgbTex, width, height);
            RenderIds(cam, renderers, idShader, idMats, idRT, idTex, width, height);

            var px = idTex.GetPixels32();
            bool vessel = false;
            foreach (var c in px) { classPixels[c.r]++; if (c.r == 12 || c.r == 14) vessel = true; }
            if (vessel) withVessel++;
            string file = $"{n:D6}.png";
            File.WriteAllBytes(Path.Combine(dir, "images", file), rgbTex.EncodeToPNG());
            File.WriteAllBytes(Path.Combine(dir, "masks", file), idTex.EncodeToPNG());
        }
        EditorUtility.ClearProgressBar();
        rgbRT.Release(); idRT.Release();

        long total = 0; foreach (var c in classPixels) total += c;
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"samples {count}  size {width}x{height}  with duct/artery visible {withVessel}");
        for (int i = 0; i < 22; i++) if (classPixels[i] > 0) sb.AppendLine($"class {i,2} {(SurgClass)i,-18} {100.0 * classPixels[i] / total,7:0.00}%");
        File.WriteAllText(Path.Combine(dir, "stats.txt"), sb.ToString());
        Debug.Log("SURGSEG DATASET DONE " + dir + "\n" + sb);
    }

    static Vector3 Centre(string objName, Vector3 fallback)
    {
        var g = GameObject.Find(objName);
        var r = g != null ? g.GetComponent<Renderer>() : null;
        return r != null ? r.bounds.center : fallback;
    }

    static void Render(Camera cam, RenderTexture rt, Texture2D tex, int w, int h)
    {
        var prev = cam.targetTexture;
        cam.targetTexture = rt;
        cam.Render();
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        tex.Apply();
        RenderTexture.active = null;
        cam.targetTexture = prev;
    }

    static void RenderIds(Camera cam, List<Renderer> renderers, Shader idShader, Dictionary<int, Material> idMats,
                          RenderTexture rt, Texture2D tex, int w, int h)
    {
        var swapped = new List<(Renderer, Material[])>();
        var clear = cam.clearFlags; var bg = cam.backgroundColor; bool hdr = cam.allowHDR, msaa = cam.allowMSAA;
        try
        {
            foreach (var r in renderers)
            {
                if (!r.enabled || !r.gameObject.activeInHierarchy) continue;
                var l = r.GetComponentInParent<SurgSegLabel>();
                int id = l != null ? (int)l.label : 0;
                if (!idMats.TryGetValue(id, out var m)) { m = new Material(idShader); m.SetFloat("_SegId", id); idMats[id] = m; }
                swapped.Add((r, r.sharedMaterials));
                var mats = new Material[Mathf.Max(1, r.sharedMaterials.Length)];
                for (int i = 0; i < mats.Length; i++) mats[i] = m;
                r.sharedMaterials = mats;
            }
            cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = Color.black;
            cam.allowHDR = false; cam.allowMSAA = false;
            Render(cam, rt, tex, w, h);
        }
        finally
        {
            foreach (var (r, mats) in swapped) r.sharedMaterials = mats;
            cam.clearFlags = clear; cam.backgroundColor = bg; cam.allowHDR = hdr; cam.allowMSAA = msaa;
        }
    }

    static Texture2D NoiseTexture(int size, int seed)
    {
        var t = new Texture2D(size, size, TextureFormat.RGB24, true) { wrapMode = TextureWrapMode.Repeat };
        float f1 = 2f + seed % 5, f2 = 9f + seed * 1.7f, ox = seed * 13.1f, oy = seed * 7.3f;
        bool veins = seed % 3 == 0;
        var px = new Color[size * size];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                float u = (float)x / size, v = (float)y / size;
                float n = 0.6f * Mathf.PerlinNoise(ox + u * f1, oy + v * f1) + 0.4f * Mathf.PerlinNoise(oy + u * f2, ox + v * f2);
                if (veins) n *= 0.55f + 0.45f * Mathf.Abs(Mathf.Sin((u + 0.3f * Mathf.PerlinNoise(u * 4 + ox, v * 4)) * 40f));
                float g = Mathf.Lerp(0.55f, 1.15f, n);
                px[y * size + x] = new Color(g, g, g);
            }
        t.SetPixels(px);
        t.Apply();
        return t;
    }
}
